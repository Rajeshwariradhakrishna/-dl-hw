import torch
import torch.optim as optim
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from homework.datasets.drive_dataset import load_data
from homework.models import Detector, HOMEWORK_DIR
from torch.optim.lr_scheduler import CosineAnnealingLR

# Define log directory
log_dir = str(HOMEWORK_DIR)
os.makedirs(log_dir, exist_ok=True)


def save_model(model, model_name, log_dir):
    model_path = os.path.join(log_dir, f"{model_name}.th")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


# IoU Loss for Segmentation
class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.softmax(preds, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float()
        intersection = (preds * targets_one_hot).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3)) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou.mean()


# Gradient Loss for Boundary Prediction
class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, preds, targets):
        # Add a dummy channel dimension to targets
        targets = targets.unsqueeze(1).float()  # (B, H, W) -> (B, 1, H, W)

        # Compute gradients for predictions and targets
        preds_grad_x = torch.abs(torch.gradient(preds, dim=2)[0])  # Gradient along height (dim=2)
        preds_grad_y = torch.abs(torch.gradient(preds, dim=3)[0])  # Gradient along width (dim=3)
        preds_grad = preds_grad_x + preds_grad_y  # Combine gradients

        targets_grad_x = torch.abs(torch.gradient(targets, dim=2)[0])  # Gradient along height (dim=2)
        targets_grad_y = torch.abs(torch.gradient(targets, dim=3)[0])  # Gradient along width (dim=3)
        targets_grad = targets_grad_x + targets_grad_y  # Combine gradients

        # Compute gradient loss
        return torch.mean(torch.abs(preds_grad - targets_grad))


# Combined Loss (Segmentation + Depth + IoU + Gradient)
class CombinedLoss(nn.Module):
    def __init__(self, seg_weight=0.3, depth_weight=0.2, iou_weight=0.5, grad_weight=0.2):
        super(CombinedLoss, self).__init__()
        self.seg_loss = nn.CrossEntropyLoss()
        self.depth_loss = nn.L1Loss()
        self.iou_loss = IoULoss()
        self.grad_loss = GradientLoss()
        self.seg_weight = seg_weight
        self.depth_weight = depth_weight
        self.iou_weight = iou_weight
        self.grad_weight = grad_weight

    def forward(self, seg_preds, depth_preds, seg_targets, depth_targets):
        seg_loss = self.seg_loss(seg_preds, seg_targets)
        depth_loss = self.depth_loss(depth_preds, depth_targets)
        iou_loss = self.iou_loss(seg_preds, seg_targets)
        grad_loss = self.grad_loss(seg_preds, seg_targets)
        return (
            self.seg_weight * seg_loss +
            self.depth_weight * depth_loss +
            self.iou_weight * iou_loss +
            self.grad_weight * grad_loss
        )


# Visualization Function
def visualize_predictions(image, segmentation_pred, depth_pred):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image.cpu().permute(1, 2, 0))
    plt.title("Input Image")

    plt.subplot(1, 3, 2)
    plt.imshow(segmentation_pred.argmax(dim=1).cpu().squeeze(), cmap='jet')
    plt.title("Segmentation Prediction")

    plt.subplot(1, 3, 3)
    plt.imshow(depth_pred.cpu().squeeze(), cmap='jet')
    plt.title("Depth Prediction")

    plt.show()


# Training Function
def train(model_name="detector", num_epoch=150, lr=1e-3, patience=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_loader = load_data("drive_data/train")
    val_loader = load_data("drive_data/val")

    # Initialize model
    model = Detector().to(device)

    # Loss function
    criterion = CombinedLoss(seg_weight=0.3, depth_weight=0.2, iou_weight=0.5, grad_weight=0.2)

    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch)

    # Training loop
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epoch):
        model.train()
        total_train_loss, total_train_iou, total_train_depth_error = 0, 0, 0

        for batch in train_loader:
            images = batch['image'].to(device)
            segmentation_labels = batch['track'].to(device).long()
            depth_labels = batch['depth'].to(device).unsqueeze(1)

            optimizer.zero_grad()
            segmentation_pred, depth_pred = model(images)

            loss = criterion(segmentation_pred, depth_pred, segmentation_labels, depth_labels)

            if torch.isnan(loss).any():
                print("NaN detected in loss. Stopping training.")
                return

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Compute IoU
            iou_value = (1 - criterion.iou_loss(segmentation_pred, segmentation_labels)).item()
            depth_error = criterion.depth_loss(depth_pred, depth_labels).item()

            total_train_loss += loss.item()
            total_train_iou += iou_value
            total_train_depth_error += depth_error

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_iou = total_train_iou / len(train_loader)
        avg_train_depth_error = total_train_depth_error / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epoch}] - Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f}, Train Depth Error: {avg_train_depth_error:.4f}")

        # Validation
        model.eval()
        total_val_loss, total_val_iou, total_val_depth_error = 0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                segmentation_labels = batch['track'].to(device).long()
                depth_labels = batch['depth'].to(device).unsqueeze(1)

                segmentation_pred, depth_pred = model(images)

                loss = criterion(segmentation_pred, depth_pred, segmentation_labels, depth_labels)

                iou_value = (1 - criterion.iou_loss(segmentation_pred, segmentation_labels)).item()
                depth_error = criterion.depth_loss(depth_pred, depth_labels).item()

                total_val_loss += loss.item()
                total_val_iou += iou_value
                total_val_depth_error += depth_error

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_iou = total_val_iou / len(val_loader)
        avg_val_depth_error = total_val_depth_error / len(val_loader)
        print(f"Epoch [{epoch + 1}/{num_epoch}] - Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}, Val Depth Error: {avg_val_depth_error:.4f}")

        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            save_model(model, model_name, log_dir)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1} with best validation loss: {best_val_loss:.4f}")
                break

        scheduler.step()

    print("Training complete!")


if __name__ == "__main__":
    train(model_name="detector", num_epoch=150, lr=1e-3, patience=20)