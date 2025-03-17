import torch
import torch.optim as optim
import torch.nn as nn
import os
from homework.datasets.drive_dataset import load_data
from models import Detector, HOMEWORK_DIR
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Define log directory
log_dir = str(HOMEWORK_DIR)
os.makedirs(log_dir, exist_ok=True)

def save_model(model, model_name, log_dir):
    """Save the model's state_dict to the specified directory."""
    model_path = os.path.join(log_dir, f"{model_name}.th")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# Focal Tversky Loss for Segmentation
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha  # Weight for false positives
        self.beta = beta    # Weight for false negatives
        self.gamma = gamma  # Focal parameter
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.softmax(preds, dim=1)  # Convert logits to probabilities
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float()
        intersection = (preds * targets_one_hot).sum(dim=(2, 3))
        fps = (preds * (1 - targets_one_hot)).sum(dim=(2, 3))
        fns = ((1 - preds) * targets_one_hot).sum(dim=(2, 3))
        tversky = (intersection + self.smooth) / (intersection + self.alpha * fps + self.beta * fns + self.smooth)
        focal_tversky = (1 - tversky) ** self.gamma
        return focal_tversky.mean()

# Huber Loss for Depth Regression
class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, preds, targets):
        residual = torch.abs(preds - targets)
        condition = residual < self.delta
        loss = torch.where(condition, 0.5 * residual ** 2, self.delta * (residual - 0.5 * self.delta))
        return loss.mean()

# IoU Metric for Segmentation
class IoUMetric(nn.Module):
    def __init__(self, num_classes, smooth=1e-6):
        super(IoUMetric, self).__init__()
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, preds, targets):
        preds = torch.argmax(preds, dim=1)  # Convert logits to class indices
        preds_one_hot = torch.nn.functional.one_hot(preds, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        intersection = (preds_one_hot * targets_one_hot).sum(dim=(2, 3))
        union = preds_one_hot.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3)) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return iou.mean()

def train(model_name="detector", num_epoch=50, lr=1e-3, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_loader = load_data("drive_data/train")
    val_loader = load_data("drive_data/val")

    # Initialize model
    model = Detector().to(device)

    # Loss functions
    criterion_segmentation = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)  # Use Focal Tversky Loss for segmentation
    criterion_depth = HuberLoss(delta=1.0)  # Use Huber Loss for depth regression

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.5, verbose=True)

    iou_metric = IoUMetric(num_classes=3)

    # Metrics tracking
    metrics = {
        "train_iou": [],
        "val_iou": [],
        "train_depth_error": [],
        "val_depth_error": [],
    }

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

            loss_segmentation = criterion_segmentation(segmentation_pred, segmentation_labels)
            loss_depth = criterion_depth(depth_pred, depth_labels)
            loss = loss_segmentation + loss_depth

            loss.backward()
            optimizer.step()

            # Compute metrics
            iou_value = iou_metric(segmentation_pred, segmentation_labels).item()
            depth_error = loss_depth.item()

            total_train_loss += loss.item()
            total_train_iou += iou_value
            total_train_depth_error += depth_error

        # Record training metrics
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_iou = total_train_iou / len(train_loader)
        avg_train_depth_error = total_train_depth_error / len(train_loader)
        metrics["train_iou"].append(avg_train_iou)
        metrics["train_depth_error"].append(avg_train_depth_error)

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

                loss_segmentation = criterion_segmentation(segmentation_pred, segmentation_labels)
                loss_depth = criterion_depth(depth_pred, depth_labels)
                loss = loss_segmentation + loss_depth

                # Compute metrics
                iou_value = iou_metric(segmentation_pred, segmentation_labels).item()
                depth_error = loss_depth.item()

                total_val_loss += loss.item()
                total_val_iou += iou_value
                total_val_depth_error += depth_error

        # Record validation metrics
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_iou = total_val_iou / len(val_loader)
        avg_val_depth_error = total_val_depth_error / len(val_loader)
        metrics["val_iou"].append(avg_val_iou)
        metrics["val_depth_error"].append(avg_val_depth_error)

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

        scheduler.step(avg_val_loss)

    print("Training complete!")

if __name__ == "__main__":
    train(model_name="detector", num_epoch=50, lr=1e-3, patience=5)