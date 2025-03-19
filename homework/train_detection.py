import torch
import torch.optim as optim
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from homework.datasets.drive_dataset import load_data
from models import Detector, HOMEWORK_DIR
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
        preds = torch.softmax(preds, dim=1)  # Apply softmax to get probabilities
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float()
        intersection = (preds * targets_one_hot).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3)) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou.mean()


# Dice Loss for Segmentation
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.softmax(preds, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float()
        intersection = (preds * targets_one_hot).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


# Focal Loss for Segmentation
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(preds, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# Combined Segmentation Loss (IoU + Dice + Focal)
class CombinedSegmentationLoss(nn.Module):
    def __init__(self, iou_weight=0.5, dice_weight=0.5, focal_weight=0.0):  # Prioritize IoU and Dice
        super(CombinedSegmentationLoss, self).__init__()
        self.iou_loss = IoULoss()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.iou_weight = iou_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, preds, targets):
        iou_loss = self.iou_loss(preds, targets)
        dice_loss = self.dice_loss(preds, targets)
        focal_loss = self.focal_loss(preds, targets)
        return self.iou_weight * iou_loss + self.dice_weight * dice_loss + self.focal_weight * focal_loss


# Depth Loss (L1 + MSE + False Positive Penalty)
class DepthLoss(nn.Module):
    def __init__(self, l1_weight=0.7, mse_weight=0.2, fp_weight=0.1):
        super(DepthLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.fp_weight = fp_weight

    def forward(self, preds, targets):
        l1_loss = self.l1_loss(preds, targets)
        mse_loss = self.mse_loss(preds, targets)
        fp_loss = torch.mean(torch.relu(preds - targets))  # Penalize false positives
        return l1_loss + mse_loss + self.fp_weight * fp_loss


# Training Function
def train(model_name="detector", num_epoch=20, lr=1e-3, patience=5):  # Reduced to 20 epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset with shuffling for training
    train_loader = load_data("drive_data/train", shuffle=True)  # Enable shuffling
    val_loader = load_data("drive_data/val", shuffle=False)     # No shuffling for validation

    # Initialize model
    model = Detector().to(device)

    # Loss functions
    criterion_segmentation = CombinedSegmentationLoss(iou_weight=0.5, dice_weight=0.5, focal_weight=0.0)
    criterion_depth = DepthLoss(l1_weight=0.7, mse_weight=0.2, fp_weight=0.1)

    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=1e-5)

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

            if torch.isnan(loss).any():
                print("NaN detected in loss. Stopping training.")
                return

            loss.backward()
            optimizer.step()

            # Compute IoU
            iou_value = 1 - criterion_segmentation.iou_loss(segmentation_pred, segmentation_labels).item()
            depth_error = loss_depth.item()

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

                loss_segmentation = criterion_segmentation(segmentation_pred, segmentation_labels)
                loss_depth = criterion_depth(depth_pred, depth_labels)
                loss = loss_segmentation + loss_depth

                iou_value = 1 - criterion_segmentation.iou_loss(segmentation_pred, segmentation_labels).item()
                depth_error = loss_depth.item()

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
    train(model_name="detector", num_epoch=20, lr=1e-3, patience=5)