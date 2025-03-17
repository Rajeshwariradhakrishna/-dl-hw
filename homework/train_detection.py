import torch
import torch.optim as optim
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from homework.datasets.drive_dataset import load_data
from models import Detector, HOMEWORK_DIR
from torch.optim.lr_scheduler import CosineAnnealingLR

# Define log directory
log_dir = str(HOMEWORK_DIR)
os.makedirs(log_dir, exist_ok=True)

def save_model(model, model_name, log_dir):
    """Save the model's state_dict to the specified directory."""
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

# Combined Depth Loss (L1 + MSE)
class CombinedDepthLoss(nn.Module):
    def __init__(self, l1_weight=0.8, mse_weight=0.2):
        super(CombinedDepthLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight

    def forward(self, preds, targets):
        l1_loss = self.l1_loss(preds, targets)
        mse_loss = self.mse_loss(preds, targets)
        return self.l1_weight * l1_loss + self.mse_weight * mse_loss

# Detection Metric for IoU
class DetectionMetric:
    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.intersection = torch.zeros(self.num_classes)
        self.union = torch.zeros(self.num_classes)

    def update(self, preds, targets):
        preds = torch.argmax(preds, dim=1)
        for cls in range(self.num_classes):
            pred_mask = (preds == cls)
            target_mask = (targets == cls)
            self.intersection[cls] += (pred_mask & target_mask).sum().item()
            self.union[cls] += (pred_mask | target_mask).sum().item()

    def compute(self):
        iou = self.intersection / (self.union + 1e-6)
        return {"iou": iou.mean().item()}

# Visualize Predictions
def visualize_predictions(model, val_loader, device):
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            segmentation_labels = batch['track'].to(device).long()
            segmentation_pred, _ = model(images)
            segmentation_pred = torch.argmax(segmentation_pred, dim=1)

            # Plot images, ground truth, and predictions
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(images[0].cpu().permute(1, 2, 0))
            plt.title("Image")
            plt.subplot(1, 3, 2)
            plt.imshow(segmentation_labels[0].cpu(), cmap="jet")
            plt.title("Ground Truth")
            plt.subplot(1, 3, 3)
            plt.imshow(segmentation_pred[0].cpu(), cmap="jet")
            plt.title("Prediction")
            plt.show()
            break

# Training Function
def train(model_name="detector", num_epoch=50, lr=1e-3, patience=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop((256, 256), scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load dataset with augmentations
    train_loader = load_data("drive_data/train")
    val_loader = load_data("drive_data/val")

    # Initialize model
    model = Detector().to(device)

    # Loss functions
    criterion_segmentation = IoULoss()  # Use IoU Loss for segmentation
    criterion_depth = CombinedDepthLoss(l1_weight=0.8, mse_weight=0.2)

    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=1e-5)

    # Metrics
    train_metrics = {"iou": [], "depth_error": []}
    val_metrics = {"iou": [], "depth_error": []}
    detection_metric = DetectionMetric(num_classes=3)

    # Training loop
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epoch):
        model.train()
        detection_metric.reset()

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

            # Update metrics
            detection_metric.update(segmentation_pred, segmentation_labels)
            train_metrics["depth_error"].append(loss_depth.item())

        # Compute training IoU
        train_iou = detection_metric.compute()["iou"]
        train_metrics["iou"].append(train_iou)

        # Validation
        model.eval()
        detection_metric.reset()

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                segmentation_labels = batch['track'].to(device).long()
                depth_labels = batch['depth'].to(device).unsqueeze(1)

                segmentation_pred, depth_pred = model(images)

                loss_segmentation = criterion_segmentation(segmentation_pred, segmentation_labels)
                loss_depth = criterion_depth(depth_pred, depth_labels)
                loss = loss_segmentation + loss_depth

                # Update metrics
                detection_metric.update(segmentation_pred, segmentation_labels)
                val_metrics["depth_error"].append(loss_depth.item())

        # Compute validation IoU
        val_iou = detection_metric.compute()["iou"]
        val_metrics["iou"].append(val_iou)

        # Print metrics
        print(f"Epoch [{epoch + 1}/{num_epoch}] - Train IoU: {train_iou:.4f}, Val IoU: {val_iou:.4f}")

        # Check for improvement
        if val_iou > best_val_loss:
            best_val_loss = val_iou
            epochs_no_improve = 0
            save_model(model, model_name, log_dir)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1} with best validation IoU: {best_val_loss:.4f}")
                break

        scheduler.step()

    # Visualize predictions after training
    visualize_predictions(model, val_loader, device)
    print("Training complete!")

if __name__ == "__main__":
    train(model_name="detector", num_epoch=50, lr=1e-3, patience=10)