import torch
import torch.optim as optim
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from homework.datasets.drive_dataset import load_data
from models import Detector, HOMEWORK_DIR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

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
        preds = torch.softmax(preds, dim=1)  # Apply softmax to get probabilities
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
def train(model_name="detector", num_epoch=50, lr=1e-3, patience=10, batch_size=32, accumulation_steps=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
    print("Loading training data...")
    train_dataset = load_data("drive_data/train")  # Ensure this returns a dataset, not a DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    print(f"Loaded {len(train_dataset)} training samples.")

    print("Loading validation data...")
    val_dataset = load_data("drive_data/val")  # Ensure this returns a dataset, not a DataLoader
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    print(f"Loaded {len(val_dataset)} validation samples.")

    # Initialize model
    model = Detector().to(device)

    # Loss functions
    criterion_segmentation = IoULoss()  # Use IoU Loss for segmentation
    criterion_depth = CombinedDepthLoss(l1_weight=0.8, mse_weight=0.2)

    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=1e-5)

    # Mixed precision training
    scaler = GradScaler()

    # Training loop
    best_val_iou = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epoch):
        model.train()
        total_train_loss, total_train_iou, total_train_depth_error = 0, 0, 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device, non_blocking=True)
            segmentation_labels = batch['track'].to(device, non_blocking=True).long()
            depth_labels = batch['depth'].to(device, non_blocking=True).unsqueeze(1)

            # Forward pass with mixed precision
            with autocast():
                segmentation_pred, depth_pred = model(images)
                loss_segmentation = criterion_segmentation(segmentation_pred, segmentation_labels)
                loss_depth = criterion_depth(depth_pred, depth_labels)
                loss = (loss_segmentation + loss_depth) / accumulation_steps

            # Backward pass with scaling
            scaler.scale(loss).backward()

            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Compute IoU
            iou_value = (1 - loss_segmentation).item()
            depth_error = loss_depth.item()

            total_train_loss += loss.item()
            total_train_iou += iou_value
            total_train_depth_error += depth_error

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_iou = total_train_iou / len(train_loader)
        avg_train_depth_error = total_train_depth_error / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epoch}] - Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f}, Train Depth Error: {avg_train_depth_error:.4f}")

        # Validation (every 2 epochs to save time)
        if (epoch + 1) % 2 == 0:
            model.eval()
            total_val_loss, total_val_iou, total_val_depth_error = 0, 0, 0

            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(device, non_blocking=True)
                    segmentation_labels = batch['track'].to(device, non_blocking=True).long()
                    depth_labels = batch['depth'].to(device, non_blocking=True).unsqueeze(1)

                    segmentation_pred, depth_pred = model(images)

                    loss_segmentation = criterion_segmentation(segmentation_pred, segmentation_labels)
                    loss_depth = criterion_depth(depth_pred, depth_labels)
                    loss = loss_segmentation + loss_depth

                    iou_value = (1 - loss_segmentation).item()
                    depth_error = loss_depth.item()

                    total_val_loss += loss.item()
                    total_val_iou += iou_value
                    total_val_depth_error += depth_error

            avg_val_loss = total_val_loss / len(val_loader)
            avg_val_iou = total_val_iou / len(val_loader)
            avg_val_depth_error = total_val_depth_error / len(val_loader)
            print(f"Epoch [{epoch + 1}/{num_epoch}] - Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}, Val Depth Error: {avg_val_depth_error:.4f}")

            # Check for improvement
            if avg_val_iou > best_val_iou:
                best_val_iou = avg_val_iou
                epochs_no_improve = 0
                save_model(model, model_name, log_dir)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1} with best validation IoU: {best_val_iou:.4f}")
                    break

        scheduler.step()

    # Visualize predictions after training
    visualize_predictions(model, val_loader, device)
    print("Training complete!")

if __name__ == "__main__":
    train(model_name="detector", num_epoch=50, lr=1e-3, patience=10, batch_size=32, accumulation_steps=4)