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

# Cross-Entropy Loss for Segmentation
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        return self.criterion(preds, targets)

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

# Confusion Matrix for IoU Calculation
class ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes))

    def update(self, preds, targets):
        preds = torch.argmax(preds, dim=1)
        for t, p in zip(targets.view(-1), preds.view(-1)):
            self.confusion_matrix[t.long(), p.long()] += 1

    def compute_iou(self):
        intersection = torch.diag(self.confusion_matrix)
        union = self.confusion_matrix.sum(0) + self.confusion_matrix.sum(1) - intersection
        iou = intersection / (union + 1e-6)
        return iou.mean().item()

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
    print("Loading training data...")
    train_loader = load_data("drive_data/train")
    print(f"Loaded {len(train_loader.dataset)} training samples.")

    print("Loading validation data...")
    val_loader = load_data("drive_data/val")
    print(f"Loaded {len(val_loader.dataset)} validation samples.")

    # Check a sample batch
    sample_batch = next(iter(train_loader))
    print("Sample batch keys:", sample_batch.keys())
    print("Sample image shape:", sample_batch['image'].shape)
    print("Sample track shape:", sample_batch['track'].shape)
    print("Sample depth shape:", sample_batch['depth'].shape)

    # Initialize model
    model = Detector().to(device)

    # Loss functions
    criterion_segmentation = CrossEntropyLoss()  # Use Cross-Entropy Loss for segmentation
    criterion_depth = CombinedDepthLoss(l1_weight=0.8, mse_weight=0.2)

    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=1e-5)

    # Metrics
    train_metrics = {"iou": [], "depth_error": []}
    val_metrics = {"iou": [], "depth_error": []}
    confusion_matrix = ConfusionMatrix(num_classes=3)

    # Training loop
    best_val_iou = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epoch):
        model.train()
        confusion_matrix.reset()

        print(f"Starting epoch {epoch + 1}/{num_epoch}...")
        for batch_idx, batch in enumerate(train_loader):
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

            # Update confusion matrix
            confusion_matrix.update(segmentation_pred, segmentation_labels)
            train_metrics["depth_error"].append(loss_depth.item())

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

        # Compute training IoU
        train_iou = confusion_matrix.compute_iou()
        train_metrics["iou"].append(train_iou)

        # Validation
        model.eval()
        confusion_matrix.reset()

        print("Running validation...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                images = batch['image'].to(device)
                segmentation_labels = batch['track'].to(device).long()
                depth_labels = batch['depth'].to(device).unsqueeze(1)

                segmentation_pred, depth_pred = model(images)

                loss_segmentation = criterion_segmentation(segmentation_pred, segmentation_labels)
                loss_depth = criterion_depth(depth_pred, depth_labels)
                loss = loss_segmentation + loss_depth

                # Update confusion matrix
                confusion_matrix.update(segmentation_pred, segmentation_labels)
                val_metrics["depth_error"].append(loss_depth.item())

                if batch_idx % 10 == 0:
                    print(f"Validation Batch {batch_idx}/{len(val_loader)} - Loss: {loss.item():.4f}")

        # Compute validation IoU
        val_iou = confusion_matrix.compute_iou()
        val_metrics["iou"].append(val_iou)

        # Print metrics
        print(f"Epoch [{epoch + 1}/{num_epoch}] - Train IoU: {train_iou:.4f}, Val IoU: {val_iou:.4f}")

        # Check for improvement
        if val_iou > best_val_iou:
            best_val_iou = val_iou
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
    train(model_name="detector", num_epoch=50, lr=1e-3, patience=10)