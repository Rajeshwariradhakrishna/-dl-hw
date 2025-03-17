import torch
import torch.optim as optim
import torch.nn as nn
import os
from homework.datasets.drive_dataset import load_data
from models import Detector, HOMEWORK_DIR  
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Define log directory
log_dir = str(HOMEWORK_DIR)
os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist

def save_model(model, model_name, log_dir):
    """Save the model's state_dict to the specified directory."""
    model_path = os.path.join(log_dir, f"{model_name}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# IoU Loss for Segmentation
class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)  # Convert logits to probabilities
        preds = (preds > 0.5).float()  # Convert to binary mask
        intersection = (preds * targets).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou.mean()

# IoU Metric
class IoUMetric(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IoUMetric, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float()
        intersection = (preds * targets).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return iou.mean()

def train(model_name="detector", num_epoch=10, lr=1e-3, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_loader = load_data("drive_data/train")
    val_loader = load_data("drive_data/val")

    # Initialize model
    model = Detector().to(device)

    # Loss functions
    criterion_segmentation = IoULoss()
    criterion_depth = nn.L1Loss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.5, verbose=True)

    iou_metric = IoUMetric()

    # Training loop
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epoch):
        model.train()
        total_train_loss, total_train_iou = 0, 0

        for batch in train_loader:
            images = batch['image'].to(device)
            segmentation_labels = batch['track'].to(device).float()
            depth_labels = batch['depth'].to(device).unsqueeze(1)

            optimizer.zero_grad()
            segmentation_pred, depth_pred = model(images)

            loss_segmentation = criterion_segmentation(segmentation_pred, segmentation_labels)
            loss_depth = criterion_depth(depth_pred, depth_labels)
            loss = loss_segmentation + loss_depth

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_iou += iou_metric(segmentation_pred, segmentation_labels).item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_iou = total_train_iou / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epoch}] - Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f}")

        # Validation
        model.eval()
        total_val_loss, total_val_iou = 0, 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                segmentation_labels = batch['track'].to(device).float()
                depth_labels = batch['depth'].to(device).unsqueeze(1)

                segmentation_pred, depth_pred = model(images)

                loss_segmentation = criterion_segmentation(segmentation_pred, segmentation_labels)
                loss_depth = criterion_depth(depth_pred, depth_labels)
                loss = loss_segmentation + loss_depth

                total_val_loss += loss.item()
                total_val_iou += iou_metric(segmentation_pred, segmentation_labels).item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_iou = total_val_iou / len(val_loader)
        print(f"Epoch [{epoch + 1}/{num_epoch}] - Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}")

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
