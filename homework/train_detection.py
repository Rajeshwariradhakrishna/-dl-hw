import torch
import torch.optim as optim
import torch.nn as nn
import os
from homework.datasets.drive_dataset import load_data
from models import Detector, HOMEWORK_DIR

# Define log_dir where you want to save the model
log_dir = str(HOMEWORK_DIR)
os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist

# 🔹 Save model function
def save_model(model, model_name, log_dir):
    """Save the model's state_dict to the specified directory."""
    model_path = os.path.join(log_dir, f"{model_name}.th")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")

# 🔹 Dice Loss for segmentation
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)  # Convert logits to probabilities
        intersection = (preds * targets).sum(dim=(1, 2, 3))
        denominator = (preds + targets).sum(dim=(1, 2, 3))

        dice_score = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1 - dice_score.mean()

# 🔹 IoU Calculation
def calculate_iou(preds, targets):
    preds = torch.sigmoid(preds) > 0.5  # Convert logits to binary masks
    intersection = (preds & targets).sum(dim=(1, 2, 3))
    union = (preds | targets).sum(dim=(1, 2, 3))
    return (intersection / (union + 1e-6)).mean()


def train(model_name="detector", num_epoch=10, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_loader = load_data("drive_data/train")
    val_loader = load_data("drive_data/val")

    # Initialize model
    model = Detector().to(device)
    model.train()

    # Define loss functions
    criterion_segmentation = DiceLoss()  # 🔹 Using Dice Loss instead of CrossEntropyLoss
    criterion_depth = nn.L1Loss()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")  # Track the best validation loss

    for epoch in range(num_epoch):
        total_train_loss = 0
        total_train_iou = 0
        num_train_batches = 0

        # 🔹 Training loop
        model.train()
        for batch in train_loader:
            images = batch["image"].to(device)
            segmentation_labels = batch["track"].to(device).float()  # One-hot labels
            depth_labels = batch["depth"].to(device).unsqueeze(1)  # Fix shape

            optimizer.zero_grad()
            segmentation_pred, depth_pred = model(images)

            # Compute losses
            loss_segmentation = criterion_segmentation(segmentation_pred, segmentation_labels)
            loss_depth = criterion_depth(depth_pred, depth_labels)
            loss = loss_segmentation + loss_depth

            loss.backward()
            optimizer.step()

            # Track metrics
            total_train_loss += loss.item()
            total_train_iou += calculate_iou(segmentation_pred, segmentation_labels).item()
            num_train_batches += 1

        avg_train_loss = total_train_loss / num_train_batches
        avg_train_iou = total_train_iou / num_train_batches

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Train IoU = {avg_train_iou:.4f}")

        # 🔹 Validation loop
        model.eval()
        total_val_loss = 0
        total_val_iou = 0
        num_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                segmentation_labels = batch["track"].to(device).float()
                depth_labels = batch["depth"].to(device).unsqueeze(1)

                segmentation_pred, depth_pred = model(images)

                # Compute validation losses
                loss_segmentation = criterion_segmentation(segmentation_pred, segmentation_labels)
                loss_depth = criterion_depth(depth_pred, depth_labels)
                loss = loss_segmentation + loss_depth

                # Track metrics
                total_val_loss += loss.item()
                total_val_iou += calculate_iou(segmentation_pred, segmentation_labels).item()
                num_val_batches += 1

        avg_val_loss = total_val_loss / num_val_batches
        avg_val_iou = total_val_iou / num_val_batches

        print(f"Epoch {epoch+1}: Val Loss = {avg_val_loss:.4f}, Val IoU = {avg_val_iou:.4f}")

        # 🔹 Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model(model, model_name, log_dir)

    print("✅ Training Complete!")


# 🔹 Run training
train(model_name="detector", num_epoch=15, lr=1e-3)
