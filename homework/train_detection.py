import torch
import torch.optim as optim
import torch.nn as nn
import os
from homework.datasets.drive_dataset import load_data
from models import Detector, HOMEWORK_DIR  

# Define log_dir where you want to save the model
log_dir = str(HOMEWORK_DIR)
os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Define the save_model function
def save_model(model, model_name, log_dir):
    """Save the model's state_dict to the specified directory."""
    model_path = os.path.join(log_dir, f"{model_name}.th")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

from torch.nn.functional import binary_cross_entropy_with_logits

def dice_loss(pred, target, smooth=1.0):
    """
    Dice Loss for segmentation.
    """
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (2 * intersection + smooth) / (union + smooth)

def train(model_name="detector", num_epoch=10, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_loader = load_data("drive_data/train")
    val_loader = load_data("drive_data/val")

    # Initialize model
    model = Detector().to(device)
    model.train()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epoch):
        total_train_loss = 0

        # Training loop
        model.train()
        for batch in train_loader:
            images = batch['image'].to(device)
            segmentation_labels = batch['track'].to(device).long()
            depth_labels = batch['depth'].to(device).unsqueeze(1)

            optimizer.zero_grad()
            segmentation_pred, depth_pred = model(images)

            # Compute losses
            loss_segmentation = dice_loss(segmentation_pred, segmentation_labels)
            loss_depth = nn.L1Loss()(depth_pred, depth_labels)
            loss = loss_segmentation + loss_depth

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                segmentation_labels = batch['track'].to(device).long()
                depth_labels = batch['depth'].to(device).unsqueeze(1)

                segmentation_pred, depth_pred = model(images)

                # Compute validation loss
                loss_segmentation = dice_loss(segmentation_pred, segmentation_labels)
                loss_depth = nn.L1Loss()(depth_pred, depth_labels)
                loss = loss_segmentation + loss_depth

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Validation Loss = {avg_val_loss:.4f}")

    # Save the trained model
    save_model(model, model_name, log_dir)