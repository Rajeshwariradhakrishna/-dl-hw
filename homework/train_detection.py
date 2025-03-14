import torch
import torch.optim as optim
import torch.nn as nn
import os
from homework.datasets.drive_dataset import load_data
from models import Detector, HOMEWORK_DIR  

# Define log_dir where you want to save the model
log_dir = str(HOMEWORK_DIR)
os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Save model function
def save_model(model, model_name, log_dir):
    model_path = os.path.join(log_dir, f"{model_name}.th")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

import torch.nn.functional as F
import torchvision.transforms as transforms

# Data augmentation to improve segmentation IoU
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Flip images randomly
    transforms.RandomRotation(10),      # Rotate images slightly
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  
    transforms.ToTensor()
])

# Dice Loss for better segmentation
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)  # Ensure values are between 0 and 1
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# Combined Loss for segmentation
def combined_loss(output, target):
    ce_loss = F.cross_entropy(output, target)
    dice = dice_loss(output, target)
    return ce_loss + dice  # Combining both losses

def train(model_name="detector", num_epoch=20, lr=1e-3, batch_size=16):  # Increased epochs, reduced batch size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset with data augmentation
    train_loader = load_data("drive_data/train", transform=data_transforms, batch_size=batch_size)  
    val_loader = load_data("drive_data/val", batch_size=batch_size)

    # Initialize model
    model = Detector().to(device)
    model.train()

    # Define loss functions
    criterion_segmentation = nn.CrossEntropyLoss()
    criterion_depth = nn.L1Loss()  # Use L1 Loss for better depth estimation

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epoch):
        total_train_loss = 0

        # Training loop
        model.train()
        for batch in train_loader:
            images = batch['image'].to(device)
            segmentation_labels = batch['track'].to(device).long()
            depth_labels = batch['depth'].to(device).unsqueeze(1)  # Fix shape

            optimizer.zero_grad()
            segmentation_pred, depth_pred = model(images)

            # Compute loss (combined loss for segmentation)
            loss_segmentation = combined_loss(segmentation_pred, segmentation_labels)
            loss_depth = criterion_depth(depth_pred, depth_labels)
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
                loss_segmentation = combined_loss(segmentation_pred, segmentation_labels)
                loss_depth = criterion_depth(depth_pred, depth_labels)
                loss = loss_segmentation + loss_depth

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Validation Loss = {avg_val_loss:.4f}")

        # Debugging: Print some predictions
        if epoch % 5 == 0:  # Every 5 epochs, print some failures
            print(f"Sample IoU: {segmentation_pred[0].max().item():.3f}, Depth Error: {loss_depth.item():.3f}")

    # Save the trained model using the defined save_model function
    save_model(model, model_name, log_dir)
