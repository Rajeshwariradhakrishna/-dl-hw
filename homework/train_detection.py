import torch
import torch.optim as optim
import torch.nn as nn
import os
from homework.datasets.drive_dataset import load_data
from models import Detector, HOMEWORK_DIR  
from torchvision import transforms

# Define log_dir where you want to save the model
log_dir = str(HOMEWORK_DIR)
os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Define the save_model function
def save_model(model, model_name, log_dir):
    """Save the model's state_dict to the specified directory."""
    model_path = os.path.join(log_dir, f"{model_name}.th")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def train(model_name="detector", num_epoch=10, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_loader = load_data("drive_data/train")
    val_loader = load_data("drive_data/val")

    # Data augmentation and normalization (optional)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2)
    ])

    # Initialize model
    model = Detector().to(device)
    model.train()

    # Define loss functions
    criterion_segmentation = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 2.0], device=device))  # Adjust weights for segmentation
    criterion_depth = nn.SmoothL1Loss()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Learning rate scheduling

    for epoch in range(num_epoch):
        total_train_loss = 0

        # Training loop
        model.train()
        for batch in train_loader:
            images = batch['image'].to(device)
            segmentation_labels = batch['track'].to(device).long()
            depth_labels = batch['depth'].to(device).unsqueeze(1)  # Fix shape

            # Apply transformations
            images = torch.stack([transform(image) for image in images])

            optimizer.zero_grad()
            segmentation_pred, depth_pred = model(images)

            # Compute loss
            loss_segmentation = criterion_segmentation(segmentation_pred, segmentation_labels)
            loss_depth = criterion_depth(depth_pred, depth_labels)
            loss = loss_segmentation + 5 * loss_depth  # Increase depth loss weight

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

        # Validation loop
        model.eval()
        total_val_loss = 0
        total_val_iou = 0
        total_val_depth_error = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                segmentation_labels = batch['track'].to(device).long()
                depth_labels = batch['depth'].to(device).unsqueeze(1)

                segmentation_pred, depth_pred = model(images)

                # Compute validation loss
                loss_segmentation = criterion_segmentation(segmentation_pred, segmentation_labels)
                loss_depth = criterion_depth(depth_pred, depth_labels)
                loss = loss_segmentation + 5 * loss_depth

                total_val_loss += loss.item()

                # Compute IoU for segmentation
                segmentation_pred = torch.argmax(segmentation_pred, dim=1)  # Apply argmax for IoU calculation
                intersection = (segmentation_pred & segmentation_labels).sum().float()
                union = (segmentation_pred | segmentation_labels).sum().float()
                iou = intersection / union if union != 0 else 0
                total_val_iou += iou.item()

                # Compute depth error
                abs_depth_error = torch.abs(depth_pred - depth_labels).mean().item()
                total_val_depth_error += abs_depth_error

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_iou = total_val_iou / len(val_loader)
        avg_val_depth_error = total_val_depth_error / len(val_loader)

        print(f"Epoch {epoch+1}: Validation Loss = {avg_val_loss:.4f}")
        print(f"Epoch {epoch+1}: IoU = {avg_val_iou:.4f} (Target > 0.75)")
        print(f"Epoch {epoch+1}: Depth Error = {avg_val_depth_error:.4f} (Target < 0.05)")

        # Save model if it meets validation criteria
        if avg_val_iou > 0.75 and avg_val_depth_error < 0.05:
            save_model(model, model_name, log_dir)

        # Step the scheduler
        scheduler.step()

# Run the training process
train(model_name="detector", num_epoch=20, lr=1e-3)