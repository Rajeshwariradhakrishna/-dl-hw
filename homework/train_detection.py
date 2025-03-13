import torch
import torch.optim as optim
import torch.nn as nn
import os
from homework.datasets.drive_dataset import load_data
from models import Detector

# Define log_dir where you want to save the model
log_dir = "/homework"
os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Define the save_model function
def save_model(model, model_name, log_dir):
    """Save the model's state_dict to the specified directory."""
    torch.save(model.state_dict(), os.path.join(log_dir, f"{model_name}.th"))
    print(f"Model saved to {os.path.join(log_dir, f'{model_name}.th')}")

def train(model_name="detector", num_epoch=10, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_loader = load_data("drive_data/train")  
    val_loader = load_data("drive_data/val")  # Now using validation data

    # Initialize model
    model = Detector().to(device)
    model.train()

    # Define loss functions
    criterion_segmentation = nn.CrossEntropyLoss()
    criterion_depth = nn.L1Loss()

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

            # Compute loss
            loss_segmentation = criterion_segmentation(segmentation_pred, segmentation_labels)
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
                loss_segmentation = criterion_segmentation(segmentation_pred, segmentation_labels)
                loss_depth = criterion_depth(depth_pred, depth_labels)
                loss = loss_segmentation + loss_depth

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Validation Loss = {avg_val_loss:.4f}")

    # Save the trained model using the defined save_model function
    save_model(model, model_name, log_dir)
