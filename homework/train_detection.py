import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from models import Detector, save_model, load_model
import os

# Function to load dataset
def load_data(data_dir, batch_size=8, shuffle=True):
    """
    Load dataset from the specified directory.

    Args:
        data_dir: Path to the dataset directory.
        batch_size: Batch size for the DataLoader.
        shuffle: Whether to shuffle the dataset.

    Returns:
        DataLoader for the dataset.
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((96, 128)),  # Match the input size expected by the model
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.ToTensor(),
    ])

    # Check if the data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The directory '{data_dir}' does not exist. Please create it and add your dataset.")

    # Load dataset (assume images are in subfolders for classes)
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader

# Training Function
def train(model_name: str = "detector", num_epoch: int = 10, lr: float = 0.001):
    """
    Train the model.

    Args:
        model_name: Name of the model to train (default: "detector").
        num_epoch: Number of training epochs (default: 10).
        lr: Learning rate for the optimizer (default: 0.001).
    """
    # Initialize model, loss functions, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model based on the model_name argument
    if model_name == "detector":
        model = Detector().to(device)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    criterion_seg = nn.CrossEntropyLoss()  # For segmentation
    criterion_depth = nn.MSELoss()  # For depth estimation
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Use lr argument
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Learning rate scheduler

    # Load datasets
    train_loader = load_data("drive_data/train")  # Load training data
    val_loader = load_data("drive_data/val", shuffle=False)  # Load validation data (no shuffling)

    # Training loop
    for epoch in range(num_epoch):  # Use num_epoch argument
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            segmentation_logits, raw_depth = model(imgs)

            # Compute losses (assume labels contain both segmentation and depth)
            loss_seg = criterion_seg(segmentation_logits, labels['segmentation'])
            loss_depth = criterion_depth(raw_depth.squeeze(1), labels['depth'])
            total_loss = loss_seg + loss_depth

            # Backpropagation
            total_loss.backward()
            optimizer.step()

        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epoch}], Loss: {total_loss.item():.4f}")

    # Save the model
    save_model(model)
    print(f"Model saved as {model_name}.th")


if __name__ == "__main__":
    train()  # Default arguments will be used