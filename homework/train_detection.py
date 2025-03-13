import torch
import torch.optim as optim
import torch.nn as nn
from homework.datasets.drive_dataset import load_data
from models import Detector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
#train_loader, val_loader = load_data("drive_data")
train_loader = load_data("drive_data/train")
val_loader = load_data("drive_data/val")

# Initialize model
model = Detector().to(device)
model.train()

# Define loss functions and optimizer
criterion_segmentation = nn.CrossEntropyLoss()
criterion_depth = nn.L1Loss()  # Mean Absolute Error
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    total_loss = 0  # Track loss per epoch
    
    for batch in train_loader:
        images = batch['image'].to(device)
        segmentation_labels = batch['track'].to(device).long()  # Ensure long type for CrossEntropyLoss
        depth_labels = batch['depth'].to(device)

        optimizer.zero_grad()
        
        # Forward pass
        segmentation_pred, depth_pred = model(images)

        # Compute losses
        loss_segmentation = criterion_segmentation(segmentation_pred, segmentation_labels)
        loss_depth = criterion_depth(depth_pred, depth_labels)
        loss = loss_segmentation + loss_depth

        # Backpropagation
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader)}")

# Save the model
torch.save(model.state_dict(), "detector.pth")
print("Model saved successfully!")