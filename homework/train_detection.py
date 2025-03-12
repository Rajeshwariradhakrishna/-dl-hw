import torch
import torch.optim as optim
import torch.nn as nn
from datasets.road_dataset import load_data
from models import Detector

# Load dataset
train_loader, val_loader = load_data("drive_data")

# Initialize model
model = Detector()
model.train()

# Define loss functions and optimizer
criterion_segmentation = nn.CrossEntropyLoss()
criterion_depth = nn.L1Loss()  # Mean Absolute Error
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    for batch in train_loader:
        images, segmentation_labels, depth_labels = batch['image'], batch['track'], batch['depth']
        
        optimizer.zero_grad()
        segmentation_pred, depth_pred = model(images)
        
        loss_segmentation = criterion_segmentation(segmentation_pred, segmentation_labels)
        loss_depth = criterion_depth(depth_pred, depth_labels)
        
        loss = loss_segmentation + loss_depth
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}: Loss = {loss.item()}")

# Save the model
torch.save(model.state_dict(), "detector.pth")
