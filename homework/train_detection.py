import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image

# Define the U-Net architecture
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Contracting path (Encoder)
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Expanding path (Decoder)
        self.decoder4 = self.conv_block(1024 + 512, 512)
        self.decoder3 = self.conv_block(512 + 256, 256)
        self.decoder2 = self.conv_block(256 + 128, 128)
        self.decoder1 = self.conv_block(128 + 64, 64)
        
        # Final output layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        
        # Decoder path (skip connections)
        dec4 = self.decoder4(torch.cat([F.upsample(bottleneck, enc4.size()[2:]), enc4], 1))
        dec3 = self.decoder3(torch.cat([F.upsample(dec4, enc3.size()[2:]), enc3], 1))
        dec2 = self.decoder2(torch.cat([F.upsample(dec3, enc2.size()[2:]), enc2], 1))
        dec1 = self.decoder1(torch.cat([F.upsample(dec2, enc1.size()[2:]), enc1], 1))
        
        # Output layer
        out = self.final(dec1)
        return out

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model, loss function and optimizer
model = UNet(in_channels=3, out_channels=1).to(device)
criterion = nn.BCEWithLogitsLoss()  # For binary segmentation
optimizer = optim.Adam(model.parameters(), lr=0.001)

# IOU Calculation function
def calculate_iou(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target) - intersection
    iou = intersection / union
    return iou.item()

# DataLoader and Dataset
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('1')  # 1 for binary masks
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# Image transformation (resize, normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Assuming you have lists of image and mask file paths
train_images = ['path_to_train_image1.jpg', 'path_to_train_image2.jpg']  # List of image paths
train_masks = ['path_to_train_mask1.png', 'path_to_train_mask2.png']  # List of mask paths
val_images = ['path_to_val_image1.jpg', 'path_to_val_image2.jpg']  # List of validation image paths
val_masks = ['path_to_val_mask1.png', 'path_to_val_mask2.png']  # List of validation mask paths

# Create Dataset and DataLoader for train and validation
train_dataset = SegmentationDataset(train_images, train_masks, transform=transform)
val_dataset = SegmentationDataset(val_images, val_masks, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Training Loop
def train(model, train_loader, val_loader, epochs=10):
    best_val_iou = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_iou = 0
        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            
            # IOU calculation
            iou = calculate_iou(outputs, labels)
            train_iou += iou
        
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        train_iou /= len(train_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0
        val_iou = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # IOU calculation
                iou = calculate_iou(outputs, labels)
                val_iou += iou

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)

        print(f"Epoch [{epoch+1}/{epochs}], Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
        
        # Save the model if the IoU improves
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), "best_unet_model.pth")
            print("Model saved!")

# Start training (make sure data loaders are ready)
train(model, train_loader, val_loader, epochs=10)
