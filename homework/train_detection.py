import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Example Dataset Class (Make sure to replace it with your actual dataset class)
class SegmentationDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images  # list of image paths
        self.masks = masks    # list of corresponding mask paths
        self.transform = transform
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = plt.imread(self.images[idx])  # Replace with your image loading
        mask = plt.imread(self.masks[idx])  # Replace with your mask loading

        if self.transform:
            image, mask = self.transform(image, mask)
        
        return image, mask

# Dice Coefficient Calculation (for IoU)
def dice_coefficient(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# IoU Calculation
def intersection_over_union(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    union = (pred + target).sum() - intersection
    return (intersection + smooth) / (union + smooth)

# UNet-like Architecture (Simple example)
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        out = self.decoder(x2)
        return out

# Loss Function: Dice Loss + Cross-Entropy (for segmentation)
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, outputs, targets):
        smooth = 1e-6
        outputs = outputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (outputs * targets).sum()
        dice = (2. * intersection + smooth) / (outputs.sum() + targets.sum() + smooth)
        return 1 - dice  # We minimize Dice Loss

# Data Augmentation Transform
class Transform:
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(256, scale=(0.8, 1.0), ratio=(0.75, 1.25)),
            transforms.ToTensor(),
        ])
    
    def __call__(self, image, mask):
        image = self.transforms(image)
        mask = self.transforms(mask)
        return image, mask

# Training Loop
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    best_model_wts = model.state_dict()
    best_iou = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_iou = 0.0
        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
            optimizer.zero_grad()
            
            images, masks = images.cuda(), masks.cuda()

            outputs = model(images)
            
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate IoU
            outputs = (outputs > 0.5).float()  # Thresholding the output
            iou = intersection_over_union(outputs, masks)
            running_iou += iou.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_iou = running_iou / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, IoU: {epoch_iou:.4f}")

        # Validation
        model.eval()
        val_iou = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.cuda(), masks.cuda()

                outputs = model(images)
                outputs = (outputs > 0.5).float()
                iou = intersection_over_union(outputs, masks)
                val_iou += iou.item()

        val_iou /= len(val_loader)
        print(f"Validation IoU: {val_iou:.4f}")

        # Save the best model based on IoU
        if val_iou > best_iou:
            best_iou = val_iou
            best_model_wts = model.state_dict()

    print(f"Best Validation IoU: {best_iou:.4f}")
    model.load_state_dict(best_model_wts)
    return model

# Main Function to Set Up the Model and Start Training
def main():
    # Example file paths (replace with your actual paths)
    train_images = ['path/to/train/image1.png', 'path/to/train/image2.png']
    train_masks = ['path/to/train/mask1.png', 'path/to/train/mask2.png']
    val_images = ['path/to/val/image1.png', 'path/to/val/image2.png']
    val_masks = ['path/to/val/mask1.png', 'path/to/val/mask2.png']

    train_transform = Transform()
    val_transform = Transform()

    # Create datasets and dataloaders
    train_dataset = SegmentationDataset(train_images, train_masks, transform=train_transform)
    val_dataset = SegmentationDataset(val_images, val_masks, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Initialize Model, Criterion, and Optimizer
    model = UNet(in_channels=3, out_channels=1).cuda()  # Use GPU if available
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    model = train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

    # Save the trained model
    torch.save(model.state_dict(), 'segmentation_model.pth')
    print("Model saved to segmentation_model.pth")

if __name__ == "__main__":
    main()
