import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from torchvision import transforms
from torch.utils.data import Dataset

# ----------------------------
# 1. Define the Dice Loss function
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)  # Ensure values are between 0 and 1
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

# Combined loss function: CrossEntropy + DiceLoss
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, preds, targets):
        return self.ce(preds, targets) + self.dice(preds, targets)


# ----------------------------
# 2. Define the U-Net model
class UNetModel(nn.Module):
    def __init__(self):
        super(UNetModel, self).__init__()
        self.model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1)

    def forward(self, x):
        return self.model(x)

# ----------------------------
# 3. Define dataset and augmentations
class CustomDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        if self.transform:
            image = self.transform(image)

        return image, mask

# ----------------------------
# 4. Apply data augmentations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

# ----------------------------
# 5. Training Loop
def train_model(model, dataloader, optimizer, criterion, num_epochs=10):
    best_model_wts = None
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_iou = 0.0
        
        for inputs, labels in dataloader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Compute IoU (Intersection over Union)
            preds = torch.sigmoid(outputs) > 0.5
            intersection = (preds * labels).sum().float()
            union = preds.sum() + labels.sum() - intersection
            iou = intersection / union if union != 0 else 0
            running_iou += iou.item()

        epoch_loss = running_loss / len(dataloader)
        epoch_iou = running_iou / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, IoU: {epoch_iou:.4f}')

        # Save model with best IoU
        if epoch_iou > best_loss:
            best_loss = epoch_iou
            best_model_wts = model.state_dict()

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# ----------------------------
# 6. Initialize Model, Optimizer, Criterion, and DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetModel().to(device)

# Using Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Combined loss
criterion = CombinedLoss()

# Example data (replace with your actual data)
train_images = []  # Add your image data here
train_masks = []   # Add your mask data here

# Create dataset and dataloader
train_dataset = CustomDataset(train_images, train_masks, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# ----------------------------
# 7. Start training
model = train_model(model, train_loader, optimizer, criterion, num_epochs=10)

# ----------------------------
# 8. Save the model
torch.save(model.state_dict(), "detector.th")
print("Model saved to detector.th")
