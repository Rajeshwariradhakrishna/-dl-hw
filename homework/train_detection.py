import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import segmentation_models_pytorch as smp
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np

# ----------------------------
# 1. Define the U-Net model
class UNetModel(nn.Module):
    def __init__(self):
        super(UNetModel, self).__init__()
        self.model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1)

    def forward(self, x):
        return self.model(x)

# ----------------------------
# 2. Define dataset and augmentations
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
# 3. Apply data augmentations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

# ----------------------------
# 4. Train the model function with confusion matrix and lr scheduler
def train_model(model, dataloader, optimizer, criterion, scheduler, num_epochs=10):
    best_model_wts = None
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for inputs, labels in dataloader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate confusion matrix
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        # Flatten all predictions and labels for confusion matrix
        all_preds = np.concatenate(all_preds).flatten()
        all_labels = np.concatenate(all_labels).flatten()

        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        print(f"Confusion Matrix for Epoch {epoch+1}: \n{cm}")

        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}')

        # Step the scheduler
        scheduler.step(epoch_loss)

        # Save model with best loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = model.state_dict()

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# ----------------------------
# 5. Initialize Model, Optimizer, Criterion, and DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetModel().to(device)

# Using Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# CrossEntropy Loss
criterion = nn.CrossEntropyLoss()

# Learning rate scheduler (Reduce LR on Plateau)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)

# Example data (replace with your actual data)
train_images = []  # Add your image data here
train_masks = []   # Add your mask data here

# Create dataset and dataloader
train_dataset = CustomDataset(train_images, train_masks, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# ----------------------------
# 6. Start training
model = train_model(model, train_loader, optimizer, criterion, scheduler, num_epochs=10)

# ----------------------------
# 7. Save the model
torch.save(model.state_dict(), "detector.th")
print("Model saved to detector.th")
