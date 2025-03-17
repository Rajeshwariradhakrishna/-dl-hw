import torch
import torch.optim as optim
import torch.nn as nn
import os
from torchvision import transforms
from homework.datasets.drive_dataset import load_data
from models import Detector, HOMEWORK_DIR

# Define log_dir where you want to save the model
log_dir = str(HOMEWORK_DIR)
os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Define the save_model function
def save_model(model, model_name, log_dir):
    """Save the model's state_dict to the specified directory."""
    model_path = os.path.join(log_dir, f"{model_name}.th")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# Custom Dice Loss to avoid shape mismatch
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)  # Convert logits to probabilities

        # Ensure targets have the same shape as preds
        if preds.shape[1] != targets.shape[1]:
            targets = torch.nn.functional.one_hot(targets.long(), num_classes=preds.shape[1])
            targets = targets.permute(0, 3, 1, 2).float()  # Convert to (B, C, H, W)

        intersection = (preds * targets).sum(dim=(2, 3))
        denominator = (preds + targets).sum(dim=(2, 3))

        dice_score = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1 - dice_score.mean()

# Combined Loss (Dice Loss + Cross-Entropy Loss)
class CombinedLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss(smooth)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        dice_loss = self.dice_loss(preds, targets)
        ce_loss = self.ce_loss(preds, targets)
        return dice_loss + ce_loss

# IoU Metric
def calculate_iou(preds, targets):
    preds = torch.argmax(preds, dim=1)  # Convert logits to class labels
    intersection = (preds & targets).float().sum((1, 2))  # Intersection
    union = (preds | targets).float().sum((1, 2))  # Union
    iou = (intersection + 1e-6) / (union + 1e-6)  # Avoid division by zero
    return iou.mean()

# Data Augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
])

def train(model_name="detector", num_epoch=10, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset with data augmentation
    train_loader = load_data("drive_data/train", transform=train_transform)
    val_loader = load_data("drive_data/val", transform=val_transform)

    # Initialize model
    model = Detector().to(device)
    model.train()

    # Define loss functions
    criterion_segmentation = CombinedLoss()
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
        total_iou = 0
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

                # Compute IoU
                iou = calculate_iou(segmentation_pred, segmentation_labels)
                total_iou += iou.item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_iou = total_iou / len(val_loader)
        print(f"Epoch {epoch+1}: Validation Loss = {avg_val_loss:.4f}, IoU = {avg_iou:.4f}")

    # Save the trained model using the defined save_model function
    save_model(model, model_name, log_dir)

# Run training
train(
    model_name="detector",
    num_epoch=10,
    lr=1e-3,
)