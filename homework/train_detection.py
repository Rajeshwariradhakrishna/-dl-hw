import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
from homework.datasets.drive_dataset import load_data
from models import Detector, HOMEWORK_DIR
import torchvision.transforms as transforms

# Define log_dir where you want to save the model
log_dir = str(HOMEWORK_DIR)
os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Define the save_model function
def save_model(model, model_name, log_dir):
    """Save the model's state_dict to the specified directory."""
    model_path = os.path.join(log_dir, f"{model_name}.th")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# 🔹 **Tversky Loss for better IoU optimization**
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)  
        targets = F.one_hot(targets, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float()
        true_pos = (preds * targets).sum(dim=(1, 2, 3))
        false_neg = ((1 - preds) * targets).sum(dim=(1, 2, 3))
        false_pos = (preds * (1 - targets)).sum(dim=(1, 2, 3))
        tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth)
        return 1 - tversky.mean()

# 🔹 **Lovász Softmax Loss to directly optimize IoU**
def lovasz_softmax_loss(preds, targets):
    preds = torch.softmax(preds, dim=1)
    targets = F.one_hot(targets, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = (preds + targets).sum(dim=(1, 2, 3)) - intersection
    jaccard_loss = 1 - (intersection + 1e-6) / (union + 1e-6)
    return jaccard_loss.mean()

# 🔹 **Dice Loss for better segmentation**
def dice_loss(preds, targets):
    preds = torch.sigmoid(preds)
    targets = F.one_hot(targets, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = (preds + targets).sum(dim=(1, 2, 3))
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    return 1 - dice.mean()

# 🔹 **Scale-Invariant Depth Loss**
def scale_invariant_depth_loss(pred, target):
    log_diff = torch.log(pred.clamp(min=1e-6)) - torch.log(target.clamp(min=1e-6))
    return torch.sqrt((log_diff ** 2).mean() - 0.5 * (log_diff.mean() ** 2))

# 🔹 **Data Augmentation**
data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),  
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(3),
    transforms.RandomCrop((224, 224)),  # Added random cropping
    transforms.Lambda(lambda x: x if isinstance(x, torch.Tensor) else transforms.ToTensor()(x)),
    transforms.Normalize(mean=[0.2788, 0.2657, 0.2629], std=[0.2064, 0.1944, 0.2252])
])

def apply_transforms(batch, transform):
    """Apply transformations to the batch of images."""
    batch['image'] = torch.stack([transform(img) for img in batch['image']])
    return batch

# 🔹 **IoU Calculation**
def calculate_iou(preds, targets):
    preds = torch.argmax(preds, dim=1)  # Convert logits to class indices
    targets = targets.squeeze(1)  # Remove channel dimension if present
    intersection = (preds & targets).float().sum((1, 2))  # Intersection
    union = (preds | targets).float().sum((1, 2))  # Union
    iou = (intersection + 1e-6) / (union + 1e-6)  # Avoid division by zero
    return iou.mean()

def train(model_name="detector", num_epoch=40, lr=5e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_loader = load_data("drive_data/train")
    val_loader = load_data("drive_data/val")

    # Initialize model
    model = Detector().to(device)
    model.train()

    # Define optimizer with gradient clipping
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Learning rate scheduler
    max_grad_norm = 1.0  

    for epoch in range(num_epoch):
        total_train_loss = 0
        total_train_iou = 0
        model.train()
        for batch in train_loader:
            batch = apply_transforms(batch, data_transforms)
            images = batch['image'].to(device)
            segmentation_labels = batch['track'].to(device).long()
            depth_labels = batch['depth'].to(device).unsqueeze(1)

            optimizer.zero_grad()
            segmentation_pred, depth_pred = model(images)

            if torch.isnan(segmentation_pred).any() or torch.isnan(depth_pred).any():
                raise ValueError("Model output contains NaN values.")

            # Resize predictions if necessary
            if segmentation_pred.shape[-2:] != segmentation_labels.shape[-2:]:
                segmentation_pred = F.interpolate(segmentation_pred, size=segmentation_labels.shape[-2:], mode='bilinear', align_corners=False)
            if depth_pred.shape[-2:] != depth_labels.shape[-2:]:
                depth_pred = F.interpolate(depth_pred, size=depth_labels.shape[-2:], mode='bilinear', align_corners=False)

            # Weighted loss for segmentation
            loss_segmentation = (
                0.5 * lovasz_softmax_loss(segmentation_pred, segmentation_labels) +
                0.3 * TverskyLoss()(segmentation_pred, segmentation_labels) +
                0.2 * dice_loss(segmentation_pred, segmentation_labels)
            )

            # Depth loss
            loss_depth = scale_invariant_depth_loss(depth_pred, depth_labels)

            # Total loss
            loss = loss_segmentation + loss_depth

            if torch.isnan(loss):
                raise ValueError("Loss contains NaN values.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            # Calculate IoU
            iou = calculate_iou(segmentation_pred, segmentation_labels)
            total_train_loss += loss.item()
            total_train_iou += iou.item()

        scheduler.step()  # Update learning rate

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_iou = total_train_iou / len(train_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Train IoU = {avg_train_iou:.4f}")

        # Validation loop
        model.eval()
        total_val_loss = 0
        total_val_iou = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = apply_transforms(batch, data_transforms)
                images = batch['image'].to(device)
                segmentation_labels = batch['track'].to(device).long()
                depth_labels = batch['depth'].to(device).unsqueeze(1)

                segmentation_pred, depth_pred = model(images)

                if segmentation_pred.shape[-2:] != segmentation_labels.shape[-2:]:
                    segmentation_pred = F.interpolate(segmentation_pred, size=segmentation_labels.shape[-2:], mode='bilinear', align_corners=False)
                if depth_pred.shape[-2:] != depth_labels.shape[-2:]:
                    depth_pred = F.interpolate(depth_pred, size=depth_labels.shape[-2:], mode='bilinear', align_corners=False)

                # Weighted loss for segmentation
                loss_segmentation = (
                    0.5 * lovasz_softmax_loss(segmentation_pred, segmentation_labels) +
                    0.3 * TverskyLoss()(segmentation_pred, segmentation_labels) +
                    0.2 * dice_loss(segmentation_pred, segmentation_labels)
                )
                loss_depth = scale_invariant_depth_loss(depth_pred, depth_labels)
                loss = loss_segmentation + loss_depth

                # Calculate IoU
                iou = calculate_iou(segmentation_pred, segmentation_labels)
                total_val_loss += loss.item()
                total_val_iou += iou.item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_iou = total_val_iou / len(val_loader)
        print(f"Epoch {epoch+1}: Validation Loss = {avg_val_loss:.4f}, Validation IoU = {avg_val_iou:.4f}")

    save_model(model, model_name, log_dir)

if __name__ == "__main__":
    train()