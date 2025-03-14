import torch
import torch.optim as optim
import torch.nn as nn
import os
import torch.nn.functional as F
import torchvision.transforms as transforms
from homework.datasets.drive_dataset import load_data
from models import Detector, HOMEWORK_DIR  

# Set logging directory
log_dir = str(HOMEWORK_DIR)
os.makedirs(log_dir, exist_ok=True)

def save_model(model, model_name, log_dir):
    """ Save the trained model """
    model_path = os.path.join(log_dir, f"{model_name}.th")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# 🔹 **Data Augmentation for Segmentation Improvement**
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.GaussianBlur(3),
    transforms.ToTensor()
])

# 🔹 **Soft Dice Loss for better IoU performance**
def soft_dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)  
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

# 🔹 **Focal Loss to balance class distribution in segmentation**
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()

# 🔹 **Smooth L1 Loss (Huber Loss) for Depth Error**
def depth_loss(pred, target):
    return F.smooth_l1_loss(pred, target, beta=0.05)  

# 🔹 **Combined Loss for Segmentation**
def combined_segmentation_loss(output, target):
    ce_loss = F.cross_entropy(output, target)
    dice = soft_dice_loss(output, target)
    focal = FocalLoss()(output, target)
    return ce_loss + dice + focal  # Weighted combination

# 🔹 **Train Function**
def train(model_name="detector", num_epoch=30, lr=5e-4, batch_size=8):  # 🔹 Reduced batch size for stability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset with augmentation
    train_loader = load_data("drive_data/train", transform=data_transforms, batch_size=batch_size)
    val_loader = load_data("drive_data/val", batch_size=batch_size)

    # Initialize model
    model = Detector().to(device)
    model.train()

    # Define loss functions
    criterion_depth = depth_loss  # Using Smooth L1 Loss for depth estimation

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epoch):
        total_train_loss = 0
        model.train()

        for batch in train_loader:
            images = batch['image'].to(device)
            segmentation_labels = batch['track'].to(device).long()
            depth_labels = batch['depth'].to(device).unsqueeze(1)

            optimizer.zero_grad()
            segmentation_pred, depth_pred = model(images)

            # Compute loss
            loss_segmentation = combined_segmentation_loss(segmentation_pred, segmentation_labels)
            loss_depth = criterion_depth(depth_pred, depth_labels)
            loss = loss_segmentation + loss_depth

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                segmentation_labels = batch['track'].to(device).long()
                depth_labels = batch['depth'].to(device).unsqueeze(1)

                segmentation_pred, depth_pred = model(images)

                loss_segmentation = combined_segmentation_loss(segmentation_pred, segmentation_labels)
                loss_depth = criterion_depth(depth_pred, depth_labels)
                loss = loss_segmentation + loss_depth

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Validation Loss = {avg_val_loss:.4f}")

        # 🔹 Debugging every 5 epochs
        if epoch % 5 == 0:
            print(f"Sample IoU: {segmentation_pred[0].max().item():.3f}, Depth Error: {loss_depth.item():.3f}")

    save_model(model, model_name, log_dir)
