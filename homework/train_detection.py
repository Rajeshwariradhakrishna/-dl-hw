import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from homework.datasets.drive_dataset import load_data
from models import Detector, HOMEWORK_DIR
import os

log_dir = str(HOMEWORK_DIR)
os.makedirs(log_dir, exist_ok=True)

def save_model(model, model_name, log_dir):
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
        true_pos = (preds * targets).sum(dim=(1, 2, 3))
        false_neg = ((1 - preds) * targets).sum(dim=(1, 2, 3))
        false_pos = (preds * (1 - targets)).sum(dim=(1, 2, 3))
        tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth)
        return 1 - tversky.mean()

# 🔹 **Lovász Softmax Loss to directly optimize IoU**
def lovasz_softmax_loss(preds, targets):
    preds = torch.softmax(preds, dim=1)
    targets = targets.float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = (preds + targets).sum(dim=(1, 2, 3)) - intersection
    jaccard_loss = 1 - intersection / (union + 1e-6)
    return jaccard_loss.mean()

# 🔹 **Data Augmentation**
data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),  # 🔹 Higher resolution for better IoU
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 🔹 **Smooth L1 Loss for Depth Error**
def depth_loss(pred, target):
    return F.smooth_l1_loss(pred, target, beta=0.02)  

# 🔹 **Train Function**
def train(model_name="detector", num_epoch=40, lr=5e-4, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pass transforms into data loading
    train_loader = load_data("drive_data/train", batch_size=16, transform=data_transforms)
    val_loader = load_data("drive_data/val", batch_size=16, transform=data_transforms)

    model = Detector().to(device)
    model.train()

    criterion_depth = depth_loss
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

            loss_segmentation = lovasz_softmax_loss(segmentation_pred, segmentation_labels) + TverskyLoss()(segmentation_pred, segmentation_labels)
            loss_depth = criterion_depth(depth_pred, depth_labels)
            loss = loss_segmentation + loss_depth

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

        # 🔹 Evaluate Model
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                segmentation_labels = batch['track'].to(device).long()
                depth_labels = batch['depth'].to(device).unsqueeze(1)

                segmentation_pred, depth_pred = model(images)

                loss_segmentation = lovasz_softmax_loss(segmentation_pred, segmentation_labels) + TverskyLoss()(segmentation_pred, segmentation_labels)
                loss_depth = criterion_depth(depth_pred, depth_labels)
                loss = loss_segmentation + loss_depth

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Validation Loss = {avg_val_loss:.4f}")

        if epoch % 5 == 0:
            print(f"Sample IoU: {segmentation_pred[0].max().item():.3f}, Depth Error: {loss_depth.item():.3f}")

    save_model(model, model_name, log_dir)
