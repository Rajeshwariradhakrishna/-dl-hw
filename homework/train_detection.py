import torch
import torch.optim as optim
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from homework.datasets.drive_dataset import load_data
from models import Detector, HOMEWORK_DIR
from torch.optim.lr_scheduler import CosineAnnealingLR

# Define log directory
log_dir = str(HOMEWORK_DIR)
os.makedirs(log_dir, exist_ok=True)


def save_model(model, model_name, log_dir):
    model_path = os.path.join(log_dir, f"{model_name}.th")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


# IoU Loss for Segmentation
class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.softmax(preds, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float()
        intersection = (preds * targets_one_hot).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3)) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou.mean()


# Gradient Loss for Boundary Prediction
class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, preds, targets):
        # Add a dummy channel dimension to targets
        targets = targets.unsqueeze(1).float()  # (B, H, W) -> (B, 1, H, W)

        # Compute gradients for predictions and targets
        preds_grad_x = torch.abs(torch.gradient(preds, dim=2)[0])  # Gradient along height (dim=2)
        preds_grad_y = torch.abs(torch.gradient(preds, dim=3)[0])  # Gradient along width (dim=3)
        preds_grad = preds_grad_x + preds_grad_y  # Combine gradients

        targets_grad_x = torch.abs(torch.gradient(targets, dim=2)[0])  # Gradient along height (dim=2)
        targets_grad_y = torch.abs(torch.gradient(targets, dim=3)[0])  # Gradient along width (dim=3)
        targets_grad = targets_grad_x + targets_grad_y  # Combine gradients

        # Compute gradient loss
        return torch.mean(torch.abs(preds_grad - targets_grad))


# Combined Loss (Segmentation + Depth + IoU + Gradient)
class CombinedLoss(nn.Module):
    def __init__(self, seg_weight=0.3, depth_weight=0.3, iou_weight=0.4, grad_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.seg_loss = nn.CrossEntropyLoss()
        self.depth_loss = nn.L1Loss()
        self.iou_loss = IoULoss()
        self.grad_loss = GradientLoss()
        self.seg_weight = seg_weight
        self.depth_weight = depth_weight
        self.iou_weight = iou_weight
        self.grad_weight = grad_weight

    def forward(self, seg_preds, depth_preds, seg_targets, depth_targets):
        seg_loss = self.seg_loss(seg_preds, seg_targets)
        depth_loss = self.depth_loss(depth_preds, depth_targets)
        iou_loss = self.iou_loss(seg_preds, seg_targets)
        grad_loss = self.grad_loss(seg_preds, seg_targets)
        return (
            self.seg_weight * seg_loss +
            self.depth_weight * depth_loss +
            self.iou_weight * iou_loss +
            self.grad_weight * grad_loss
        )


# Visualization Function
def visualize_predictions(image, segmentation_pred, depth_pred):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image.cpu().permute(1, 2, 0))
    plt.title("Input Image")

    plt.subplot(1, 3, 2)
    plt.imshow(segmentation_pred.argmax(dim=1).cpu().squeeze(), cmap='jet')
    plt.title("Segmentation Prediction")

    plt.subplot(1, 3, 3)
    plt.im