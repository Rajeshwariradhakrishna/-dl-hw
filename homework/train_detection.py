import torch
import torch.optim as optim
import torch.nn as nn
import os
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

        # 🔹 Ensure targets have the same shape as preds
        if preds.shape[1] != targets.shape[1]:  
            targets = torch.nn.functional.one_hot(targets.long(), num_classes=preds.shape[1])
            targets = targets.permute(0, 3, 1, 2).float()  # Convert to (B, C, H, W)

        intersection = (preds * targets).sum(dim=(2, 3))
        denominator = (preds + targets).sum(dim=(2, 3))

        dice_score = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1 - dice_score.mean()

def train(model_name="detector", num_epoch=20, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_loader = load_data("drive_data/train")
    val_loader = load_data("drive_data/val")

    # Initialize model
    model = Detector().to(device)
    model.train()

    # Define loss functions
    criterion_segmentation = DiceLoss()
    criterion_depth = nn.L1Loss()

    # Define optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)  # Learning rate decay every 5 epochs

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

        # Step the scheduler to adjust the learning rate
        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epoch}], Loss: {total_train_loss/len(train_loader):.4f}")

        # Save model every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_model(model, f"{model_name}_epoch_{epoch+1}", log_dir)

# Run training
train(
    model_name="detector",
    num_epoch=10,
    lr=1e-3,
)
