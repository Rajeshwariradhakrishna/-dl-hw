import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import os
from homework.datasets.drive_dataset import load_data
from models import Detector, HOMEWORK_DIR  

# Define log_dir where you want to save the model
log_dir = str(HOMEWORK_DIR)
os.makedirs(log_dir, exist_ok=True)

# Define the save_model function
def save_model(model, model_name, log_dir):
    model_path = os.path.join(log_dir, f"{model_name}.th")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# Cross-Entropy Loss for multi-class segmentation
criterion_segmentation = nn.CrossEntropyLoss()  # CrossEntropyLoss expects integer class labels

# Confusion Matrix Calculation
def calculate_confusion_matrix(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true.cpu().numpy().flatten(), y_pred.cpu().numpy().flatten(), labels=range(num_classes))
    return cm

def train(model_name="detector", num_epoch=20, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_loader = load_data("drive_data/train")
    val_loader = load_data("drive_data/val")

    # Initialize model
    model = Detector().to(device)
    model.train()

    # Optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    for epoch in range(num_epoch):
        total_train_loss = 0
        all_preds = []
        all_labels = []

        # Training loop
        model.train()
        for batch in train_loader:
            images = batch['image'].to(device)
            segmentation_labels = batch['track'].to(device).long()

            optimizer.zero_grad()
            segmentation_pred, _ = model(images)

            # Compute loss
            loss = criterion_segmentation(segmentation_pred, segmentation_labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # Collect predictions and true labels for confusion matrix
            preds = segmentation_pred.argmax(dim=1)
            all_preds.append(preds)
            all_labels.append(segmentation_labels)

        # Confusion matrix after each epoch
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        cm = calculate_confusion_matrix(all_labels, all_preds, num_classes=3)  # Adjust for the number of classes
        print(f"Confusion Matrix after epoch {epoch + 1}:\n{cm}")

        # Learning rate scheduler step
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
