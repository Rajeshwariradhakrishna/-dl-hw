import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
from homework.datasets.drive_dataset import load_data
from models import Detector, HOMEWORK_DIR
import torchvision.transforms as transforms

log_dir = str(HOMEWORK_DIR)
os.makedirs(log_dir, exist_ok=True)

def save_model(model, model_name, log_dir):
    model_path = os.path.join(log_dir, f"{model_name}.th")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

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

def lovasz_softmax_loss(preds, targets):
    preds = torch.softmax(preds, dim=1)
    targets = F.one_hot(targets, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = (preds + targets).sum(dim=(1, 2, 3)) - intersection
    jaccard_loss = 1 - (intersection + 1e-6) / (union + 1e-6)
    return jaccard_loss.mean()

def dice_loss(preds, targets):
    preds = torch.sigmoid(preds)
    targets = F.one_hot(targets, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1 - (2 * intersection + 1e-6) / (union + 1e-6)

def train(model, train_loader, val_loader, epochs=10, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = TverskyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

        if (epoch + 1) % 5 == 0:
            save_model(model, "detector_model", log_dir)

def main():
    train_data, val_data = load_data()
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=16, shuffle=False)

    model = Detector()
    train(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
