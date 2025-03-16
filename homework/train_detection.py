import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os

# Detector Model
class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 16 * 16, 1)  # Adjusted for correct flattening

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)  # Using BCEWithLogitsLoss, no sigmoid needed

# Training Function
def train_detector():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    
    # Load dataset (Assume images are in 'data/' directory with subfolders for classes)
    dataset = datasets.ImageFolder("data", transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = Detector()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    for epoch in range(num_epochs):
        for imgs, labels in dataloader:
            labels = labels.float().unsqueeze(1)  # BCE Loss expects (batch, 1)
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "detector.th")
    print("Model saved as detector.th")

if __name__ == "__main__":
    train_detector()
