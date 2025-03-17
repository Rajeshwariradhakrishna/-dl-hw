import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from models import Detector, save_model

# Training Function
def train():
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((96, 128)),  # Match the input size expected by the model
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.ToTensor(),
    ])

    # Load dataset (assume images are in 'data/' directory with subfolders for classes)
    dataset = datasets.ImageFolder("data", transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Initialize model, loss functions, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Detector().to(device)
    criterion_seg = nn.CrossEntropyLoss()  # For segmentation
    criterion_depth = nn.MSELoss()  # For depth estimation
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Learning rate scheduler

    # Training loop
    num_epochs = 50  # Increase epochs
    for epoch in range(num_epochs):
        model.train()
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            segmentation_logits, raw_depth = model(imgs)

            # Compute losses (assume labels contain both segmentation and depth)
            loss_seg = criterion_seg(segmentation_logits, labels['segmentation'])
            loss_depth = criterion_depth(raw_depth.squeeze(1), labels['depth'])
            total_loss = loss_seg + loss_depth

            # Backpropagation
            total_loss.backward()
            optimizer.step()

        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.4f}")

    # Save the model
    save_model(model)
    print("Model saved as detector.th")


if __name__ == "__main__":
    train()