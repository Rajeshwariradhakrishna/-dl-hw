import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import datasets, transforms
from model import Detector
import time

def train_detector():
    # Hyperparameters
    batch_size = 16
    epochs = 10
    learning_rate = 1e-4
    num_classes = 3  # Change this as per your requirement
    model_name = "detector"

    # Data Transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2788, 0.2657, 0.2629], std=[0.2064, 0.1944, 0.2252])
    ])

    # Load dataset (use your specific dataset for detection)
    train_dataset = datasets.CocoDetection(root="train_data_dir", annFile="train_annotations.json", transform=transform)
    val_dataset = datasets.CocoDetection(root="val_data_dir", annFile="val_annotations.json", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss, and optimizer
    model = Detector(in_channels=3, num_classes=num_classes)
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()

            # Forward pass
            logits, depth = model(images)

            # Compute loss
            loss = criterion(logits, targets)
            loss.backward()

            # Update weights
            optimizer.step()

            running_loss += loss.item()

        # Log training loss
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}, Time: {time.time()-start_time}s")

    # Save model after training
    model_path = save_model(model)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    train_detector()
