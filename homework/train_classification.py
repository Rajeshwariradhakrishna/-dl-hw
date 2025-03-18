import torch
from homework.models import Classifier
from homework.datasets.classification_dataset import load_data
from homework.metrics import AccuracyMetric

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model_name, num_epoch, lr):
    # Load dataset and create DataLoader
    train_loader = load_data("classification_data/train", transform_pipeline="aug", shuffle=True)
    val_loader = load_data("classification_data/val", transform_pipeline="default", shuffle=False)

    # Define model, loss, and optimizer
    model = Classifier().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Initialize metrics
    train_metric = AccuracyMetric()
    val_metric = AccuracyMetric()

    # Training loop
    for epoch in range(num_epoch):
        model.train()
        train_metric.reset()  # Reset metric before each epoch
        for batch in train_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update training metric
            _, preds = torch.max(outputs, dim=1)
            train_metric.add(preds, labels)

        # Validation
        model.eval()
        val_metric.reset()  # Reset metric before validation
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                # Update validation metric
                _, preds = torch.max(outputs, dim=1)
                val_metric.add(preds, labels)

        # Log metrics
        train_accuracy = train_metric.compute()["accuracy"]
        val_accuracy = val_metric.compute()["accuracy"]
        print(f"Epoch {epoch+1}, Train Accuracy: {train_accuracy}, Val Accuracy: {val_accuracy}")

    # Save the trained model
    from homework.models import save_model
    save_model(model)

# Run training
train(
    model_name="classifier",
    num_epoch=10,
    lr=1e-3,
)