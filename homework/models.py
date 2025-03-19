from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 6):
        super().__init__()
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # Convolutional Layers with BatchNorm and ReLU
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Convolutional layers with ReLU and BatchNorm
        x = self.pool(torch.relu(self.bn1(self.conv1(z))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))

        # Flatten the output for fully connected layer
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self(x).argmax(dim=1)


class Detector(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 3):
        super().__init__()
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # Encoder (3 downsampling layers)
        self.encoder1 = self._conv_block(in_channels, 64)  # (B, 64, H/2, W/2)
        self.encoder2 = self._conv_block(64, 128)          # (B, 128, H/4, W/4)
        self.encoder3 = self._conv_block(128, 256)         # (B, 256, H/8, W/8)

        # Decoder (3 upsampling layers)
        self.decoder1 = self._upconv_block(256, 128)       # (B, 128, H/4, W/4)
        self.decoder2 = self._upconv_block(128 + 128, 64)  # (B, 64, H/2, W/2)
        self.decoder3 = self._upconv_block(64 + 64, 32)    # (B, 32, H, W)

        # Segmentation Head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

        # Depth Head
        self.depth_head = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def _upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Normalize input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Encoder
        e1 = self.encoder1(z)  # (B, 64, H/2, W/2)
        e2 = self.encoder2(e1)  # (B, 128, H/4, W/4)
        e3 = self.encoder3(e2)  # (B, 256, H/8, W/8)

        # Decoder with skip connections
        d1 = self.decoder1(e3)  # (B, 128, H/4, W/4)
        d1 = torch.cat([d1, e2], dim=1)  # Skip connection with e2 (128 channels)

        d2 = self.decoder2(d1)  # (B, 64, H/2, W/2)
        d2 = torch.cat([d2, e1], dim=1)  # Skip connection with e1 (64 channels)

        d3 = self.decoder3(d2)  # (B, 32, H, W)

        # Segmentation and Depth Heads
        logits = self.segmentation_head(d3)  # (B, num_classes, H, W)
        raw_depth = self.depth_head(d3)  # (B, 1, H, W)

        return logits, raw_depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)  # (B, H, W)
        depth = raw_depth.squeeze(1)  # (B, H, W)
        return pred, depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(model_name: str, with_weights: bool = False, **model_kwargs) -> nn.Module:
    m = MODEL_FACTORY[model_name](**model_kwargs)
    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(f"Failed to load {model_path.name}") from e
    return m


def save_model(model: nn.Module) -> str:
    model_name = None
    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n
    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")
    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)
    return output_path


def calculate_model_size_mb(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024