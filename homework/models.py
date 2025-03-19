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


class AttentionGate(nn.Module):
    def __init__(self, in_channels, gate_channels):
        super(AttentionGate, self).__init__()
        self.conv_gate = nn.Conv2d(gate_channels, in_channels, kernel_size=1)
        self.conv_input = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv_attention = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        gate = self.conv_gate(g)
        x_input = self.conv_input(x)
        combined = self.relu(gate + x_input)
        attention = self.sigmoid(self.conv_attention(combined))
        return x * attention


class Detector(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 3):
        super().__init__()
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # Encoder (3 downsampling layers)
        self.encoder1 = self._conv_block(in_channels, 64)
        self.encoder2 = self._conv_block(64, 128)
        self.encoder3 = self._conv_block(128, 256)

        # Attention Gates
        self.attention1 = AttentionGate(128, 256)
        self.attention2 = AttentionGate(64, 128)

        # Decoder (3 upsampling layers)
        self.decoder1 = self._upconv_block(256, 128)
        self.decoder2 = self._upconv_block(128, 64)
        self.decoder3 = self._upconv_block(64, 32)

        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.5)

        # Segmentation Head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Input channels: 32
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, kernel_size=1)  # Output channels: num_classes
        )

        # Depth Head
        self.depth_head = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
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
        e1 = self.encoder1(z)  # 64 channels
        e2 = self.encoder2(e1)  # 128 channels
        e3 = self.encoder3(e2)  # 256 channels

        # Decoder with skip connections and attention gates
        d1 = self.decoder1(e3)
        d1 = self.attention1(e2, d1)  # Pass e2 (128 channels) as gate

        d2 = self.decoder2(d1)
        d2 = self.attention2(e1, d2)  # Pass e1 (64 channels) as gate

        d3 = self.decoder3(d2)

        # Apply dropout
        d3 = self.dropout(d3)

        # Segmentation and Depth Heads
        logits = self.segmentation_head(d3)
        raw_depth = self.depth_head(d3)

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