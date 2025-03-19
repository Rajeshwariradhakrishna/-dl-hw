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


class Detector(torch.nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 3):
        super().__init__()
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # Encoder
        self.encoder1 = self._conv_block(in_channels, 64)
        self.encoder2 = self._conv_block(64, 128)
        self.encoder3 = self._conv_block(128, 256)
        self.encoder4 = self._conv_block(256, 512)

        # Decoder
        self.decoder1 = self._upconv_block(512, 256)
        self.decoder2 = self._upconv_block(512, 128)
        self.decoder3 = self._upconv_block(256, 64)
        self.decoder4 = self._upconv_block(128, 32)

        # Segmentation Head
        self.segmentation_conv = nn.Conv2d(32, num_classes, kernel_size=1)

        # Depth Head
        self.depth_conv = nn.Conv2d(32, 1, kernel_size=1)

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
        e1 = self.encoder1(z)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with skip connections
        d1 = self.decoder1(e4)
        d1 = torch.cat([d1, e3], dim=1)

        d2 = self.decoder2(d1)
        d2 = torch.cat([d2, e2], dim=1)

        d3 = self.decoder3(d2)
        d3 = torch.cat([d3, e1], dim=1)

        d4 = self.decoder4(d3)

        # Segmentation and Depth Heads
        logits = self.segmentation_conv(d4)
        raw_depth = self.depth_conv(d4)

        return logits, raw_depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)
        depth = raw_depth.squeeze(1)
        return pred, depth

MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(model_name: str, with_weights: bool = False, **model_kwargs) -> torch.nn.Module:
    m = MODEL_FACTORY[model_name](**model_kwargs)
    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(f"Failed to load {model_path.name}") from e
    return m


def save_model(model: torch.nn.Module) -> str:
    model_name = None
    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n
    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")
    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)
    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024