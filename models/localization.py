import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder, IMAGE_SIZE
from .layers import CustomDropout

_FLAT_DIM = 7 * 7 * 512

# Predicts bounding boxes in [cx, cy, w, h] format in pixel space.
# A sigmoid is applied at the end so the values stay within the image range (0 to IMAGE_SIZE).
# This helps avoid weird cases like negative width or height in predictions.
class RegressionHead(nn.Module):

    def __init__(self, dropout_p: float = 0.5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(_FLAT_DIM, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 4),
        )
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(x)) * IMAGE_SIZE


class VGG11Localizer(nn.Module):
    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.head    = RegressionHead(dropout_p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x, return_features=False))