import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder, IMAGE_SIZE
from .layers import CustomDropout

_FLAT_DIM = 7 * 7 * 512   # bottleneck size for 224x224 input


class ClassificationHead(nn.Module):
    # three-layer FC head on the flattened bottleneck
    # BN1d before dropout so normalisation sees the full distribution first
    # dropout prevents co-adaptation of FC neurons

    def __init__(self, num_classes: int = 37, dropout_p: float = 0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(_FLAT_DIM, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class VGG11Classifier(nn.Module):
    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.head    = ClassificationHead(num_classes=num_classes, dropout_p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x, return_features=False))