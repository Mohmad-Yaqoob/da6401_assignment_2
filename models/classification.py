import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout


class VGG11Classifier(nn.Module):
    # encoder + a 3-layer FC head for 37-class breed classification
    # dropout only in FC layers — conv activations are too spatially
    # correlated for per-unit dropout to be effective there

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)

        self.head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )

        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck = self.encoder(x, return_features=False)
        flat = torch.flatten(bottleneck, 1)
        return self.head(flat)