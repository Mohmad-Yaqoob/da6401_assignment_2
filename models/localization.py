import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder


class VGG11Localizer(nn.Module):
    # predicts a single bounding box in pixel coordinates
    # output format: [x_center, y_center, width, height] — all in pixels
    # ReLU at the end keeps coords non-negative, which makes sense for pixel space
    # trained with MSE + IoU loss combined

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)

        self.head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.ReLU(inplace=True),
        )

        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck = self.encoder(x, return_features=False)
        flat = torch.flatten(bottleneck, 1)
        return self.head(flat)