import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder


class VGG11Localizer(nn.Module):
    """
    Predict bounding box in format:
    [cx, cy, w, h] in pixel space (0–224)
    """

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)

        self.head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4)   # ✅ NO ReLU here
        )

        # weight initialization
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck = self.encoder(x, return_features=False)
        flat = torch.flatten(bottleneck, 1)

        loc = self.head(flat)

        # ✅ Convert to valid bounding box format
        cx = torch.sigmoid(loc[:, 0]) * 224
        cy = torch.sigmoid(loc[:, 1]) * 224
        w  = torch.sigmoid(loc[:, 2]) * 224
        h  = torch.sigmoid(loc[:, 3]) * 224

        return torch.stack([cx, cy, w, h], dim=1)