import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder


class VGG11Localizer(nn.Module):
    # predicts bounding box in pixel coordinates [cx, cy, w, h]
    # all values in range 0-224 (pixel space, not normalised)
    # trained with SmoothL1 loss on pixel targets
    # sigmoid at output ensures values are bounded to (0, 224)

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024), nn.ReLU(True),
            nn.Dropout(p=dropout_p),
            nn.Linear(1024, 256), nn.ReLU(True),
            nn.Linear(256, 4),
            # no activation here — sigmoid applied in forward for bounded output
        )
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = torch.flatten(self.encoder(x), 1)
        # sigmoid keeps output in (0,1) then scale to pixel space (0,224)
        # this guarantees valid pixel coordinates and stable training
        return torch.sigmoid(self.head(flat)) * 224.0