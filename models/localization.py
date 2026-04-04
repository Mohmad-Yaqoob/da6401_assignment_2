import torch
import torch.nn as nn
from models.vgg11 import VGG11


class LocalizationModel(nn.Module):
    """
    VGG11 encoder + regression head for single object localization.

    Output: [x_center, y_center, width, height] in PIXEL space (not normalized).
    The sigmoid is removed — ReLU at the end keeps values >= 0,
    and the network learns the actual pixel coordinates directly.

    Loss used during training: MSE + IoULoss (both in pixel space).

    Backbone freezing: we fine-tune the full backbone with a lower lr
    to preserve low-level features while adapting to spatial regression.
    """

    def __init__(self, vgg: VGG11, freeze_backbone: bool = False, dropout_p: float = 0.3):
        super().__init__()

        self.block1  = vgg.block1
        self.block2  = vgg.block2
        self.block3  = vgg.block3
        self.block4  = vgg.block4
        self.block5  = vgg.block5
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        if freeze_backbone:
            for block in [self.block1, self.block2, self.block3,
                          self.block4, self.block5]:
                for p in block.parameters():
                    p.requires_grad = False

        # regression head — outputs pixel-space [cx, cy, w, h]
        self.reg_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.ReLU(inplace=True),   # keeps output >= 0 (pixel coords can't be negative)
        )

        self._init_head()

    def _init_head(self):
        for m in self.reg_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            bbox : (N, 4) — [cx, cy, w, h] in pixel coordinates
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.reg_head(x)


if __name__ == "__main__":
    vgg = VGG11(num_classes=37)
    loc = LocalizationModel(vgg, freeze_backbone=False)
    loc.eval()
    dummy = torch.randn(4, 3, 224, 224)
    out   = loc(dummy)
    print("bbox output shape:", out.shape)   # (4, 4)
    print("bbox sample (pixel space):", out[0])
    assert out.shape == (4, 4)
    print("Localization model check passed.")