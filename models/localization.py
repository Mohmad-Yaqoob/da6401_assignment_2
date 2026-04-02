"""
Object localisation model: VGG11 backbone + a small regression head
that outputs [x_center, y_center, width, height] normalised to [0,1].

Backbone freezing strategy:
    We fine-tune the entire backbone (no freezing) because:
    - The Oxford Pets dataset is visually quite different from ImageNet
      in terms of the specific textures and spatial compositions that
      matter for bounding box regression (vs. classification).
    - Bounding box regression needs the network to understand spatial
      layout, not just category evidence. Frozen features that were
      optimised for classification sometimes struggle here.
    - We still use a lower learning rate for the backbone vs. the head
      to avoid catastrophic forgetting of the useful low-level features.

    This is controlled by the `freeze_backbone` flag so you can switch
    between strategies for the W&B transfer learning comparison.
"""

import torch
import torch.nn as nn
from models.vgg11 import VGG11


class LocalizationModel(nn.Module):
    """
    VGG11 encoder + regression decoder for single-object localisation.

    Args:
        vgg          : a VGG11 instance (pre-trained or fresh)
        freeze_backbone: if True, conv blocks' weights won't be updated
        dropout_p    : dropout for the regression head
    """

    def __init__(self, vgg: VGG11, freeze_backbone: bool = False, dropout_p: float = 0.3):
        super().__init__()

        # ── encoder — borrow the 5 conv blocks from VGG11 ──────────────────
        self.block1 = vgg.block1
        self.block2 = vgg.block2
        self.block3 = vgg.block3
        self.block4 = vgg.block4
        self.block5 = vgg.block5
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        if freeze_backbone:
            for block in [self.block1, self.block2, self.block3, self.block4, self.block5]:
                for param in block.parameters():
                    param.requires_grad = False

        # ── regression head ─────────────────────────────────────────────────
        # Takes the 512×7×7 feature map and regresses 4 box coordinates.
        # Sigmoid on the output keeps values in (0, 1) which matches our
        # normalised [cx, cy, w, h] target space.
        self.reg_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),   # standard dropout is fine here (not autograded)

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 4),
            nn.Sigmoid(),              # output in (0, 1) — matches normalised coords
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
            bbox : (N, 4) tensor of [cx, cy, w, h] in [0, 1]
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        bbox = self.reg_head(x)
        return bbox


# ─── shape check ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    vgg = VGG11(num_classes=37)
    loc = LocalizationModel(vgg, freeze_backbone=False)
    loc.eval()

    dummy = torch.randn(4, 3, 224, 224)
    out   = loc(dummy)
    print("bbox output shape:", out.shape)   # (4, 4)
    print("bbox sample:", out[0])            # all values in (0,1)
    assert out.shape == (4, 4)
    assert out.min() >= 0 and out.max() <= 1
    print("Localization model check passed.")