"""
U-Net style segmentation model using VGG11 as the contracting path.

Architecture overview:
    Encoder (VGG11 blocks):
        Input  224×224 → block1 → 112×112 (64ch)
                       → block2 →  56×56 (128ch)
                       → block3 →  28×28 (256ch)
                       → block4 →  14×14 (512ch)
                       → block5 →   7×7  (512ch)

    Decoder (mirrored, transposed convolutions):
        7×7   → TransConv → 14×14  + skip from block4 (512ch) → conv → 512ch
        14×14  → TransConv → 28×28  + skip from block3 (256ch) → conv → 256ch
        28×28  → TransConv → 56×56  + skip from block2 (128ch) → conv → 128ch
        56×56  → TransConv → 112×112 + skip from block1 (64ch)  → conv → 64ch
        112×112 → TransConv → 224×224 → conv → num_classes (3 for trimaps)

Loss function: CrossEntropyLoss
    Reason: The trimap has 3 classes (foreground / background / border).
    CE handles multi-class pixel classification naturally and has better
    gradient behaviour than binary losses. We do NOT weight the classes
    here by default, but class_weight can be passed to handle imbalance
    if the border class is underrepresented.

Upsampling: ONLY ConvTranspose2d is used — no bilinear interpolation,
    no unpooling. This satisfies the assignment constraint.

Skip connections: encoder feature maps are concatenated with the
    upsampled decoder maps along the channel dimension before the next
    conv block — classic U-Net feature fusion.
"""

import torch
import torch.nn as nn
from models.vgg11 import VGG11


def dec_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two conv layers that process concatenated (upsampled + skip) features."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SegmentationModel(nn.Module):
    """
    U-Net with VGG11 encoder.

    Args:
        vgg           : VGG11 instance (may be pre-trained)
        num_classes   : 3 for Oxford Pets trimaps
        freeze_backbone: whether to freeze encoder weights
    """

    def __init__(self, vgg: VGG11, num_classes: int = 3, freeze_backbone: bool = False):
        super().__init__()

        # ── encoder — plug in VGG11 blocks ─────────────────────────────────
        self.enc1 = vgg.block1   # → (B, 64,  112, 112)
        self.enc2 = vgg.block2   # → (B, 128,  56,  56)
        self.enc3 = vgg.block3   # → (B, 256,  28,  28)
        self.enc4 = vgg.block4   # → (B, 512,  14,  14)
        self.enc5 = vgg.block5   # → (B, 512,   7,   7)

        if freeze_backbone:
            for enc in [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]:
                for p in enc.parameters():
                    p.requires_grad = False

        # ── decoder — transposed convs for upsampling ───────────────────────
        # Naming: up{i} upsamples, dec{i} processes after skip concat

        # 7×7 → 14×14
        self.up5   = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5  = dec_block(512 + 512, 512)   # concat with enc4 output (512ch)

        # 14×14 → 28×28
        self.up4   = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4  = dec_block(256 + 256, 256)   # concat with enc3 output (256ch)

        # 28×28 → 56×56
        self.up3   = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3  = dec_block(128 + 128, 128)   # concat with enc2 output (128ch)

        # 56×56 → 112×112
        self.up2   = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2  = dec_block(64 + 64, 64)      # concat with enc1 output (64ch)

        # 112×112 → 224×224
        self.up1   = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1  = dec_block(32, 32)            # no skip at this stage

        # final 1×1 conv to get per-pixel class scores
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

        self._init_decoder()

    def _init_decoder(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 3, 224, 224)
        Returns:
            logits : (B, num_classes, 224, 224) — raw scores, no softmax
        """
        # ── encoder forward — save skip connections ─────────────────────────
        s1 = self.enc1(x)    # (B, 64,  112, 112)
        s2 = self.enc2(s1)   # (B, 128,  56,  56)
        s3 = self.enc3(s2)   # (B, 256,  28,  28)
        s4 = self.enc4(s3)   # (B, 512,  14,  14)
        s5 = self.enc5(s4)   # (B, 512,   7,   7)

        # ── decoder with skip connections ───────────────────────────────────
        d = self.up5(s5)              # (B, 512, 14, 14)
        d = self.dec5(torch.cat([d, s4], dim=1))   # concat → (B,1024,14,14) → (B,512,14,14)

        d = self.up4(d)               # (B, 256, 28, 28)
        d = self.dec4(torch.cat([d, s3], dim=1))   # concat → (B,512,28,28) → (B,256,28,28)

        d = self.up3(d)               # (B, 128, 56, 56)
        d = self.dec3(torch.cat([d, s2], dim=1))   # concat → (B,256,56,56) → (B,128,56,56)

        d = self.up2(d)               # (B, 64, 112, 112)
        d = self.dec2(torch.cat([d, s1], dim=1))   # concat → (B,128,112,112) → (B,64,112,112)

        d = self.up1(d)               # (B, 32, 224, 224)
        d = self.dec1(d)              # (B, 32, 224, 224)

        return self.final(d)          # (B, num_classes, 224, 224)


# ─── shape check ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    vgg = VGG11(num_classes=37)
    seg = SegmentationModel(vgg, num_classes=3)
    seg.eval()

    dummy = torch.randn(2, 3, 224, 224)
    out = seg(dummy)
    print("segmentation output shape:", out.shape)  # (2, 3, 224, 224)
    assert out.shape == (2, 3, 224, 224)
    print("Segmentation model check passed.")