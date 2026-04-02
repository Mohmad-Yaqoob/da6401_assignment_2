"""
Unified Multi-Task Pipeline.

A single forward pass yields:
    1. Classification logits  (B, 37)
    2. Bounding box coords    (B, 4)  — [cx, cy, w, h] in [0,1]
    3. Segmentation mask      (B, 3, H, W) — raw logits

Shared backbone: VGG11 conv blocks (enc1–enc5).
The three heads branch off the same feature representation.
Classification and localisation branch from the pooled 7×7 features.
Segmentation gets all 5 skip connections for the U-Net decoder.
"""

import torch
import torch.nn as nn
from models.vgg11 import VGG11
from models.layers import CustomDropout


class MultiTaskModel(nn.Module):
    """
    Unified model branching into classification, localisation, and segmentation.

    Args:
        num_classes  : number of breed classes (37)
        dropout_p    : dropout probability in classification head
        seg_classes  : number of segmentation classes (3 for trimaps)
    """

    def __init__(self, num_classes: int = 37, dropout_p: float = 0.5, seg_classes: int = 3):
        super().__init__()

        # ── shared backbone (VGG11 conv blocks) ────────────────────────────
        vgg = VGG11(num_classes=num_classes, dropout_p=dropout_p)

        self.enc1 = vgg.block1   # → (B, 64,  112, 112)
        self.enc2 = vgg.block2   # → (B, 128,  56,  56)
        self.enc3 = vgg.block3   # → (B, 256,  28,  28)
        self.enc4 = vgg.block4   # → (B, 512,  14,  14)
        self.enc5 = vgg.block5   # → (B, 512,   7,   7)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        flat_dim = 512 * 7 * 7

        # ── task 1: classification head ─────────────────────────────────────
        self.cls_head = nn.Sequential(
            nn.Linear(flat_dim, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(4096, num_classes),
        )

        # ── task 2: localisation head ───────────────────────────────────────
        self.loc_head = nn.Sequential(
            nn.Linear(flat_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

        # ── task 3: segmentation decoder (U-Net style) ──────────────────────
        # mirrors the encoder depth
        self.up5  = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = self._dec_block(512 + 512, 512)

        self.up4  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._dec_block(256 + 256, 256)

        self.up3  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._dec_block(128 + 128, 128)

        self.up2  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._dec_block(64 + 64, 64)

        self.up1  = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._dec_block(32, 32)

        self.seg_final = nn.Conv2d(32, seg_classes, kernel_size=1)

        self._init_weights()

    @staticmethod
    def _dec_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        Single forward pass — all three tasks at once.

        Args:
            x : (B, 3, H, W)

        Returns:
            cls_logits : (B, 37)
            bbox       : (B, 4) in [0,1]
            seg_logits : (B, 3, H, W)
        """
        # ── shared encoder ──────────────────────────────────────────────────
        s1 = self.enc1(x)    # (B, 64,  112, 112)
        s2 = self.enc2(s1)   # (B, 128,  56,  56)
        s3 = self.enc3(s2)   # (B, 256,  28,  28)
        s4 = self.enc4(s3)   # (B, 512,  14,  14)
        s5 = self.enc5(s4)   # (B, 512,   7,   7)

        # ── pooled features for cls + loc heads ─────────────────────────────
        pooled = self.avgpool(s5)
        flat   = torch.flatten(pooled, 1)   # (B, 512*7*7)

        cls_logits = self.cls_head(flat)    # (B, 37)
        bbox       = self.loc_head(flat)    # (B, 4)

        # ── segmentation decoder with skip connections ───────────────────────
        d = self.up5(s5)
        d = self.dec5(torch.cat([d, s4], dim=1))

        d = self.up4(d)
        d = self.dec4(torch.cat([d, s3], dim=1))

        d = self.up3(d)
        d = self.dec3(torch.cat([d, s2], dim=1))

        d = self.up2(d)
        d = self.dec2(torch.cat([d, s1], dim=1))

        d = self.up1(d)
        d = self.dec1(d)
        seg_logits = self.seg_final(d)      # (B, 3, 224, 224)

        return cls_logits, bbox, seg_logits


# ─── shape check ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = MultiTaskModel(num_classes=37, dropout_p=0.5, seg_classes=3)
    model.eval()

    dummy = torch.randn(2, 3, 224, 224)
    cls, box, seg = model(dummy)

    print("cls  shape:", cls.shape)   # (2, 37)
    print("bbox shape:", box.shape)   # (2, 4)
    print("seg  shape:", seg.shape)   # (2, 3, 224, 224)
    assert cls.shape == (2, 37)
    assert box.shape == (2, 4)
    assert seg.shape == (2, 3, 224, 224)
    print("MultiTask model check passed.")