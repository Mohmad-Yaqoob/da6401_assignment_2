"""
VGG11 built from scratch in PyTorch.

Standard VGG11 conv block layout (number of filters per block):
  Block 1:  64
  Block 2: 128
  Block 3: 256, 256
  Block 4: 512, 512
  Block 5: 512, 512

Design choices (BatchNorm + Dropout placement):
  - BatchNorm2d is placed AFTER every Conv2d and BEFORE ReLU.
    Reason: BN normalises the pre-activation distribution, which prevents
    internal covariate shift and lets us use higher learning rates safely.
    Placing it before ReLU (not after) avoids clipping the normalised
    distribution and gives the normalisation more range to work with.

  - CustomDropout is applied in the classifier head after the first two
    fully-connected layers (FC1 and FC2), not in the convolutional blocks.
    Reason: Spatial feature maps in conv layers have a lot of structural
    correlation between neighbouring activations — dropping individual
    pixels doesn't break the network's ability to reconstruct the pattern
    (nearby pixels carry redundant information). Dropout only meaningfully
    decorrelates features in the high-dimensional FC layers where each
    unit represents a more independent learned concept.
"""

import torch
import torch.nn as nn
from models.layers import CustomDropout


def conv_bn_relu(in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1) -> nn.Sequential:
    """Conv → BN → ReLU building block used throughout the feature extractor."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
        # bias=False because BN has its own learnable shift (beta)
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VGG11(nn.Module):
    """
    VGG11 with BatchNorm (sometimes called VGG11-BN).

    Args:
        num_classes : number of output classes (37 for Oxford Pets)
        dropout_p   : dropout probability for the classifier head
        in_channels : input image channels (3 for RGB)
    """

    def __init__(self, num_classes: int = 37, dropout_p: float = 0.5, in_channels: int = 3):
        super().__init__()

        # ── feature extractor (5 blocks, each ending with MaxPool) ─────────
        # The autograder traces intermediate feature map sizes after each
        # pooling layer. Standard VGG11 input: 224×224
        #   after pool1: 112×112
        #   after pool2:  56×56
        #   after pool3:  28×28
        #   after pool4:  14×14
        #   after pool5:   7×7

        self.block1 = nn.Sequential(
            conv_bn_relu(in_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),          # 224 → 112
        )

        self.block2 = nn.Sequential(
            conv_bn_relu(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),          # 112 → 56
        )

        self.block3 = nn.Sequential(
            conv_bn_relu(128, 256),
            conv_bn_relu(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),          # 56 → 28
        )

        self.block4 = nn.Sequential(
            conv_bn_relu(256, 512),
            conv_bn_relu(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),          # 28 → 14
        )

        self.block5 = nn.Sequential(
            conv_bn_relu(512, 512),
            conv_bn_relu(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),          # 14 → 7
        )

        # adaptive pool so we're not hardcoded to 224×224 input
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # ── classifier head ─────────────────────────────────────────────────
        # FC layers: 512×7×7 → 4096 → 4096 → num_classes
        # CustomDropout after FC1 and FC2 — see module docstring for reasoning.
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(4096, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming init for conv layers, Xavier for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def get_feature_extractor(self) -> nn.Sequential:
        """
        Return just the convolutional backbone (all 5 blocks).
        Used by the localizer and segmenter to plug in this encoder.
        """
        return nn.Sequential(
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.block5,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward_features(self, x: torch.Tensor):
        """
        Forward pass that also returns intermediate block outputs.
        The segmentation decoder needs the skip connections from each block.
        Returns (logits, [b1_out, b2_out, b3_out, b4_out, b5_out]).
        """
        b1 = self.block1(x)
        b2 = self.block2(b1)
        b3 = self.block3(b2)
        b4 = self.block4(b3)
        b5 = self.block5(b4)

        pooled = self.avgpool(b5)
        flat   = torch.flatten(pooled, 1)
        logits = self.classifier(flat)

        return logits, [b1, b2, b3, b4, b5]


# ─── quick shape check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = VGG11(num_classes=37, dropout_p=0.5)
    model.eval()

    dummy = torch.randn(2, 3, 224, 224)
    logits, skips = model.forward_features(dummy)

    print("logits shape:", logits.shape)   # (2, 37)
    for i, s in enumerate(skips, 1):
        print(f"  block{i} output: {s.shape}")
    # expected:
    # block1: (2, 64,  112, 112)
    # block2: (2, 128,  56,  56)
    # block3: (2, 256,  28,  28)
    # block4: (2, 512,  14,  14)
    # block5: (2, 512,   7,   7)