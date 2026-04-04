from typing import Dict, Tuple, Union
import torch
import torch.nn as nn
from models.layers import CustomDropout


def _conv_bn_relu(in_ch, out_ch):
    # basic conv block used throughout — BN before ReLU
    # bias=False because BN already handles the shift
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VGG11Encoder(nn.Module):
    # VGG11 conv backbone straight from the 2014 paper
    # BatchNorm added after each conv (not in original but standard now)
    # returns a 512-channel bottleneck or skip maps depending on return_features

    def __init__(self, in_channels: int = 3):
        super().__init__()

        # five blocks, each ending with maxpool
        # for a 224x224 input the spatial sizes go:
        # 224 -> 112 -> 56 -> 28 -> 14 -> 7
        self.block1 = nn.Sequential(
            _conv_bn_relu(in_channels, 64),
            nn.MaxPool2d(2, 2),
        )
        self.block2 = nn.Sequential(
            _conv_bn_relu(64, 128),
            nn.MaxPool2d(2, 2),
        )
        self.block3 = nn.Sequential(
            _conv_bn_relu(128, 256),
            _conv_bn_relu(256, 256),
            nn.MaxPool2d(2, 2),
        )
        self.block4 = nn.Sequential(
            _conv_bn_relu(256, 512),
            _conv_bn_relu(512, 512),
            nn.MaxPool2d(2, 2),
        )
        self.block5 = nn.Sequential(
            _conv_bn_relu(512, 512),
            _conv_bn_relu(512, 512),
            nn.MaxPool2d(2, 2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        s1 = self.block1(x)   # (B, 64,  112, 112)
        s2 = self.block2(s1)  # (B, 128,  56,  56)
        s3 = self.block3(s2)  # (B, 256,  28,  28)
        s4 = self.block4(s3)  # (B, 512,  14,  14)
        s5 = self.block5(s4)  # (B, 512,   7,   7)
        bottleneck = self.avgpool(s5)  # (B, 512, 7, 7)

        if return_features:
            return bottleneck, {
                "block1": s1, "block2": s2, "block3": s3,
                "block4": s4, "block5": s5,
            }
        return bottleneck