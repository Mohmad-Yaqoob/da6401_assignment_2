from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from .layers import CustomDropout

# VGG11 usually takes 224×224 images, which end up as a 7×7 feature map after the 5 max-pooling layers.
IMAGE_SIZE = 224

# 3×3 conv with padding=1, so the size stays the same.  
# BatchNorm is applied before ReLU.
def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# VGG11 backbone, mostly following the original paper.
# Blocks are kept separate from pooling, so skip connections can use the higher-resolution features (before pooling).
class VGG11Encoder(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()

        self.block1 = nn.Sequential(_conv_bn_relu(in_channels, 64))
        self.pool1  = nn.MaxPool2d(2, 2)   # 224 -> 112

        self.block2 = nn.Sequential(_conv_bn_relu(64, 128))
        self.pool2  = nn.MaxPool2d(2, 2)   # 112 -> 56

        self.block3 = nn.Sequential(_conv_bn_relu(128, 256), _conv_bn_relu(256, 256))
        self.pool3  = nn.MaxPool2d(2, 2)   # 56 -> 28

        self.block4 = nn.Sequential(_conv_bn_relu(256, 512), _conv_bn_relu(512, 512))
        self.pool4  = nn.MaxPool2d(2, 2)   # 28 -> 14

        self.block5 = nn.Sequential(_conv_bn_relu(512, 512), _conv_bn_relu(512, 512))
        self.pool5  = nn.MaxPool2d(2, 2)   # 14 -> 7

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        # Each block runs first, then pooling is applied.  
        # The feature maps before pooling are saved for skip connections.
        f1 = self.block1(x);       p1 = self.pool1(f1)
        f2 = self.block2(p1);      p2 = self.pool2(f2)
        f3 = self.block3(p2);      p3 = self.pool3(f3)
        f4 = self.block4(p3);      p4 = self.pool4(f4)
        f5 = self.block5(p4);      bottleneck = self.pool5(f5)

        if return_features:
            return bottleneck, {"b1": f1, "b2": f2, "b3": f3, "b4": f4, "b5": f5}
        return bottleneck


VGG11 = VGG11Encoder