import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DecoderBlock(nn.Module):
# Upsamples using ConvTranspose2d, then joins with the skip connection and refines it.
# Also takes care of small size mismatches (like off by 1) that happen with odd-sized inputs.

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            _conv_bn_relu(in_ch + skip_ch, out_ch),
            _conv_bn_relu(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            skip = skip[:, :, :x.shape[2], :x.shape[3]]
        return self.conv(torch.cat([x, skip], dim=1))


class VGG11UNet(nn.Module):
    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Decoder is basically the reverse of the encoder, each step doubles the spatial size.
        self.dec5 = DecoderBlock(512, 512, 512)   # 7  -> 14
        self.dec4 = DecoderBlock(512, 512, 256)   # 14 -> 28
        self.dec3 = DecoderBlock(256, 256, 128)   # 28 -> 56
        self.dec2 = DecoderBlock(128, 128,  64)   # 56 -> 112
        self.dec1 = DecoderBlock( 64,  64,  32)   # 112 -> 224

        self.dropout    = CustomDropout(p=dropout_p)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

        self._init_decoder()

    def _init_decoder(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, feats = self.encoder(x, return_features=True)
        d = self.dec5(bottleneck,  feats["b5"])
        d = self.dec4(d,           feats["b4"])
        d = self.dec3(d,           feats["b3"])
        d = self.dec2(d,           feats["b2"])
        d = self.dec1(d,           feats["b1"])
        return self.final_conv(self.dropout(d))