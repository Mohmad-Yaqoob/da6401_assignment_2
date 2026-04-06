import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder


def _dec_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VGG11UNet(nn.Module):
    """
    U-Net style segmentation model using VGG11 encoder
    Output: [B, num_classes, H, W]
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Decoder
        self.up5  = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec5 = _dec_block(512 + 512, 512)

        self.up4  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = _dec_block(256 + 256, 256)

        self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = _dec_block(128 + 128, 128)

        self.up2  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = _dec_block(64 + 64, 64)

        self.up1  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = _dec_block(32, 32)

        self.final = nn.Conv2d(32, num_classes, 1)

        self._init_decoder()

    def _init_decoder(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, feats = self.encoder(x, return_features=True)

        s1 = feats["block1"]
        s2 = feats["block2"]
        s3 = feats["block3"]
        s4 = feats["block4"]
        s5 = feats["block5"]

        d = self.up5(s5);  d = self.dec5(torch.cat([d, s4], dim=1))
        d = self.up4(d);   d = self.dec4(torch.cat([d, s3], dim=1))
        d = self.up3(d);   d = self.dec3(torch.cat([d, s2], dim=1))
        d = self.up2(d);   d = self.dec2(torch.cat([d, s1], dim=1))
        d = self.up1(d);   d = self.dec1(d)

        return self.final(d)