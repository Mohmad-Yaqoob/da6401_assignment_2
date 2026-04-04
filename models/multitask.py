import os
import torch
import torch.nn as nn
from models.vgg11 import VGG11
from models.layers import CustomDropout


def _dec_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class MultiTaskPerceptionModel(nn.Module):
    """
    Unified multi-task model — single forward pass yields:
        1. Classification logits  (B, 37)
        2. Bounding box coords    (B, 4)  pixel space [cx, cy, w, h]
        3. Segmentation mask      (B, 3, H, W) raw logits

    On init, downloads checkpoints from Google Drive and loads weights
    for the shared backbone and all three task heads.

    Args:
        classifier_path : local path for classifier checkpoint
        localizer_path  : local path for localizer checkpoint
        unet_path       : local path for unet checkpoint
        num_classes     : number of breed classes (37)
        dropout_p       : dropout probability in classification head
        seg_classes     : segmentation output classes (3 for trimaps)
    """

    def __init__(
        self,
        classifier_path: str = "checkpoints/classifier.pth",
        localizer_path:  str = "checkpoints/localizer.pth",
        unet_path:       str = "checkpoints/unet.pth",
        num_classes:     int = 37,
        dropout_p:       float = 0.5,
        seg_classes:     int = 3,
    ):
        super().__init__()

        # ── download checkpoints from Google Drive ──────────────────────────
        import gdown
        gdown.download(id="CLASSIFIER_DRIVE_ID", output=classifier_path, quiet=False)
        gdown.download(id="LOCALIZER_DRIVE_ID",  output=localizer_path,  quiet=False)
        gdown.download(id="UNET_DRIVE_ID",       output=unet_path,       quiet=False)

        # ── shared VGG11 backbone ───────────────────────────────────────────
        backbone = VGG11(num_classes=num_classes, dropout_p=dropout_p)

        self.enc1    = backbone.block1
        self.enc2    = backbone.block2
        self.enc3    = backbone.block3
        self.enc4    = backbone.block4
        self.enc5    = backbone.block5
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

        # ── task 2: localization head ───────────────────────────────────────
        self.loc_head = nn.Sequential(
            nn.Linear(flat_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.ReLU(inplace=True),   # pixel coords >= 0
        )

        # ── task 3: segmentation decoder ───────────────────────────────────
        self.up5  = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec5 = _dec_block(512 + 512, 512)
        self.up4  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = _dec_block(256 + 256, 256)
        self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = _dec_block(128 + 128, 128)
        self.up2  = nn.ConvTranspose2d(128, 64,  2, stride=2)
        self.dec2 = _dec_block(64  + 64,  64)
        self.up1  = nn.ConvTranspose2d(64,  32,  2, stride=2)
        self.dec1 = _dec_block(32, 32)
        self.seg_final = nn.Conv2d(32, seg_classes, 1)

        # ── load trained weights ────────────────────────────────────────────
        self._load_weights(classifier_path, localizer_path, unet_path)

    def _load_weights(self, cls_path, loc_path, seg_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load classifier — backbone + cls_head
        if os.path.exists(cls_path):
            cls_state = torch.load(cls_path, map_location=device)
            state = cls_state.get("model_state", cls_state)
            # remap block -> enc keys
            remapped = {}
            for k, v in state.items():
                new_k = k.replace("block1","enc1").replace("block2","enc2") \
                         .replace("block3","enc3").replace("block4","enc4") \
                         .replace("block5","enc5")
                remapped[new_k] = v
            missing, unexpected = self.load_state_dict(remapped, strict=False)
            print(f"Classifier loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

        # load localizer — backbone + loc_head
        if os.path.exists(loc_path):
            loc_state = torch.load(loc_path, map_location=device)
            state = loc_state.get("model_state", loc_state)
            remapped = {}
            for k, v in state.items():
                new_k = k.replace("block1","enc1").replace("block2","enc2") \
                         .replace("block3","enc3").replace("block4","enc4") \
                         .replace("block5","enc5") \
                         .replace("reg_head","loc_head")
                remapped[new_k] = v
            missing, unexpected = self.load_state_dict(remapped, strict=False)
            print(f"Localizer loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

        # load unet — backbone + seg decoder
        if os.path.exists(seg_path):
            seg_state = torch.load(seg_path, map_location=device)
            state = seg_state.get("model_state", seg_state)
            remapped = {}
            for k, v in state.items():
                new_k = k.replace("enc1","enc1").replace("enc2","enc2") \
                         .replace("enc3","enc3").replace("enc4","enc4") \
                         .replace("enc5","enc5")
                remapped[new_k] = v
            missing, unexpected = self.load_state_dict(remapped, strict=False)
            print(f"UNet loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    def forward(self, x: torch.Tensor):
        """
        Single forward pass — all three tasks simultaneously.

        Args:
            x : (B, 3, H, W) normalized input image

        Returns:
            cls_logits : (B, 37)
            bbox       : (B, 4)  pixel space [cx, cy, w, h]
            seg_logits : (B, 3, H, W)
        """
        # shared encoder
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        s5 = self.enc5(s4)

        # classification + localization from pooled features
        pooled     = self.avgpool(s5)
        flat       = torch.flatten(pooled, 1)
        cls_logits = self.cls_head(flat)
        bbox       = self.loc_head(flat)

        # segmentation decoder with skip connections
        d = self.up5(s5);  d = self.dec5(torch.cat([d, s4], dim=1))
        d = self.up4(d);   d = self.dec4(torch.cat([d, s3], dim=1))
        d = self.up3(d);   d = self.dec3(torch.cat([d, s2], dim=1))
        d = self.up2(d);   d = self.dec2(torch.cat([d, s1], dim=1))
        d = self.up1(d);   d = self.dec1(d)
        seg_logits = self.seg_final(d)

        return cls_logits, bbox, seg_logits


if __name__ == "__main__":
    # shape check without loading checkpoints
    import torch

    class _TestModel(MultiTaskPerceptionModel):
        def __init__(self):
            # skip gdown and weight loading for shape test
            nn.Module.__init__(self)
            backbone = VGG11(num_classes=37, dropout_p=0.5)
            self.enc1 = backbone.block1; self.enc2 = backbone.block2
            self.enc3 = backbone.block3; self.enc4 = backbone.block4
            self.enc5 = backbone.block5
            self.avgpool = nn.AdaptiveAvgPool2d((7,7))
            flat_dim = 512*7*7
            self.cls_head = nn.Sequential(nn.Linear(flat_dim,4096),nn.ReLU(True),
                                          nn.Linear(4096,4096),nn.ReLU(True),nn.Linear(4096,37))
            self.loc_head = nn.Sequential(nn.Linear(flat_dim,1024),nn.ReLU(True),
                                          nn.Linear(1024,256),nn.ReLU(True),nn.Linear(256,4),nn.ReLU(True))
            self.up5  = nn.ConvTranspose2d(512,512,2,stride=2); self.dec5 = _dec_block(1024,512)
            self.up4  = nn.ConvTranspose2d(512,256,2,stride=2); self.dec4 = _dec_block(512,256)
            self.up3  = nn.ConvTranspose2d(256,128,2,stride=2); self.dec3 = _dec_block(256,128)
            self.up2  = nn.ConvTranspose2d(128,64,2,stride=2);  self.dec2 = _dec_block(128,64)
            self.up1  = nn.ConvTranspose2d(64,32,2,stride=2);   self.dec1 = _dec_block(32,32)
            self.seg_final = nn.Conv2d(32,3,1)

    m = _TestModel()
    m.eval()
    x = torch.randn(2, 3, 224, 224)
    cls, box, seg = m.forward(x)
    print("cls :", cls.shape)   # (2, 37)
    print("bbox:", box.shape)   # (2, 4)
    print("seg :", seg.shape)   # (2, 3, 224, 224)
    assert cls.shape == (2,37) and box.shape == (2,4) and seg.shape == (2,3,224,224)
    print("MultiTaskPerceptionModel shape check passed.")