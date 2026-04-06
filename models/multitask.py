import os
import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout


def _dec_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class MultiTaskPerceptionModel(nn.Module):
    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "classifier.pth",
        localizer_path: str = "localizer.pth",
        unet_path: str = "unet.pth",
    ):
        super().__init__()

        # ✅ Download weights
        import gdown
        gdown.download(id="1n85YO1B-IRm9GFq2L--RfjsMsOuRo9I8", output=classifier_path, quiet=False)
        gdown.download(id="1rMfun1L2hs2nq1QKF8Pd7p-HeBiGVdKk", output=localizer_path, quiet=False)
        gdown.download(id="1-bt2g5vfxKMEN5hsbmTf4WERiSHhtXfu", output=unet_path, quiet=False)

        # ✅ Encoder
        self.encoder = VGG11Encoder(in_channels=in_channels)
        flat = 512 * 7 * 7

        # ✅ Classification Head
        self.cls_head = nn.Sequential(
            nn.Linear(flat, 4096), nn.ReLU(True),
            CustomDropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(True),
            CustomDropout(0.5),
            nn.Linear(4096, num_breeds),
        )

        # ✅ Localization Head (FIXED: removed ReLU at end)
        self.loc_head = nn.Sequential(
            nn.Linear(flat, 1024), nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 256), nn.ReLU(True),
            nn.Linear(256, 4)   # ❌ NO ReLU here
        )

        # ✅ Segmentation Decoder
        self.up5  = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec5 = _dec_block(1024, 512)
        self.up4  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = _dec_block(512, 256)
        self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = _dec_block(256, 128)
        self.up2  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = _dec_block(128, 64)
        self.up1  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = _dec_block(32, 32)

        # self.seg_final = nn.Conv2d(32, seg_classes, 1)
        self.final = nn.Conv2d(32, seg_classes, 1)

        # ✅ Load weights
        self._load_weights(classifier_path, localizer_path, unet_path)

    # ============================================================
    # ✅ FIXED WEIGHT LOADING
    # ============================================================
    def _load_weights(self, cls_path, loc_path, seg_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def load_state(path):
            ckpt = torch.load(path, map_location=device)
            if isinstance(ckpt, dict):
                return ckpt.get("state_dict", ckpt.get("model_state", ckpt))
            return ckpt

        merged = {}

        # ✅ Segmentation weights (TRY SAFE MAPPING)
        if os.path.exists(seg_path):
            seg_state = load_state(seg_path)
            for k, v in seg_state.items():
                merged[k] = v   # direct — no renaming needed

        # ✅ Localization weights
        if os.path.exists(loc_path):
            loc_state = load_state(loc_path)
            for k, v in loc_state.items():
                new_k = k.replace("head.", "loc_head.")
                merged[new_k] = v

        # ✅ Classification weights (overwrite encoder last)
        if os.path.exists(cls_path):
            cls_state = load_state(cls_path)
            for k, v in cls_state.items():
                new_k = k.replace("head.", "cls_head.")
                merged[new_k] = v

        miss, unexp = self.load_state_dict(merged, strict=False)
        print(f"loaded weights — missing: {len(miss)}, unexpected: {len(unexp)}")
        if miss:
            print("missing sample:", miss[:5])

    # ============================================================
    # ✅ FORWARD (FIXED BBOX FORMAT)
    # ============================================================
    def forward(self, x: torch.Tensor):
        _, feats = self.encoder(x, return_features=True)

        s1, s2, s3 = feats["block1"], feats["block2"], feats["block3"]
        s4, s5 = feats["block4"], feats["block5"]

        pooled = nn.functional.adaptive_avg_pool2d(s5, (7, 7))
        flat = torch.flatten(pooled, 1)

        # Classification
        cls = self.cls_head(flat)

        # Localization
        # loc_raw = self.loc_head(flat)
        # cx = torch.sigmoid(loc_raw[:, 0]) * 224
        # cy = torch.sigmoid(loc_raw[:, 1]) * 224
        # w  = torch.sigmoid(loc_raw[:, 2]) * 224
        # h  = torch.sigmoid(loc_raw[:, 3]) * 224
        # loc = torch.stack([cx, cy, w, h], dim=1)
        # loc = self.loc_head(flat)

        loc_norm = torch.sigmoid(self.loc_head(flat))  # normalised 0-1
        loc = loc_norm * 224  # convert to pixel space

        # Segmentation
        d = self.up5(s5);  d = self.dec5(torch.cat([d, s4], 1))
        d = self.up4(d);   d = self.dec4(torch.cat([d, s3], 1))
        d = self.up3(d);   d = self.dec3(torch.cat([d, s2], 1))
        d = self.up2(d);   d = self.dec2(torch.cat([d, s1], 1))
        d = self.up1(d);   d = self.dec1(d)

        # seg = self.seg_final(d)
        seg = self.final(d)

        return {
            "classification": cls,
            "localization": loc,
            "segmentation": seg,
        }