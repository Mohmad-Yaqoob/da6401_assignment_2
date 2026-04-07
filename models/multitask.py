import os

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .classification import ClassificationHead
from .localization import RegressionHead
from .segmentation import DecoderBlock
from .layers import CustomDropout

# checkpoints live in the checkpoints/ subfolder by default
_CKPT_DIR = "checkpoints"


def _load_state(path: str, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def _sub(sd: dict, prefix: str) -> dict:
    # pull out keys starting with prefix and strip the prefix
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(
        self,
        num_breeds:       int = 37,
        seg_classes:      int = 3,
        in_channels:      int = 3,
        classifier_path:  str = os.path.join(_CKPT_DIR, "classifier.pth"),
        localizer_path:   str = os.path.join(_CKPT_DIR, "localizer.pth"),
        unet_path:        str = os.path.join(_CKPT_DIR, "unet.pth"),
    ):
        import gdown
        gdown.download(id="CLASSIFIER_DRIVE_ID", output=classifier_path, quiet=False)
        gdown.download(id="LOCALIZER_DRIVE_ID",  output=localizer_path,  quiet=False)
        gdown.download(id="UNET_DRIVE_ID",        output=unet_path,       quiet=False)

        super().__init__()

        # three encoders — one per task
        # keeping them separate avoids the feature mismatch that happens when
        # a single shared encoder is loaded from one checkpoint but used with
        # heads trained against a different encoder's output distribution
        self.encoder_cls = VGG11Encoder(in_channels=in_channels)
        self.encoder_loc = VGG11Encoder(in_channels=in_channels)
        self.encoder_seg = VGG11Encoder(in_channels=in_channels)

        self.cls_head = ClassificationHead(num_classes=num_breeds, dropout_p=0.5)
        self.loc_head = RegressionHead(dropout_p=0.5)

        self.dec5 = DecoderBlock(512, 512, 512)
        self.dec4 = DecoderBlock(512, 512, 256)
        self.dec3 = DecoderBlock(256, 256, 128)
        self.dec2 = DecoderBlock(128, 128,  64)
        self.dec1 = DecoderBlock( 64,  64,  32)
        self.seg_dropout = CustomDropout(p=0.5)
        self.seg_final   = nn.Conv2d(32, seg_classes, kernel_size=1)

        self._load(classifier_path, localizer_path, unet_path)

    def _load(self, clf_path: str, loc_path: str, seg_path: str) -> None:
        dev = torch.device("cpu")

        if os.path.isfile(clf_path):
            sd = _load_state(clf_path, dev)
            self.encoder_cls.load_state_dict(_sub(sd, "encoder."), strict=False)
            self.cls_head.load_state_dict(_sub(sd, "head."), strict=False)
            print(f"[MultiTask] classifier loaded from '{clf_path}'")
        else:
            print(f"[MultiTask] WARNING: '{clf_path}' not found")

        if os.path.isfile(loc_path):
            sd = _load_state(loc_path, dev)
            self.encoder_loc.load_state_dict(_sub(sd, "encoder."), strict=False)
            self.loc_head.load_state_dict(_sub(sd, "head."), strict=False)
            print(f"[MultiTask] localizer loaded from '{loc_path}'")
        else:
            print(f"[MultiTask] WARNING: '{loc_path}' not found")

        if os.path.isfile(seg_path):
            sd = _load_state(seg_path, dev)
            self.encoder_seg.load_state_dict(_sub(sd, "encoder."), strict=False)
            for name in ["dec5", "dec4", "dec3", "dec2", "dec1"]:
                getattr(self, name).load_state_dict(_sub(sd, f"{name}."), strict=False)
            fc_sd = _sub(sd, "final_conv.")
            if fc_sd:
                self.seg_final.load_state_dict(fc_sd, strict=False)
            print(f"[MultiTask] unet loaded from '{seg_path}'")
        else:
            print(f"[MultiTask] WARNING: '{seg_path}' not found")

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass for multi-task model.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization':   [B, 4] bounding box tensor.
            - 'segmentation':   [B, seg_classes, H, W] segmentation logits tensor
        """
        # each encoder runs independently — no weight sharing
        cls_out = self.cls_head(self.encoder_cls(x, return_features=False))
        loc_out = self.loc_head(self.encoder_loc(x, return_features=False))

        bn, feats = self.encoder_seg(x, return_features=True)
        d = self.dec5(bn,         feats["b5"])
        d = self.dec4(d,          feats["b4"])
        d = self.dec3(d,          feats["b3"])
        d = self.dec2(d,          feats["b2"])
        d = self.dec1(d,          feats["b1"])
        seg_out = self.seg_final(self.seg_dropout(d))

        return {
            "classification": cls_out,
            "localization":   loc_out,
            "segmentation":   seg_out,
        }