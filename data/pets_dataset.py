import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2


IMG_SIZE = 224
MEAN     = (0.485, 0.456, 0.406)
STD      = (0.229, 0.224, 0.225)


def _train_tfm() -> A.Compose:
    # heavy augmentation — scale, colour, noise, coarse dropout
    return A.Compose([
        A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.5, 1.0),
                            ratio=(0.75, 1.33), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.1, p=0.8),
        A.CLAHE(clip_limit=4.0, p=0.3),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
        A.GaussianBlur(blur_limit=3, p=0.15),
        A.CoarseDropout(num_holes_range=(6, 12), hole_height_range=(16, 32),
                        hole_width_range=(16, 32), fill=0, p=0.4),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="pascal_voc",
                                label_fields=["bbox_labels"],
                                min_visibility=0.2))


def _val_tfm() -> A.Compose:
    # plain resize — no padding artefacts
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="pascal_voc",
                                label_fields=["bbox_labels"],
                                min_visibility=0.1))


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet loader for classification, localisation and segmentation.

    Mode controls which annotations are required:
      'cls' — image + label only
      'loc' — image + label + XML bbox
      'seg' — image + label + trimap mask
      'all' — image + label + bbox + mask

    Bbox output: [x1, y1, x2, y2] in pixel space (pascal_voc format).
    Mask output: {0=foreground, 1=background, 2=boundary} as long tensor.
    """

    def __init__(
        self,
        root:       str,
        split:      str   = "train",
        mode:       str   = "cls",
        transform         = None,
        test_size:  float = 0.10,
        val_size:   float = 0.10,
    ):
        self.root     = Path(root)
        self.split    = split
        self.mode     = mode
        self.img_dir  = self.root / "images"
        self.ann_dir  = self.root / "annotations"
        self.trim_dir = self.ann_dir / "trimaps"
        self.xml_dir  = self.ann_dir / "xmls"

        self.transform = (transform if transform is not None
                          else (_train_tfm() if split == "train" else _val_tfm()))
        self.samples   = self._build(test_size, val_size)

    def _parse_list(self):
        records = []
        with open(self.ann_dir / "list.txt") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                p = line.strip().split()
                records.append({"name": p[0], "label": int(p[1]) - 1})
        return records

    def _build(self, test_size, val_size):
        records = self._parse_list()
        names   = [r["name"]  for r in records]
        labels  = [r["label"] for r in records]

        tr_n, te_n, tr_l, _ = train_test_split(
            names, labels, test_size=test_size, stratify=labels, random_state=7)
        tr_n, va_n, _, _ = train_test_split(
            tr_n, tr_l, test_size=val_size, stratify=tr_l, random_state=7)

        chosen = {"train": set(tr_n), "val": set(va_n), "test": set(te_n)}[self.split]

        out = []
        for r in records:
            name = r["name"]
            if name not in chosen:
                continue
            if not (self.img_dir / f"{name}.jpg").exists():
                continue

            has_mask = (self.trim_dir / f"{name}.png").exists()
            has_xml  = (self.xml_dir  / f"{name}.xml").exists()

            if self.mode == "seg" and not has_mask:   continue
            if self.mode == "loc" and not has_xml:    continue
            if self.mode == "all" and not (has_mask and has_xml): continue

            out.append({"name": name, "label": r["label"],
                        "has_mask": has_mask, "has_xml": has_xml})

        print(f"[Dataset] {self.split}/{self.mode}: {len(out)} samples")
        return out

    def _mask(self, name):
        p = self.trim_dir / f"{name}.png"
        if not p.exists():
            return None
        raw = np.array(Image.open(p), dtype=np.int32)
        return (raw - 1).clip(0, 2).astype(np.uint8)

    def _bbox(self, name):
        p = self.xml_dir / f"{name}.xml"
        if not p.exists():
            return None
        root = ET.parse(p).getroot()
        obj  = root.find("object")
        if obj is None:
            return None
        b = obj.find("bndbox")
        return [float(b.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax")]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s    = self.samples[idx]
        name = s["name"]

        img  = np.array(Image.open(self.img_dir / f"{name}.jpg").convert("RGB"))
        mask = self._mask(name)
        bbox = self._bbox(name)

        if mask is None:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)

        bboxes      = [bbox] if bbox is not None else []
        bbox_labels = [0]    if bbox is not None else []

        t     = self.transform(image=img, mask=mask,
                               bboxes=bboxes, bbox_labels=bbox_labels)
        image = t["image"]
        mask  = t["mask"].long()

        if t["bboxes"]:
            x1, y1, x2, y2 = t["bboxes"][0]
            bbox_t    = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
            bbox_flag = torch.tensor(1.0)
        else:
            bbox_t    = torch.zeros(4)
            bbox_flag = torch.tensor(0.0)

        return {
            "image":     image,
            "label":     torch.tensor(s["label"]),
            "mask":      mask if s["has_mask"] else None,
            "bbox":      bbox_t,
            "bbox_mask": bbox_flag,
            "name":      name,
        }


def collate_fn(batch):
    images     = torch.stack([b["image"] for b in batch])
    labels     = torch.stack([b["label"] for b in batch])
    H, W       = images.shape[2:]
    masks      = torch.stack([
        b["mask"] if b["mask"] is not None
        else torch.full((H, W), -1, dtype=torch.long)
        for b in batch
    ])
    bboxes     = torch.stack([b["bbox"]      for b in batch])
    bbox_masks = torch.stack([b["bbox_mask"] for b in batch])
    return {"image": images, "label": labels, "mask": masks,
            "bbox": bboxes, "bbox_mask": bbox_masks}