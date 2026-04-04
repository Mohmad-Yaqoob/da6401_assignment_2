import os
import tarfile
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGES_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
ANNOTS_URL  = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"


def _download(url, dest):
    os.makedirs(dest, exist_ok=True)
    fname = url.split("/")[-1]
    fpath = os.path.join(dest, fname)
    if not os.path.exists(fpath):
        print(f"downloading {fname} ...")
        urllib.request.urlretrieve(url, fpath)
    marker = fpath.replace(".tar.gz", "")
    if not os.path.exists(marker):
        print(f"extracting {fname} ...")
        with tarfile.open(fpath, "r:gz") as t:
            t.extractall(dest)


def prepare_dataset(root="./pets_data"):
    _download(IMAGES_URL, root)
    _download(ANNOTS_URL, root)
    return root


def _parse_list(ann_dir):
    entries = []
    with open(os.path.join(ann_dir, "list.txt")) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            entries.append({"name": parts[0], "cls": int(parts[1]) - 1})
    return entries


def _parse_bboxes(ann_dir):
    xml_dir  = os.path.join(ann_dir, "xmls")
    bbox_map = {}
    if not os.path.exists(xml_dir):
        return bbox_map
    for fn in os.listdir(xml_dir):
        if not fn.endswith(".xml"):
            continue
        name = fn[:-4]
        root = ET.parse(os.path.join(xml_dir, fn)).getroot()
        obj  = root.find("object")
        if obj is None:
            continue
        b = obj.find("bndbox")
        bbox_map[name] = [
            float(b.find("xmin").text), float(b.find("ymin").text),
            float(b.find("xmax").text), float(b.find("ymax").text),
        ]
    return bbox_map


def _train_aug(s=224):
    return A.Compose([
        A.Resize(s, s),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.4),
        A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.4),
        A.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="pascal_voc",
                                label_fields=["bbox_labels"],
                                min_visibility=0.3))


def _val_aug(s=224):
    return A.Compose([
        A.Resize(s, s),
        A.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="pascal_voc",
                                label_fields=["bbox_labels"],
                                min_visibility=0.3))


class OxfordIIITPetDataset(Dataset):
    # loads images, class labels, bboxes, and segmentation trimaps together
    # split is derived from the official trainval.txt / test.txt files
    # val is carved out of trainval using a fixed seed split

    def __init__(self, root="./pets_data", split="train",
                 img_size=224, val_fraction=0.15, seed=42):
        super().__init__()
        assert split in ("train", "val", "test")
        self.img_size = img_size

        img_dir = os.path.join(root, "images")
        ann_dir = os.path.join(root, "annotations")

        all_entries = _parse_list(ann_dir)
        self.bbox_map = _parse_bboxes(ann_dir)

        tv_names, te_names = set(), set()
        for fname, s in [("trainval.txt", tv_names), ("test.txt", te_names)]:
            with open(os.path.join(ann_dir, fname)) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    s.add(line.split()[0])

        tv = [e for e in all_entries if e["name"] in tv_names]
        te = [e for e in all_entries if e["name"] in te_names]

        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(tv)).tolist()
        cut = int(len(tv) * val_fraction)

        if split == "train":
            self.entries = [tv[i] for i in idx[cut:]]
            self.tfm     = _train_aug(img_size)
        elif split == "val":
            self.entries = [tv[i] for i in idx[:cut]]
            self.tfm     = _val_aug(img_size)
        else:
            self.entries = te
            self.tfm     = _val_aug(img_size)

        self.img_dir = img_dir
        self.ann_dir = ann_dir
        print(f"[OxfordPets] {split}: {len(self.entries)} samples")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e    = self.entries[idx]
        name = e["name"]
        cls  = e["cls"]

        img = np.array(Image.open(
            os.path.join(self.img_dir, name + ".jpg")).convert("RGB"))
        h, w = img.shape[:2]

        has_box = name in self.bbox_map
        if has_box:
            x1, y1, x2, y2 = self.bbox_map[name]
            x1 = max(0., x1); y1 = max(0., y1)
            x2 = min(float(w), x2); y2 = min(float(h), y2)
            bboxes = [[x1, y1, x2, y2]]; blabels = [0]
        else:
            bboxes = []; blabels = []

        t     = self.tfm(image=img, bboxes=bboxes, bbox_labels=blabels)
        image = t["image"]

        if has_box and len(t["bboxes"]) > 0:
            bx1, by1, bx2, by2 = t["bboxes"][0]
            S = self.img_size
            bbox = torch.tensor([
                (bx1+bx2)/2/S, (by1+by2)/2/S,
                (bx2-bx1)/S,   (by2-by1)/S,
            ], dtype=torch.float32)
        else:
            bbox = torch.tensor([0.5, 0.5, 1.0, 1.0], dtype=torch.float32)

        # load mask — trimaps: 1=fg, 2=bg, 3=border -> remap to 0,1,2
        mask_path = os.path.join(self.ann_dir, "trimaps", name + ".png")
        if os.path.exists(mask_path):
            m = np.array(Image.open(mask_path)).astype(np.int64) - 1
            m = np.clip(m, 0, 2).astype(np.uint8)
        else:
            m = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        m = np.array(Image.fromarray(m).resize(
            (self.img_size, self.img_size), Image.NEAREST))
        mask = torch.tensor(m.copy(), dtype=torch.long)

        return {
            "image": image,
            "label": torch.tensor(cls, dtype=torch.long),
            "bbox":  bbox,
            "mask":  mask,
            "name":  name,
        }


def get_dataloaders(root="./pets_data", img_size=224, batch_size=32,
                    num_workers=2, val_fraction=0.15, seed=42):
    tr = OxfordIIITPetDataset(root, "train", img_size, val_fraction, seed)
    va = OxfordIIITPetDataset(root, "val",   img_size, val_fraction, seed)
    te = OxfordIIITPetDataset(root, "test",  img_size, val_fraction, seed)
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=True, drop_last=True),
        DataLoader(va, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=True),
        DataLoader(te, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=True),
    )