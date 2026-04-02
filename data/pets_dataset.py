import os
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─── download URLs ────────────────────────────────────────────────────────────
IMAGES_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
ANNOTS_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"


def download_and_extract(url: str, dest_dir: str):
    """Pull a tar.gz from the web and unpack it, skip if already done."""
    os.makedirs(dest_dir, exist_ok=True)
    filename = url.split("/")[-1]
    filepath = os.path.join(dest_dir, filename)

    if not os.path.exists(filepath):
        print(f"Downloading {filename} ...")
        urllib.request.urlretrieve(url, filepath)
        print("Done.")

    extracted_marker = filepath.replace(".tar.gz", "")
    if not os.path.exists(extracted_marker):
        print(f"Extracting {filename} ...")
        with tarfile.open(filepath, "r:gz") as tar:
            tar.extractall(dest_dir)
        print("Extraction complete.")


def prepare_dataset(root: str = "./pets_data"):
    """Download both archives if not already present."""
    download_and_extract(IMAGES_URL, root)
    download_and_extract(ANNOTS_URL, root)
    return root


# ─── breed name → class index mapping ────────────────────────────────────────
# The dataset encodes class in the filename: e.g. "Abyssinian_34.jpg" → class 1
# The list file gives us (image_name, class_id, species, breed_id)
def parse_list_file(annotations_dir: str):
    """
    Read the trainval.txt / test.txt list files.
    Returns a list of dicts with image_name and class_id (1-indexed → we convert to 0-indexed).
    """
    list_file = os.path.join(annotations_dir, "list.txt")
    entries = []
    with open(list_file, "r") as f:
        for line in f:
            line = line.strip()
            # skip comment lines
            if line.startswith("#") or len(line) == 0:
                continue
            parts = line.split()
            image_name = parts[0]       # e.g. Abyssinian_34
            class_id   = int(parts[1])  # 1-indexed
            entries.append({
                "image_name": image_name,
                "class_id":   class_id - 1,   # convert to 0-indexed
            })
    return entries


def parse_bbox_file(annotations_dir: str):
    """
    Read the XML-style bounding box file that ships with the dataset.
    Returns a dict mapping image_name → [x_min, y_min, x_max, y_max] in pixel coords.
    We later normalise these to [0,1] inside the dataset.
    """
    import xml.etree.ElementTree as ET
    bbox_dir = os.path.join(annotations_dir, "xmls")
    bbox_map = {}

    if not os.path.exists(bbox_dir):
        # no bbox annotations present — return empty dict and handle downstream
        return bbox_map

    for xml_file in os.listdir(bbox_dir):
        if not xml_file.endswith(".xml"):
            continue
        image_name = xml_file.replace(".xml", "")
        tree = ET.parse(os.path.join(bbox_dir, xml_file))
        root = tree.getroot()

        # some files have multiple objects; we grab the first one (head bbox)
        obj = root.find("object")
        if obj is None:
            continue
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        bbox_map[image_name] = [xmin, ymin, xmax, ymax]

    return bbox_map


# ─── albumentations transform sets ───────────────────────────────────────────
def get_train_transforms(img_size: int = 224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["bbox_labels"], min_visibility=0.3))


def get_val_transforms(img_size: int = 224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["bbox_labels"], min_visibility=0.3))


def get_seg_train_transforms(img_size: int = 224):
    """Separate pipeline for segmentation — mask must be transformed alongside image."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_seg_val_transforms(img_size: int = 224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ─── main dataset class ───────────────────────────────────────────────────────
class OxfordPetsDataset(Dataset):
    """
    Loads images, class labels, bounding boxes and segmentation trimaps
    for the Oxford-IIIT Pet dataset all in one place.

    Args:
        root        : folder where pets_data lives
        split       : "train", "val", or "test"
        img_size    : resize target for images and masks
        val_fraction: fraction of trainval to use as validation
        seed        : random seed for the split
    """

    def __init__(
        self,
        root: str = "./pets_data",
        split: str = "train",
        img_size: int = 224,
        val_fraction: float = 0.15,
        seed: int = 42,
    ):
        super().__init__()
        assert split in ("train", "val", "test"), "split must be train / val / test"

        self.split    = split
        self.img_size = img_size
        self.root     = root

        images_dir      = os.path.join(root, "images")
        annotations_dir = os.path.join(root, "annotations")

        # parse the full list and the bounding box map
        all_entries = parse_list_file(annotations_dir)
        self.bbox_map = parse_bbox_file(annotations_dir)

        # ── split: trainval → train + val; test stays separate ────────────
        # The dataset provides a "trainval.txt" and "test.txt" but the master
        # list.txt covers everything. We recreate the official split manually.
        trainval_file = os.path.join(annotations_dir, "trainval.txt")
        test_file     = os.path.join(annotations_dir, "test.txt")

        trainval_names = set()
        test_names     = set()

        with open(trainval_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                trainval_names.add(line.split()[0])

        with open(test_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                test_names.add(line.split()[0])

        trainval_entries = [e for e in all_entries if e["image_name"] in trainval_names]
        test_entries     = [e for e in all_entries if e["image_name"] in test_names]

        # deterministic shuffle then cut
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(trainval_entries)).tolist()
        cut = int(len(trainval_entries) * val_fraction)

        val_idx   = indices[:cut]
        train_idx = indices[cut:]

        if split == "train":
            self.entries = [trainval_entries[i] for i in train_idx]
            self.img_transform = get_train_transforms(img_size)
            self.seg_transform = get_seg_train_transforms(img_size)
        elif split == "val":
            self.entries = [trainval_entries[i] for i in val_idx]
            self.img_transform = get_val_transforms(img_size)
            self.seg_transform = get_seg_val_transforms(img_size)
        else:  # test
            self.entries = test_entries
            self.img_transform = get_val_transforms(img_size)
            self.seg_transform = get_seg_val_transforms(img_size)

        self.images_dir      = images_dir
        self.annotations_dir = annotations_dir

        print(f"[OxfordPets] {split}: {len(self.entries)} samples loaded.")

    def __len__(self):
        return len(self.entries)

    def _load_image(self, name: str) -> np.ndarray:
        path = os.path.join(self.images_dir, name + ".jpg")
        img  = Image.open(path).convert("RGB")
        return np.array(img)

    def _load_mask(self, name: str) -> np.ndarray:
        """
        Trimaps: 1=foreground, 2=background, 3=not classified (border).
        We remap to 0/1/2 for cross-entropy (3 classes).
        """
        mask_path = os.path.join(self.annotations_dir, "trimaps", name + ".png")
        if not os.path.exists(mask_path):
            # return a blank mask if the file is somehow missing
            return np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        mask = np.array(Image.open(mask_path))
        # values are 1,2,3 → remap to 0,1,2
        mask = mask.astype(np.int64) - 1
        mask = np.clip(mask, 0, 2).astype(np.uint8)
        return mask

    def _get_bbox_normalised(self, name: str, img_w: int, img_h: int):
        """
        Return [x_center, y_center, width, height] normalised to [0,1].
        If no bbox annotation exists we return a dummy full-image box.
        """
        if name not in self.bbox_map:
            return torch.tensor([0.5, 0.5, 1.0, 1.0], dtype=torch.float32)

        xmin, ymin, xmax, ymax = self.bbox_map[name]

        # clip to image boundary (some annotations slightly exceed it)
        xmin = max(0.0, xmin)
        ymin = max(0.0, ymin)
        xmax = min(float(img_w), xmax)
        ymax = min(float(img_h), ymax)

        x_center = ((xmin + xmax) / 2.0) / img_w
        y_center = ((ymin + ymax) / 2.0) / img_h
        width    = (xmax - xmin) / img_w
        height   = (ymax - ymin) / img_h

        return torch.tensor([x_center, y_center, width, height], dtype=torch.float32)

    def __getitem__(self, idx: int):
        entry      = self.entries[idx]
        name       = entry["image_name"]
        class_id   = entry["class_id"]

        img_np = self._load_image(name)
        h, w   = img_np.shape[:2]

        # ── bounding box (for albumentations we keep pascal_voc format) ────
        has_bbox = name in self.bbox_map
        if has_bbox:
            xmin, ymin, xmax, ymax = self.bbox_map[name]
            xmin = max(0.0, xmin); ymin = max(0.0, ymin)
            xmax = min(float(w),  xmax); ymax = min(float(h), ymax)
            bboxes       = [[xmin, ymin, xmax, ymax]]
            bbox_labels  = [0]
        else:
            bboxes      = []
            bbox_labels = []

        # ── apply image + bbox transforms ─────────────────────────────────
        transformed = self.img_transform(
            image=img_np,
            bboxes=bboxes,
            bbox_labels=bbox_labels,
        )
        image_tensor = transformed["image"]  # (3, H, W) float

        # rebuild normalised bbox from transformed coords
        if has_bbox and len(transformed["bboxes"]) > 0:
            tx1, ty1, tx2, ty2 = transformed["bboxes"][0]
            bbox_tensor = torch.tensor([
                (tx1 + tx2) / 2.0 / self.img_size,
                (ty1 + ty2) / 2.0 / self.img_size,
                (tx2 - tx1) / self.img_size,
                (ty2 - ty1) / self.img_size,
            ], dtype=torch.float32)
        else:
            bbox_tensor = torch.tensor([0.5, 0.5, 1.0, 1.0], dtype=torch.float32)

        # ── segmentation mask ──────────────────────────────────────────────
        mask_np = self._load_mask(name)
        seg_out = self.seg_transform(image=img_np, mask=mask_np)
        # note: we use the original img_np here so mask aligns with
        # the same geometric augmentation seed. For simplicity we keep
        # both transforms independent — if you want joint aug, pass mask
        # into the img_transform pipeline too (albumentations supports it).
        # mask_tensor = torch.from_numpy(seg_out["mask"]).long()  # (H, W)
        # fixed
        raw_mask = seg_out["mask"]
        mask_tensor = raw_mask.long() if isinstance(raw_mask, torch.Tensor) else torch.from_numpy(raw_mask).long()

        return {
            "image":    image_tensor,              # (3, 224, 224) float
            "label":    torch.tensor(class_id, dtype=torch.long),  # scalar
            "bbox":     bbox_tensor,               # (4,) float [cx,cy,w,h] normalised
            "mask":     mask_tensor,               # (224, 224) long {0,1,2}
            "name":     name,
        }


# ─── convenience loaders ──────────────────────────────────────────────────────
def get_dataloaders(
    root: str = "./pets_data",
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
    val_fraction: float = 0.15,
    seed: int = 42,
):
    """
    Returns train_loader, val_loader, test_loader ready for training.
    Call prepare_dataset(root) before this if data isn't downloaded yet.
    """
    train_ds = OxfordPetsDataset(root, "train", img_size, val_fraction, seed)
    val_ds   = OxfordPetsDataset(root, "val",   img_size, val_fraction, seed)
    test_ds  = OxfordPetsDataset(root, "test",  img_size, val_fraction, seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ─── quick sanity check ───────────────────────────────────────────────────────
if __name__ == "__main__":
    root = prepare_dataset("./pets_data")
    train_loader, val_loader, test_loader = get_dataloaders(root=root, batch_size=8)

    batch = next(iter(train_loader))
    print("image shape  :", batch["image"].shape)    # (8, 3, 224, 224)
    print("label shape  :", batch["label"].shape)    # (8,)
    print("bbox shape   :", batch["bbox"].shape)     # (8, 4)
    print("mask shape   :", batch["mask"].shape)     # (8, 224, 224)
    print("label sample :", batch["label"])
    print("bbox sample  :", batch["bbox"][0])