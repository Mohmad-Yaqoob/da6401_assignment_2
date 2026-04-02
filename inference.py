"""
inference.py — run the full multi-task pipeline on a single image.

Usage:
    python inference.py --image path/to/pet.jpg --ckpt checkpoints/best_multitask.pth
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.multitask import MultiTaskModel

BREEDS = [
    'Abyssinian','Bengal','Birman','Bombay','British Shorthair',
    'Egyptian Mau','Maine Coon','Persian','Ragdoll','Russian Blue',
    'Siamese','Sphynx','American Bulldog','American Pit Bull Terrier',
    'Basset Hound','Beagle','Boxer','Chihuahua','English Cocker Spaniel',
    'English Setter','German Shorthaired','Great Pyrenees','Havanese',
    'Japanese Chin','Keeshond','Leonberger','Miniature Pinscher',
    'Newfoundland','Pomeranian','Pug','Saint Bernard','Samoyed',
    'Scottish Terrier','Shiba Inu','Staffordshire Bull Terrier',
    'Wheaten Terrier','Yorkshire Terrier'
]

SEG_COLORS = {0:(0.9,0.2,0.2), 1:(0.2,0.6,0.9), 2:(0.9,0.8,0.2)}
SEG_LABELS  = {0:'Foreground', 1:'Background', 2:'Border'}


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    rgb  = np.zeros((h, w, 3))
    for c, col in SEG_COLORS.items():
        rgb[mask == c] = col
    return rgb


def run_inference(image_path: str, ckpt_path: str, out_path: str = "output.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = MultiTaskModel(num_classes=37, dropout_p=0.5, seg_classes=3).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    # preprocess
    img_pil = Image.open(image_path).convert("RGB")
    img_np  = np.array(img_pil)

    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    inp = transform(image=img_np)["image"].unsqueeze(0).to(device)

    # forward
    with torch.no_grad():
        cls_logits, bbox, seg_logits = model(inp)

    breed_idx  = cls_logits.argmax(1).item()
    breed_name = BREEDS[breed_idx] if breed_idx < len(BREEDS) else f"Class {breed_idx}"
    confidence = torch.softmax(cls_logits, dim=1).max().item()

    box = bbox[0].cpu().numpy()          # [cx, cy, w, h] normalised
    S   = 224
    cx, cy, w, h = box
    x1, y1 = (cx - w/2)*S, (cy - h/2)*S
    x2, y2 = (cx + w/2)*S, (cy + h/2)*S

    seg_mask = seg_logits.argmax(1)[0].cpu().numpy()

    # denormalise image for display
    mean_t = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std_t  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    vis    = (inp[0].cpu() * std_t + mean_t).permute(1,2,0).clamp(0,1).numpy()

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].imshow(vis)
    axes[0].add_patch(patches.Rectangle(
        (x1, y1), x2-x1, y2-y1,
        linewidth=2, edgecolor="red", facecolor="none"
    ))
    axes[0].set_title(f"Detection\n{breed_name} ({confidence:.1%})", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(mask_to_rgb(seg_mask))
    axes[1].set_title("Segmentation mask", fontsize=10)
    axes[1].axis("off")

    # legend for seg
    for c, col in SEG_COLORS.items():
        axes[1].add_patch(patches.Patch(color=col, label=SEG_LABELS[c]))
    axes[1].legend(loc="lower right", fontsize=8)

    axes[2].imshow(vis)
    axes[2].imshow(mask_to_rgb(seg_mask), alpha=0.45)
    axes[2].set_title("Overlay", fontsize=10)
    axes[2].axis("off")

    plt.suptitle("DA6401 — Multi-Task Visual Perception Pipeline", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\nPredicted breed : {breed_name}")
    print(f"Confidence      : {confidence:.2%}")
    print(f"Bounding box    : cx={cx:.3f} cy={cy:.3f} w={w:.3f} h={h:.3f}")
    print(f"Output saved to : {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, required=True, help="path to input image")
    p.add_argument("--ckpt",  type=str, default="checkpoints/best_multitask.pth")
    p.add_argument("--out",   type=str, default="output.png")
    args = p.parse_args()
    run_inference(args.image, args.ckpt, args.out)