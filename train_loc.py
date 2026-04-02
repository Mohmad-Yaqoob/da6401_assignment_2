"""
Task 2 training script — bounding box regression using VGG11 encoder.

Loads the best classification checkpoint, plugs it into LocalizationModel,
then trains the regression head (with optional backbone fine-tuning).

Run example:
    python train_loc.py --epochs 20 --lr 5e-4 --freeze_backbone False
"""

import argparse
import os

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

from data.pets_dataset import prepare_dataset, get_dataloaders
from models.vgg11 import VGG11
from models.localization import LocalizationModel
from losses.iou_loss import IoULoss


# ── metric helpers ────────────────────────────────────────────────────────────

@torch.no_grad()
def mean_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """Compute mean IoU across a batch (for logging — not the loss)."""
    # convert cx/cy/w/h → x1/y1/x2/y2
    def to_corners(b):
        return (
            b[:, 0] - b[:, 2] / 2,
            b[:, 1] - b[:, 3] / 2,
            b[:, 0] + b[:, 2] / 2,
            b[:, 1] + b[:, 3] / 2,
        )

    px1, py1, px2, py2 = to_corners(pred)
    tx1, ty1, tx2, ty2 = to_corners(target)

    ix1 = torch.max(px1, tx1);  iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2);  iy2 = torch.min(py2, ty2)

    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    p_area = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    t_area = (tx2 - tx1).clamp(0) * (ty2 - ty1).clamp(0)
    union  = p_area + t_area - inter
    iou    = inter / (union + eps)
    return iou.mean().item()


# ── train / eval loops ────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_iou = 0.0, 0.0

    for batch in loader:
        images = batch["image"].to(device)
        bboxes = batch["bbox"].to(device)

        optimizer.zero_grad()
        pred = model(images)
        loss = criterion(pred, bboxes)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iou  += mean_iou(pred.detach(), bboxes)

    n = len(loader)
    return total_loss / n, total_iou / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_iou = 0.0, 0.0

    for batch in loader:
        images = batch["image"].to(device)
        bboxes = batch["bbox"].to(device)

        pred = model(images)
        loss = criterion(pred, bboxes)

        total_loss += loss.item()
        total_iou  += mean_iou(pred, bboxes)

    n = len(loader)
    return total_loss / n, total_iou / n


# ── main ──────────────────────────────────────────────────────────────────────

def train(args):
    wandb.init(
        project="da6401-assignment2",
        name=f"loc_freeze{args.freeze_backbone}_lr{args.lr}",
        config=vars(args),
        tags=["localization", "iou-loss"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root = prepare_dataset(args.data_dir)
    train_loader, val_loader, _ = get_dataloaders(
        root=root, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # load pretrained VGG11 backbone if checkpoint exists
    vgg = VGG11(num_classes=37, dropout_p=0.5)
    cls_ckpt = os.path.join(args.ckpt_dir, "best_cls.pth")
    if os.path.exists(cls_ckpt):
        state = torch.load(cls_ckpt, map_location="cpu")
        vgg.load_state_dict(state["model_state"])
        print(f"Loaded cls checkpoint from {cls_ckpt}")
    else:
        print("No cls checkpoint found — training localization from scratch")

    model = LocalizationModel(
        vgg,
        freeze_backbone=args.freeze_backbone,
        dropout_p=0.3,
    ).to(device)

    criterion = IoULoss(reduction="mean")

    # lower lr for backbone, higher for regression head
    backbone_params = list(model.block1.parameters()) + \
                      list(model.block2.parameters()) + \
                      list(model.block3.parameters()) + \
                      list(model.block4.parameters()) + \
                      list(model.block5.parameters())
    head_params = list(model.reg_head.parameters())

    optimizer = optim.Adam([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_iou = 0.0
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_iou = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        wandb.log({
            "epoch":       epoch,
            "train/loss":  train_loss,
            "train/iou":   train_iou,
            "val/loss":    val_loss,
            "val/iou":     val_iou,
            "lr":          scheduler.get_last_lr()[0],
        }, step=epoch)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_loss:.4f} iou {train_iou:.4f} | "
            f"val loss {val_loss:.4f} iou {val_iou:.4f}"
        )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            ckpt_path = os.path.join(args.ckpt_dir, "best_loc.pth")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_iou":     val_iou,
                "args":        vars(args),
            }, ckpt_path)
            print(f"  → saved best checkpoint (val_iou={val_iou:.4f})")

    print(f"\nDone. Best val IoU: {best_val_iou:.4f}")
    wandb.finish()


def parse_args():
    p = argparse.ArgumentParser(description="Train localization model")
    p.add_argument("--data_dir",        type=str,   default="./pets_data")
    p.add_argument("--ckpt_dir",        type=str,   default="./checkpoints")
    p.add_argument("--epochs",          type=int,   default=20)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--lr",              type=float, default=5e-4)
    p.add_argument("--freeze_backbone", type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--num_workers",     type=int,   default=2)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)