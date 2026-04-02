"""
Task 3 training script — U-Net segmentation on Oxford Pets trimaps.

Trimap classes: 0=foreground, 1=background, 2=border/unknown.
Loss: CrossEntropyLoss (justified in segmentation.py docstring).

Also runs the 3-strategy transfer learning comparison for W&B section 2.3:
    --strategy frozen     → entire backbone frozen
    --strategy partial    → only last 2 enc blocks unfrozen
    --strategy full       → everything unfrozen

Run example:
    python train_seg.py --epochs 30 --lr 1e-3 --strategy full
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

from data.pets_dataset import prepare_dataset, get_dataloaders
from models.vgg11 import VGG11
from models.segmentation import SegmentationModel


# ── metrics ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_metrics(pred_logits: torch.Tensor, masks: torch.Tensor, num_classes: int = 3):
    """
    Returns pixel accuracy and mean Dice score across classes.
    pred_logits : (B, C, H, W)
    masks       : (B, H, W) long
    """
    preds = pred_logits.argmax(dim=1)   # (B, H, W)

    # pixel accuracy
    pixel_acc = (preds == masks).float().mean().item()

    # dice per class then average
    dice_scores = []
    for c in range(num_classes):
        pred_c   = (preds == c).float()
        target_c = (masks == c).float()
        inter    = (pred_c * target_c).sum()
        union    = pred_c.sum() + target_c.sum()
        dice     = (2 * inter + 1e-6) / (union + 1e-6)
        dice_scores.append(dice.item())

    return pixel_acc, sum(dice_scores) / len(dice_scores)


# ── train / eval ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_acc, total_dice = 0.0, 0.0, 0.0

    for batch in loader:
        images = batch["image"].to(device)
        masks  = batch["mask"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        acc, dice = compute_metrics(logits.detach(), masks)
        total_loss += loss.item()
        total_acc  += acc
        total_dice += dice

    n = len(loader)
    return total_loss / n, total_acc / n, total_dice / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, total_dice = 0.0, 0.0, 0.0

    for batch in loader:
        images = batch["image"].to(device)
        masks  = batch["mask"].to(device)

        logits = model(images)
        loss   = criterion(logits, masks)

        acc, dice = compute_metrics(logits, masks)
        total_loss += loss.item()
        total_acc  += acc
        total_dice += dice

    n = len(loader)
    return total_loss / n, total_acc / n, total_dice / n


def setup_optimizer(model, strategy: str, base_lr: float):
    """
    Build optimizer param groups based on the transfer learning strategy.
    frozen  → only decoder trains
    partial → decoder + enc4 + enc5 train
    full    → everything trains
    """
    if strategy == "frozen":
        # freeze entire encoder
        for enc in [model.enc1, model.enc2, model.enc3, model.enc4, model.enc5]:
            for p in enc.parameters():
                p.requires_grad = False
        trainable = [p for p in model.parameters() if p.requires_grad]
        return optim.Adam(trainable, lr=base_lr, weight_decay=1e-4)

    elif strategy == "partial":
        # freeze early blocks, unfreeze last two
        for enc in [model.enc1, model.enc2, model.enc3]:
            for p in enc.parameters():
                p.requires_grad = False
        for enc in [model.enc4, model.enc5]:
            for p in enc.parameters():
                p.requires_grad = True

        early_params  = [p for p in model.parameters() if not p.requires_grad]
        late_enc_params = list(model.enc4.parameters()) + list(model.enc5.parameters())
        decoder_params  = [p for name, p in model.named_parameters()
                           if "enc" not in name]
        return optim.Adam([
            {"params": late_enc_params, "lr": base_lr * 0.1},
            {"params": decoder_params,  "lr": base_lr},
        ], weight_decay=1e-4)

    else:  # full fine-tuning
        enc_params     = (list(model.enc1.parameters()) + list(model.enc2.parameters()) +
                          list(model.enc3.parameters()) + list(model.enc4.parameters()) +
                          list(model.enc5.parameters()))
        decoder_params = [p for name, p in model.named_parameters() if "enc" not in name]
        return optim.Adam([
            {"params": enc_params,     "lr": base_lr * 0.1},
            {"params": decoder_params, "lr": base_lr},
        ], weight_decay=1e-4)


# ── main ──────────────────────────────────────────────────────────────────────

def train(args):
    wandb.init(
        project="da6401-assignment2",
        name=f"seg_{args.strategy}_lr{args.lr}",
        config=vars(args),
        tags=["segmentation", "unet", args.strategy],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root = prepare_dataset(args.data_dir)
    train_loader, val_loader, _ = get_dataloaders(
        root=root, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # load pretrained VGG11
    vgg = VGG11(num_classes=37, dropout_p=0.5)
    cls_ckpt = os.path.join(args.ckpt_dir, "best_cls.pth")
    if os.path.exists(cls_ckpt):
        state = torch.load(cls_ckpt, map_location="cpu")
        vgg.load_state_dict(state["model_state"])
        print(f"Loaded cls checkpoint from {cls_ckpt}")
    else:
        print("No cls checkpoint found — using random VGG11 weights")

    model = SegmentationModel(vgg, num_classes=3, freeze_backbone=False).to(device)

    # class weights to handle background/border imbalance slightly
    # foreground ~30%, background ~55%, border ~15% of pixels roughly
    class_weights = torch.tensor([1.5, 1.0, 2.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = setup_optimizer(model, args.strategy, args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_dice = 0.0
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_dice = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        wandb.log({
            "epoch":            epoch,
            "train/loss":       tr_loss,
            "train/pixel_acc":  tr_acc,
            "train/dice":       tr_dice,
            "val/loss":         val_loss,
            "val/pixel_acc":    val_acc,
            "val/dice":         val_dice,
            "lr":               scheduler.get_last_lr()[0],
        }, step=epoch)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {tr_loss:.4f} dice {tr_dice:.4f} | "
            f"val loss {val_loss:.4f} dice {val_dice:.4f} acc {val_acc:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            ckpt_path = os.path.join(args.ckpt_dir, f"best_seg_{args.strategy}.pth")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_dice":    val_dice,
                "args":        vars(args),
            }, ckpt_path)
            print(f"  → saved best checkpoint (val_dice={val_dice:.4f})")

    print(f"\nDone. Best val Dice: {best_val_dice:.4f}")
    wandb.finish()


def parse_args():
    p = argparse.ArgumentParser(description="Train U-Net segmentation")
    p.add_argument("--data_dir",    type=str,   default="./pets_data")
    p.add_argument("--ckpt_dir",    type=str,   default="./checkpoints")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--strategy",    type=str,   default="full",
                   choices=["frozen", "partial", "full"])
    p.add_argument("--num_workers", type=int,   default=2)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)