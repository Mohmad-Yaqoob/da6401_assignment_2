"""
Task 4 training script — unified multi-task pipeline.

Single forward pass → classification + localization + segmentation.
Loss is a weighted sum of the three task losses.

Run example:
    python train_multi.py --epochs 30 --lr 5e-4
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score
import numpy as np
import wandb

from data.pets_dataset import prepare_dataset, get_dataloaders
from models.multitask import MultiTaskModel
from losses.iou_loss import IoULoss


# ── metric helpers ────────────────────────────────────────────────────────────

def cls_accuracy(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item()


@torch.no_grad()
def mean_iou(pred, target, eps=1e-6):
    def corners(b):
        return b[:,0]-b[:,2]/2, b[:,1]-b[:,3]/2, b[:,0]+b[:,2]/2, b[:,1]+b[:,3]/2
    px1,py1,px2,py2 = corners(pred)
    tx1,ty1,tx2,ty2 = corners(target)
    inter = (torch.min(px2,tx2)-torch.max(px1,tx1)).clamp(0) * \
            (torch.min(py2,ty2)-torch.max(py1,ty1)).clamp(0)
    pa = (px2-px1).clamp(0)*(py2-py1).clamp(0)
    ta = (tx2-tx1).clamp(0)*(ty2-ty1).clamp(0)
    return (inter/(pa+ta-inter+eps)).mean().item()


def dice_score(pred_logits, masks, num_classes=3):
    preds = pred_logits.argmax(1)
    scores = []
    for c in range(num_classes):
        p = (preds==c).float(); t = (masks==c).float()
        scores.append(((2*(p*t).sum()+1e-6)/((p+t).sum()+1e-6)).item())
    return sum(scores)/len(scores)


# ── train / eval ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, cls_crit, loc_crit, seg_crit,
                    optimizer, device, w_cls, w_loc, w_seg):
    model.train()
    metrics = dict(loss=0, cls_loss=0, loc_loss=0, seg_loss=0,
                   cls_acc=0, iou=0, dice=0)

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        bboxes = batch["bbox"].to(device)
        masks  = batch["mask"].to(device)

        optimizer.zero_grad()
        cls_logits, bbox_pred, seg_logits = model(images)

        loss_cls = cls_crit(cls_logits, labels)
        loss_loc = loc_crit(bbox_pred, bboxes)
        loss_seg = seg_crit(seg_logits, masks)
        loss = w_cls * loss_cls + w_loc * loss_loc + w_seg * loss_seg

        loss.backward()
        # gradient clipping helps stability in multi-task training
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        metrics["loss"]     += loss.item()
        metrics["cls_loss"] += loss_cls.item()
        metrics["loc_loss"] += loss_loc.item()
        metrics["seg_loss"] += loss_seg.item()
        metrics["cls_acc"]  += cls_accuracy(cls_logits.detach(), labels)
        metrics["iou"]      += mean_iou(bbox_pred.detach(), bboxes)
        metrics["dice"]     += dice_score(seg_logits.detach(), masks)

    n = len(loader)
    return {k: v/n for k, v in metrics.items()}


@torch.no_grad()
def evaluate(model, loader, cls_crit, loc_crit, seg_crit,
             device, w_cls, w_loc, w_seg):
    model.eval()
    metrics = dict(loss=0, cls_loss=0, loc_loss=0, seg_loss=0,
                   cls_acc=0, iou=0, dice=0)
    all_preds, all_labels = [], []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        bboxes = batch["bbox"].to(device)
        masks  = batch["mask"].to(device)

        cls_logits, bbox_pred, seg_logits = model(images)

        loss_cls = cls_crit(cls_logits, labels)
        loss_loc = loc_crit(bbox_pred, bboxes)
        loss_seg = seg_crit(seg_logits, masks)
        loss = w_cls * loss_cls + w_loc * loss_loc + w_seg * loss_seg

        metrics["loss"]     += loss.item()
        metrics["cls_loss"] += loss_cls.item()
        metrics["loc_loss"] += loss_loc.item()
        metrics["seg_loss"] += loss_seg.item()
        metrics["cls_acc"]  += cls_accuracy(cls_logits, labels)
        metrics["iou"]      += mean_iou(bbox_pred, bboxes)
        metrics["dice"]     += dice_score(seg_logits, masks)

        all_preds.extend(cls_logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n = len(loader)
    result = {k: v/n for k, v in metrics.items()}

    # macro F1 for the final evaluation metric
    result["macro_f1"] = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def train(args):
    wandb.init(
        project="da6401-assignment2",
        name=f"multitask_lr{args.lr}",
        config=vars(args),
        tags=["multitask", "unified"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root = prepare_dataset(args.data_dir)
    train_loader, val_loader, test_loader = get_dataloaders(
        root=root, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = MultiTaskModel(num_classes=37, dropout_p=0.5, seg_classes=3).to(device)

    # optionally load from best cls checkpoint to warm-start the backbone
    cls_ckpt = os.path.join(args.ckpt_dir, "best_cls.pth")
    if os.path.exists(cls_ckpt) and args.load_cls:
        state = torch.load(cls_ckpt, map_location="cpu")
        # load only the encoder blocks (keys that match)
        model_state = model.state_dict()
        for k, v in state["model_state"].items():
            # VGG11 keys like "block1.0.0.weight" → multitask "enc1.0.0.weight"
            new_k = k.replace("block1", "enc1").replace("block2", "enc2") \
                     .replace("block3", "enc3").replace("block4", "enc4") \
                     .replace("block5", "enc5")
            if new_k in model_state and model_state[new_k].shape == v.shape:
                model_state[new_k] = v
        model.load_state_dict(model_state)
        print("Warm-started backbone from cls checkpoint.")

    cls_crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    loc_crit = IoULoss(reduction="mean")
    seg_crit = nn.CrossEntropyLoss(
        weight=torch.tensor([1.5, 1.0, 2.0], device=device)
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_dice = 0.0
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(
            model, train_loader, cls_crit, loc_crit, seg_crit,
            optimizer, device, args.w_cls, args.w_loc, args.w_seg
        )
        val = evaluate(
            model, val_loader, cls_crit, loc_crit, seg_crit,
            device, args.w_cls, args.w_loc, args.w_seg
        )
        scheduler.step()

        wandb.log({
            "epoch":             epoch,
            # train
            "train/loss":        tr["loss"],
            "train/cls_loss":    tr["cls_loss"],
            "train/loc_loss":    tr["loc_loss"],
            "train/seg_loss":    tr["seg_loss"],
            "train/cls_acc":     tr["cls_acc"],
            "train/iou":         tr["iou"],
            "train/dice":        tr["dice"],
            # val
            "val/loss":          val["loss"],
            "val/cls_loss":      val["cls_loss"],
            "val/loc_loss":      val["loc_loss"],
            "val/seg_loss":      val["seg_loss"],
            "val/cls_acc":       val["cls_acc"],
            "val/iou":           val["iou"],
            "val/dice":          val["dice"],
            "val/macro_f1":      val["macro_f1"],
            "lr":                scheduler.get_last_lr()[0],
        }, step=epoch)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"loss {tr['loss']:.4f} | "
            f"cls_acc {tr['cls_acc']:.3f} iou {tr['iou']:.3f} dice {tr['dice']:.3f} | "
            f"val_f1 {val['macro_f1']:.3f} val_dice {val['dice']:.3f}"
        )

        if val["dice"] > best_val_dice:
            best_val_dice = val["dice"]
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_metrics": val,
                "args":        vars(args),
            }, os.path.join(args.ckpt_dir, "best_multitask.pth"))
            print(f"  → saved best multitask checkpoint (dice={val['dice']:.4f})")

    # final test set evaluation
    print("\nRunning test set evaluation...")
    test_metrics = evaluate(
        model, test_loader, cls_crit, loc_crit, seg_crit,
        device, args.w_cls, args.w_loc, args.w_seg
    )
    wandb.log({
        "test/macro_f1": test_metrics["macro_f1"],
        "test/iou":      test_metrics["iou"],
        "test/dice":     test_metrics["dice"],
    })
    print(
        f"Test → F1: {test_metrics['macro_f1']:.4f} | "
        f"IoU: {test_metrics['iou']:.4f} | "
        f"Dice: {test_metrics['dice']:.4f}"
    )

    wandb.finish()


def parse_args():
    p = argparse.ArgumentParser(description="Train unified multi-task model")
    p.add_argument("--data_dir",    type=str,   default="./pets_data")
    p.add_argument("--ckpt_dir",    type=str,   default="./checkpoints")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=5e-4)
    p.add_argument("--w_cls",       type=float, default=1.0,
                   help="weight for classification loss")
    p.add_argument("--w_loc",       type=float, default=1.0,
                   help="weight for localization loss")
    p.add_argument("--w_seg",       type=float, default=1.0,
                   help="weight for segmentation loss")
    p.add_argument("--load_cls",    type=lambda x: x.lower()=="true", default=True,
                   help="warm-start backbone from cls checkpoint")
    p.add_argument("--num_workers", type=int,   default=2)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)