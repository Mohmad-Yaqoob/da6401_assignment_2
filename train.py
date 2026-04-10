# Training script for DA6401 Assignment 2.

# Usage:
#   python train.py --task classification --epochs 60 --lr 5e-4  
#   python train.py --task localization   --epochs 40 --lr 5e-4  
#   python train.py --task segmentation   --epochs 40 --lr 5e-4

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import wandb
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    precision_recall_fscore_support, confusion_matrix,
)

from data.pets_dataset import OxfordIIITPetDataset, collate_fn
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses.iou_loss import IoULoss


DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_BREEDS = 37
IMAGE_SIZE = 224.0

BREED_NAMES = [
    "Abyssinian", "american_bulldog", "american_pit_bull_terrier",
    "basset_hound", "beagle", "Bengal", "Birman", "Bombay", "boxer",
    "British_Shorthair", "chihuahua", "Egyptian_Mau",
    "english_cocker_spaniel", "english_setter", "german_shorthaired",
    "great_pyrenees", "havanese", "japanese_chin", "keeshond",
    "leonberger", "Maine_Coon", "miniature_pinscher", "newfoundland",
    "Persian", "pomeranian", "pug", "Ragdoll", "Russian_Blue",
    "saint_bernard", "samoyed", "scottish_terrier", "shiba_inu",
    "Siamese", "Sphynx", "staffordshire_bull_terrier",
    "wheaten_terrier", "yorkshire_terrier",
]
SEG_NAMES = ["foreground", "background", "boundary"]


# ── helpers ────────────────────────────────────────────────────────────────

def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return torch.stack([(x1+x2)/2, (y1+y2)/2,
                        (x2-x1).clamp(0), (y2-y1).clamp(0)], dim=1)


def batch_iou(pred: torch.Tensor, target: torch.Tensor, eps=1e-6) -> torch.Tensor:
    def corners(b):
        return torch.stack([b[:,0]-b[:,2]/2, b[:,1]-b[:,3]/2,
                            b[:,0]+b[:,2]/2, b[:,1]+b[:,3]/2], dim=1)
    p, t  = corners(pred), corners(target)
    ix1   = torch.max(p[:,0], t[:,0]); iy1 = torch.max(p[:,1], t[:,1])
    ix2   = torch.min(p[:,2], t[:,2]); iy2 = torch.min(p[:,3], t[:,3])
    inter = (ix2-ix1).clamp(0) * (iy2-iy1).clamp(0)
    pa    = (p[:,2]-p[:,0]).clamp(0) * (p[:,3]-p[:,1]).clamp(0)
    ta    = (t[:,2]-t[:,0]).clamp(0) * (t[:,3]-t[:,1]).clamp(0)
    return inter / (pa + ta - inter + eps)


def precision_at_iou(pred, target, thr):
    return (batch_iou(pred, target) >= thr).float().mean().item()


def clf_metrics(y_true, y_pred):
    labels = list(range(NUM_BREEDS))
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
    pre  = precision_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
    pf1, *_ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                               average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(np.float32)
    cm = cm / (cm.sum(1, keepdims=True) + 1e-6)
    return {"f1": f1, "pre": pre, "rec": rec, "per_f1": pf1, "cm": cm,
            "y_true": np.array(y_true)}


def mixup(imgs, labels, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    return lam * imgs + (1-lam) * imgs[idx], labels, labels[idx], lam


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay  = decay
        self.shadow = {k: v.clone().detach().float() for k, v in model.state_dict().items()}

    def update(self, model):
        d = self.decay
        for k, v in model.state_dict().items():
            self.shadow[k] = d * self.shadow[k] + (1-d) * v.detach().float()

    def apply(self, model):
        self._bak = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict({k: v.to(dtype=self._bak[k].dtype)
                               for k, v in self.shadow.items()})

    def restore(self, model):
        model.load_state_dict(self._bak)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=-1):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        C     = logits.size(1)
        probs = torch.softmax(logits, 1)
        valid = targets != self.ignore_index
        tgt   = targets.clone(); tgt[~valid] = 0
        oh    = nn.functional.one_hot(tgt, C).permute(0,3,1,2).float()
        m     = valid.unsqueeze(1).float()
        inter = (probs*m * oh*m).sum(dim=(0,2,3))
        card  = (probs*m + oh*m).sum(dim=(0,2,3))
        return 1.0 - ((2*inter + self.smooth) / (card + self.smooth)).mean()


def save_ckpt(model, name, ckpt_dir, epoch, metric):
    path = Path(ckpt_dir) / f"{name}.pth"
    torch.save({"state_dict": model.state_dict(),
                "epoch": epoch, "best_metric": metric}, path)
    art = wandb.Artifact(name=f"{name}_model", type="model")
    art.add_file(str(path))
    wandb.log_artifact(art)


def make_loaders(args, mode):
    kw = dict(num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    tr = DataLoader(OxfordIIITPetDataset(args.data_dir, "train", mode),
                    batch_size=args.batch_size, shuffle=True, **kw)
    va = DataLoader(OxfordIIITPetDataset(args.data_dir, "val",   mode),
                    batch_size=args.batch_size, shuffle=False, **kw)
    return tr, va


def make_sched(opt, warmup, total):
    return SequentialLR(opt,
        schedulers=[LinearLR(opt, 0.1, 1.0, total_iters=warmup),
                    CosineAnnealingLR(opt, T_max=max(total-warmup, 1), eta_min=1e-6)],
        milestones=[warmup])


# Task 1

def train_classification(args):
    wandb.init(project=args.wandb_project, name="classification", config=vars(args))
    tr_dl, va_dl = make_loaders(args, "cls")

    model = VGG11Classifier(num_classes=NUM_BREEDS, dropout_p=args.dropout_p).to(DEVICE)
    crit  = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = make_sched(opt, 5, args.epochs)
    ema   = EMA(model)

    best = 0.0
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs+1):
        model.train()
        tl, tt, ty, tp = 0.0, 0, [], []
        for b in tr_dl:
            imgs, lbls = b["image"].to(DEVICE), b["label"].to(DEVICE)
            opt.zero_grad()
            if ep > 5 and args.mixup_alpha > 0:
                imgs, la, lb, lam = mixup(imgs, lbls, args.mixup_alpha)
                out  = model(imgs)
                loss = lam*crit(out, la) + (1-lam)*crit(out, lb)
            else:
                out  = model(imgs)
                loss = crit(out, lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); ema.update(model)
            tl += loss.item()*imgs.size(0); tt += imgs.size(0)
            ty.extend(lbls.cpu().tolist()); tp.extend(out.argmax(1).cpu().tolist())
        sched.step()

        ema.apply(model); model.eval()
        vl, vt, vy, vp = 0.0, 0, [], []
        with torch.no_grad():
            for b in va_dl:
                imgs, lbls = b["image"].to(DEVICE), b["label"].to(DEVICE)
                out   = model(imgs)
                vl   += crit(out, lbls).item()*imgs.size(0); vt += imgs.size(0)
                vy.extend(lbls.cpu().tolist()); vp.extend(out.argmax(1).cpu().tolist())
        ema.restore(model)

        tm = clf_metrics(ty, tp); vm = clf_metrics(vy, vp)
        vacc = float(np.mean(np.array(vy)==np.array(vp)))

        wandb.log({"epoch": ep, "lr": sched.get_last_lr()[0],
                   "train/loss": tl/tt, "val/loss": vl/vt, "val/acc": vacc,
                   "train/macro_f1": tm["f1"], "val/macro_f1": vm["f1"]}, step=ep)

        print(f"[Cls {ep:03d}/{args.epochs}] loss={tl/tt:.4f} val_loss={vl/vt:.4f} "
              f"acc={vacc:.4f} f1={vm['f1']:.4f}")

        if vm["f1"] > best:
            best = vm["f1"]
            ema.apply(model)
            save_ckpt(model, "classifier", args.ckpt_dir, ep, best)
            ema.restore(model)

    wandb.finish()
    print(f"Best val F1: {best:.4f}")


# Task 2

def train_localization(args):
    wandb.init(project=args.wandb_project, name="localization", config=vars(args))
    tr_dl, va_dl = make_loaders(args, "loc")

    model = VGG11Localizer(dropout_p=args.dropout_p).to(DEVICE)

    clf_p = Path(args.ckpt_dir) / "classifier.pth"
    if clf_p.exists():
        sd  = torch.load(clf_p, map_location="cpu")
        sd  = sd.get("state_dict", sd)
        enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
        model.encoder.load_state_dict(enc, strict=True)
        print("Loaded encoder from classifier.pth")

    mse = nn.MSELoss()
    iou = IoULoss(reduction="mean")
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = make_sched(opt, 3, args.epochs)
    best  = 0.0
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs+1):
        model.train()
        tl, tt = 0.0, 0
        for b in tr_dl:
            imgs      = b["image"].to(DEVICE)
            bbox      = b["bbox"].to(DEVICE)              # xyxy pixel
            bmask     = b["bbox_mask"].to(DEVICE).bool()
            if bmask.sum() == 0: continue

            bbox_cx   = xyxy_to_cxcywh(bbox)              # cxcywh pixel
            opt.zero_grad()
            pred      = model(imgs)                        # cxcywh pixel sigmoid*224
            pn        = pred[bmask]   / IMAGE_SIZE         # normalise for loss
            tn        = bbox_cx[bmask] / IMAGE_SIZE
            loss      = mse(pn, tn) + iou(pn, tn)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n = bmask.sum().item()
            tl += loss.item()*n; tt += n

        sched.step()

        model.eval()
        vl, vi, vp50, vp75, vt = 0.0, 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for b in va_dl:
                imgs  = b["image"].to(DEVICE)
                bbox  = b["bbox"].to(DEVICE)
                bmask = b["bbox_mask"].to(DEVICE).bool()
                if bmask.sum() == 0: continue
                bbox_cx = xyxy_to_cxcywh(bbox)
                pred    = model(imgs)
                n       = bmask.sum().item()
                pn      = pred[bmask] / IMAGE_SIZE
                tn      = bbox_cx[bmask] / IMAGE_SIZE
                vl     += (mse(pn, tn) + iou(pn, tn)).item() * n
                vi     += batch_iou(pred[bmask], bbox_cx[bmask]).mean().item() * n
                vp50   += precision_at_iou(pred[bmask], bbox_cx[bmask], 0.50) * n
                vp75   += precision_at_iou(pred[bmask], bbox_cx[bmask], 0.75) * n
                vt     += n

        vl /= max(vt,1); vi /= max(vt,1); vp50 /= max(vt,1); vp75 /= max(vt,1)

        wandb.log({"epoch": ep, "lr": sched.get_last_lr()[0],
                   "train/loss": tl/max(tt,1), "val/loss": vl,
                   "val/mean_iou": vi, "val/p50": vp50, "val/p75": vp75}, step=ep)

        print(f"[Loc {ep:03d}/{args.epochs}] loss={tl/max(tt,1):.4f} "
              f"iou={vi:.4f} P@50={vp50:.4f} P@75={vp75:.4f}")

        if vi > best:
            best = vi
            save_ckpt(model, "localizer", args.ckpt_dir, ep, best)

    wandb.finish()
    print(f"Best val IoU: {best:.4f}")


# Task 3

def train_segmentation(args):
    wandb.init(project=args.wandb_project, name="segmentation", config=vars(args))
    tr_dl, va_dl = make_loaders(args, "seg")

    model = VGG11UNet(num_classes=3, in_channels=3, dropout_p=args.dropout_p).to(DEVICE)

    clf_p = Path(args.ckpt_dir) / "classifier.pth"
    if clf_p.exists():
        sd  = torch.load(clf_p, map_location="cpu").get("state_dict", {})
        enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
        model.encoder.load_state_dict(enc, strict=False)
        print("Loaded encoder from classifier.pth")

    if args.freeze_encoder:
        for p in model.encoder.parameters(): p.requires_grad_(False)

    weights = torch.tensor([1.0, 0.8, 3.0], device=DEVICE)
    ce      = nn.CrossEntropyLoss(ignore_index=-1, weight=weights)
    dice    = DiceLoss(ignore_index=-1)
    opt     = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)
    sched   = make_sched(opt, 5, args.epochs)
    best    = 0.0
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs+1):
        model.train()
        tl, tt = 0.0, 0
        for b in tr_dl:
            imgs, masks = b["image"].to(DEVICE), b["mask"].to(DEVICE)
            opt.zero_grad()
            out  = model(imgs)
            loss = ce(out, masks) + dice(out, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()*imgs.size(0); tt += imgs.size(0)
        sched.step()

        model.eval()
        vl, vt = 0.0, 0
        dice_sum = np.zeros(3)
        px_ok, px_tot = 0, 0
        ap, at = [], []
        with torch.no_grad():
            for b in va_dl:
                imgs, masks = b["image"].to(DEVICE), b["mask"].to(DEVICE)
                out   = model(imgs); preds = out.argmax(1)
                valid = masks >= 0; n = imgs.size(0)
                vl   += (ce(out, masks) + dice(out, masks)).item() * n; vt += n
                for c in range(3):
                    tp = ((preds==c)&(masks==c)&valid).sum().item()
                    fp = ((preds==c)&(masks!=c)&valid).sum().item()
                    fn = ((preds!=c)&(masks==c)&valid).sum().item()
                    d  = 2*tp+fp+fn
                    dice_sum[c] += (2*tp/d if d > 0 else 0.0) * n
                px_ok  += ((preds==masks)&valid).sum().item()
                px_tot += valid.sum().item()
                ap.append(preds[valid].cpu()); at.append(masks[valid].cpu())

        macro_dice = float((dice_sum/vt).mean())
        px_acc     = px_ok / max(px_tot, 1)
        all_p      = torch.cat(ap).numpy(); all_t = torch.cat(at).numpy()
        seg_f1     = f1_score(all_t, all_p, average="macro", zero_division=0)

        wandb.log({"epoch": ep, "lr": sched.get_last_lr()[0],
                   "train/loss": tl/max(tt,1), "val/loss": vl/vt,
                   "val/dice_macro": macro_dice, "val/pixel_acc": px_acc,
                   "val/macro_f1": seg_f1,
                   **{f"val/dice_{SEG_NAMES[i]}": float(dice_sum[i]/vt) for i in range(3)}},
                  step=ep)

        print(f"[Seg {ep:03d}/{args.epochs}] dice={macro_dice:.4f} "
              f"px_acc={px_acc:.4f} f1={seg_f1:.4f}")

        if macro_dice > best:
            best = macro_dice
            save_ckpt(model, "unet", args.ckpt_dir, ep, best)

    wandb.finish()
    print(f"Best val Dice: {best:.4f}")


# CLI

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",            choices=["classification","localization","segmentation"],
                   default="classification")
    p.add_argument("--data_dir",        default="data/oxford_pet")
    p.add_argument("--ckpt_dir",        default="checkpoints")
    p.add_argument("--epochs",          type=int,   default=60)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--lr",              type=float, default=5e-4)
    p.add_argument("--dropout_p",       type=float, default=0.5)
    p.add_argument("--weight_decay",    type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--mixup_alpha",     type=float, default=0.4)
    p.add_argument("--freeze_encoder",  action="store_true")
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--wandb_project",   default="da6401-assignment2")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    {"classification": train_classification,
     "localization":   train_localization,
     "segmentation":   train_segmentation}[args.task](args)