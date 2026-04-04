"""
Training script for all 4 tasks.

Usage:
    python train.py --task cls   --epochs 40 --lr 1e-4 --dropout_p 0.3
    python train.py --task loc   --epochs 30 --lr 5e-4
    python train.py --task seg   --epochs 30 --lr 1e-3 --strategy full
    python train.py --task multi --epochs 30 --lr 5e-4
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score
import wandb

from data.pets_dataset import prepare_dataset, get_dataloaders
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses.iou_loss import IoULoss


IMG_SIZE = 224  # fixed per VGG11 paper


def save_ckpt(path, model, epoch, metric):
    # saving as state_dict key — autograder expects this format
    torch.save({
        "state_dict":  model.state_dict(),
        "epoch":       epoch,
        "best_metric": metric,
    }, path)


# ── metrics ──────────────────────────────────────────────────────────────────

def cls_acc(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item()


def batch_iou(pred, target, eps=1e-6):
    def corners(b):
        return b[:,0]-b[:,2]/2, b[:,1]-b[:,3]/2, b[:,0]+b[:,2]/2, b[:,1]+b[:,3]/2
    px1,py1,px2,py2 = corners(pred)
    tx1,ty1,tx2,ty2 = corners(target)
    inter = (torch.min(px2,tx2)-torch.max(px1,tx1)).clamp(0) * \
            (torch.min(py2,ty2)-torch.max(py1,ty1)).clamp(0)
    pa = (px2-px1).clamp(0)*(py2-py1).clamp(0)
    ta = (tx2-tx1).clamp(0)*(ty2-ty1).clamp(0)
    return (inter/(pa+ta-inter+eps)).mean().item()


def dice(logits, masks, nc=3):
    preds = logits.argmax(1)
    scores = []
    for c in range(nc):
        p = (preds==c).float(); t = (masks==c).float()
        scores.append(((2*(p*t).sum()+1e-6)/((p+t).sum()+1e-6)).item())
    return sum(scores)/len(scores)


# ── Task 1 ────────────────────────────────────────────────────────────────────

def train_cls(args):
    wandb.init(project="da6401-a2", name=f"cls_dp{args.dropout_p}",
               config=vars(args), tags=["cls"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root   = prepare_dataset(args.data_dir)
    tr, va, _ = get_dataloaders(root, batch_size=args.batch_size,
                                num_workers=args.num_workers)

    model = VGG11Classifier(num_classes=37, dropout_p=args.dropout_p).to(device)
    crit  = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt   = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    best = 0.0
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for ep in range(1, args.epochs+1):
        model.train()
        tl, ta = 0.0, 0.0
        for b in tr:
            imgs, lbl = b["image"].to(device), b["label"].to(device)
            opt.zero_grad()
            out  = model(imgs)
            loss = crit(out, lbl)
            loss.backward(); opt.step()
            tl += loss.item(); ta += cls_acc(out, lbl)

        model.eval()
        vl, va_ = 0.0, 0.0
        preds_all, lbl_all = [], []
        with torch.no_grad():
            for b in va:
                imgs, lbl = b["image"].to(device), b["label"].to(device)
                out  = model(imgs)
                vl  += crit(out, lbl).item()
                va_ += cls_acc(out, lbl)
                preds_all.extend(out.argmax(1).cpu().numpy())
                lbl_all.extend(lbl.cpu().numpy())

        n_tr, n_va = len(tr), len(va)
        f1 = f1_score(lbl_all, preds_all, average="macro", zero_division=0)
        sched.step()

        wandb.log({"epoch": ep, "lr": sched.get_last_lr()[0],
                   "train/loss": tl/n_tr, "train/acc": ta/n_tr,
                   "val/loss": vl/n_va, "val/acc": va_/n_va,
                   "val/f1": f1}, step=ep)

        print(f"ep {ep:03d}/{args.epochs} | "
              f"loss {tl/n_tr:.4f} acc {ta/n_tr:.4f} | "
              f"val_loss {vl/n_va:.4f} val_acc {va_/n_va:.4f} f1 {f1:.4f}")

        if va_/n_va > best:
            best = va_/n_va
            save_ckpt(os.path.join(args.ckpt_dir, "classifier.pth"),
                      model, ep, best)
            print(f"  saved classifier.pth (val_acc={best:.4f})")

    wandb.finish()


# ── Task 2 ────────────────────────────────────────────────────────────────────

def train_loc(args):
    wandb.init(project="da6401-a2", name=f"loc_lr{args.lr}",
               config=vars(args), tags=["loc"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root   = prepare_dataset(args.data_dir)
    tr, va, _ = get_dataloaders(root, batch_size=args.batch_size,
                                num_workers=args.num_workers)

    model = VGG11Localizer(dropout_p=args.dropout_p).to(device)

    # load classifier encoder as starting point
    cls_ckpt = os.path.join(args.ckpt_dir, "classifier.pth")
    if os.path.exists(cls_ckpt):
        ckpt = torch.load(cls_ckpt, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        # only load encoder weights
        enc_state = {k.replace("encoder.", ""): v
                     for k, v in state.items() if k.startswith("encoder.")}
        model.encoder.load_state_dict(enc_state, strict=False)
        print("loaded encoder from classifier.pth")

    mse_loss = nn.MSELoss()
    iou_loss = IoULoss(reduction="mean")

    # lower lr for encoder, higher for head
    enc_params  = list(model.encoder.parameters())
    head_params = list(model.head.parameters())
    opt   = optim.Adam([
        {"params": enc_params,  "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    best = 0.0
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for ep in range(1, args.epochs+1):
        model.train()
        tl, ti = 0.0, 0.0
        for b in tr:
            imgs  = b["image"].to(device)
            bboxes = b["bbox"].to(device) * IMG_SIZE  # normalised -> pixel
            opt.zero_grad()
            pred = model(imgs)
            loss = mse_loss(pred, bboxes) + iou_loss(pred, bboxes)
            loss.backward(); opt.step()
            tl += loss.item()
            ti += batch_iou(pred.detach(), bboxes)

        model.eval()
        vl, vi = 0.0, 0.0
        with torch.no_grad():
            for b in va:
                imgs   = b["image"].to(device)
                bboxes = b["bbox"].to(device) * IMG_SIZE
                pred   = model(imgs)
                vl    += (mse_loss(pred, bboxes) + iou_loss(pred, bboxes)).item()
                vi    += batch_iou(pred, bboxes)

        n_tr, n_va = len(tr), len(va)
        sched.step()

        wandb.log({"epoch": ep, "lr": sched.get_last_lr()[0],
                   "train/loss": tl/n_tr, "train/iou": ti/n_tr,
                   "val/loss": vl/n_va,   "val/iou":  vi/n_va}, step=ep)

        print(f"ep {ep:03d}/{args.epochs} | "
              f"loss {tl/n_tr:.4f} iou {ti/n_tr:.4f} | "
              f"val_iou {vi/n_va:.4f}")

        if vi/n_va > best:
            best = vi/n_va
            save_ckpt(os.path.join(args.ckpt_dir, "localizer.pth"),
                      model, ep, best)
            print(f"  saved localizer.pth (val_iou={best:.4f})")

    wandb.finish()


# ── Task 3 ────────────────────────────────────────────────────────────────────

def train_seg(args):
    wandb.init(project="da6401-a2", name=f"seg_{args.strategy}",
               config=vars(args), tags=["seg", args.strategy])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root   = prepare_dataset(args.data_dir)
    tr, va, _ = get_dataloaders(root, batch_size=args.batch_size,
                                num_workers=args.num_workers)

    model = VGG11UNet(num_classes=3).to(device)

    cls_ckpt = os.path.join(args.ckpt_dir, "classifier.pth")
    if os.path.exists(cls_ckpt):
        ckpt = torch.load(cls_ckpt, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        enc_state = {k.replace("encoder.", ""): v
                     for k, v in state.items() if k.startswith("encoder.")}
        model.encoder.load_state_dict(enc_state, strict=False)
        print("loaded encoder from classifier.pth")

    crit = nn.CrossEntropyLoss(
        weight=torch.tensor([1.5, 1.0, 2.0], device=device)
    )

    if args.strategy == "frozen":
        for p in model.encoder.parameters():
            p.requires_grad = False
        opt = optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=1e-4
        )
    elif args.strategy == "partial":
        for p in model.encoder.block1.parameters(): p.requires_grad = False
        for p in model.encoder.block2.parameters(): p.requires_grad = False
        for p in model.encoder.block3.parameters(): p.requires_grad = False
        late = (list(model.encoder.block4.parameters()) +
                list(model.encoder.block5.parameters()))
        dec  = [p for n, p in model.named_parameters() if "encoder" not in n]
        opt  = optim.Adam([
            {"params": late, "lr": args.lr * 0.1},
            {"params": dec,  "lr": args.lr},
        ], weight_decay=1e-4)
    else:  # full
        enc_p = list(model.encoder.parameters())
        dec_p = [p for n, p in model.named_parameters() if "encoder" not in n]
        opt   = optim.Adam([
            {"params": enc_p, "lr": args.lr * 0.1},
            {"params": dec_p, "lr": args.lr},
        ], weight_decay=1e-4)

    sched = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    best  = 0.0
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for ep in range(1, args.epochs+1):
        model.train()
        tl, td = 0.0, 0.0
        for b in tr:
            imgs, masks = b["image"].to(device), b["mask"].to(device)
            opt.zero_grad()
            out  = model(imgs)
            loss = crit(out, masks)
            loss.backward(); opt.step()
            tl += loss.item(); td += dice(out.detach(), masks)

        model.eval()
        vl, vd, va = 0.0, 0.0, 0.0
        with torch.no_grad():
            for b in va:
                imgs, masks = b["image"].to(device), b["mask"].to(device)
                out  = model(imgs)
                vl  += crit(out, masks).item()
                vd  += dice(out, masks)
                va  += (out.argmax(1) == masks).float().mean().item()

        n_tr, n_va = len(tr), len(va)
        sched.step()

        wandb.log({"epoch": ep, "lr": sched.get_last_lr()[0],
                   "train/loss": tl/n_tr, "train/dice": td/n_tr,
                   "val/loss": vl/n_va,   "val/dice":   vd/n_va,
                   "val/pixel_acc": va/n_va}, step=ep)

        print(f"ep {ep:03d}/{args.epochs} | "
              f"dice {td/n_tr:.4f} | val_dice {vd/n_va:.4f} acc {va/n_va:.4f}")

        if vd/n_va > best:
            best = vd/n_va
            save_ckpt(os.path.join(args.ckpt_dir, "unet.pth"),
                      model, ep, best)
            print(f"  saved unet.pth (val_dice={best:.4f})")

    wandb.finish()


# ── Task 4 ────────────────────────────────────────────────────────────────────

def train_multi(args):
    from multitask import MultiTaskPerceptionModel
    wandb.init(project="da6401-a2", name=f"multi_lr{args.lr}",
               config=vars(args), tags=["multi"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root   = prepare_dataset(args.data_dir)
    tr, va, te = get_dataloaders(root, batch_size=args.batch_size,
                                 num_workers=args.num_workers)

    # build multitask without gdown by loading locally
    from models.vgg11 import VGG11Encoder
    from models.layers import CustomDropout
    from models.segmentation import VGG11UNet
    from models.localization import VGG11Localizer
    from models.classification import VGG11Classifier

    # just reuse the three trained models directly in a wrapper
    cls_model = VGG11Classifier(num_classes=37, dropout_p=0.3).to(device)
    loc_model = VGG11Localizer(dropout_p=0.5).to(device)
    seg_model = VGG11UNet(num_classes=3).to(device)

    for name, m, key in [
        ("classifier.pth", cls_model, "state_dict"),
        ("localizer.pth",  loc_model, "state_dict"),
        ("unet.pth",       seg_model, "state_dict"),
    ]:
        path = os.path.join(args.ckpt_dir, name)
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=device)
            m.load_state_dict(ckpt.get(key, ckpt), strict=True)
            print(f"loaded {name}")

    cls_crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    loc_crit = nn.MSELoss()
    iou_crit = IoULoss(reduction="mean")
    seg_crit = nn.CrossEntropyLoss(
        weight=torch.tensor([1.5, 1.0, 2.0], device=device)
    )

    all_params = (list(cls_model.parameters()) +
                  list(loc_model.parameters()) +
                  list(seg_model.parameters()))
    opt   = optim.Adam(all_params, lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    best_f1 = 0.0
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for ep in range(1, args.epochs+1):
        cls_model.train(); loc_model.train(); seg_model.train()
        tl = 0.0
        for b in tr:
            imgs   = b["image"].to(device)
            labels = b["label"].to(device)
            bboxes = b["bbox"].to(device) * IMG_SIZE
            masks  = b["mask"].to(device)

            opt.zero_grad()
            cls_out = cls_model(imgs)
            loc_out = loc_model(imgs)
            seg_out = seg_model(imgs)

            loss = (args.w_cls * cls_crit(cls_out, labels) +
                    args.w_loc * (loc_crit(loc_out, bboxes) + iou_crit(loc_out, bboxes)) +
                    args.w_seg * seg_crit(seg_out, masks))
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, 5.0)
            opt.step()
            tl += loss.item()

        cls_model.eval(); loc_model.eval(); seg_model.eval()
        vl, preds_all, lbl_all = 0.0, [], []
        vi, vd = 0.0, 0.0
        with torch.no_grad():
            for b in va:
                imgs   = b["image"].to(device)
                labels = b["label"].to(device)
                bboxes = b["bbox"].to(device) * IMG_SIZE
                masks  = b["mask"].to(device)
                cls_out = cls_model(imgs)
                loc_out = loc_model(imgs)
                seg_out = seg_model(imgs)
                vl += (args.w_cls * cls_crit(cls_out, labels) +
                       args.w_loc * (loc_crit(loc_out, bboxes) + iou_crit(loc_out, bboxes)) +
                       args.w_seg * seg_crit(seg_out, masks)).item()
                preds_all.extend(cls_out.argmax(1).cpu().numpy())
                lbl_all.extend(labels.cpu().numpy())
                vi += batch_iou(loc_out, bboxes)
                vd += dice(seg_out, masks)

        n_tr, n_va = len(tr), len(va)
        f1 = f1_score(lbl_all, preds_all, average="macro", zero_division=0)
        sched.step()

        wandb.log({"epoch": ep, "train/loss": tl/n_tr,
                   "val/loss": vl/n_va, "val/f1": f1,
                   "val/iou": vi/n_va, "val/dice": vd/n_va}, step=ep)

        print(f"ep {ep:03d}/{args.epochs} | f1 {f1:.4f} iou {vi/n_va:.4f} dice {vd/n_va:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            # save updated checkpoints
            save_ckpt(os.path.join(args.ckpt_dir, "classifier.pth"), cls_model, ep, f1)
            save_ckpt(os.path.join(args.ckpt_dir, "localizer.pth"),  loc_model, ep, vi/n_va)
            save_ckpt(os.path.join(args.ckpt_dir, "unet.pth"),       seg_model, ep, vd/n_va)
            print(f"  updated all checkpoints (f1={f1:.4f})")

    wandb.finish()


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",        type=str,   default="cls",
                   choices=["cls","loc","seg","multi"])
    p.add_argument("--data_dir",    type=str,   default="./pets_data")
    p.add_argument("--ckpt_dir",    type=str,   default="./checkpoints")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--dropout_p",   type=float, default=0.3)
    p.add_argument("--strategy",    type=str,   default="full",
                   choices=["frozen","partial","full"])
    p.add_argument("--w_cls",       type=float, default=1.0)
    p.add_argument("--w_loc",       type=float, default=1.0)
    p.add_argument("--w_seg",       type=float, default=1.0)
    p.add_argument("--num_workers", type=int,   default=2)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    {"cls": train_cls, "loc": train_loc,
     "seg": train_seg, "multi": train_multi}[args.task](args)