"""
train.py — unified training script for all 4 tasks.

Usage:
    python train.py --task cls --epochs 30 --lr 1e-4 --dropout_p 0.5
    python train.py --task loc --epochs 30 --lr 5e-4
    python train.py --task seg --epochs 30 --lr 1e-3 --strategy full
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
from models.vgg11 import VGG11
from models.localization import LocalizationModel
from models.segmentation import SegmentationModel
from losses.iou_loss import IoULoss
from models.layers import CustomDropout


# ── shared metrics ─────────────────────────────────────────────────────────────

def cls_accuracy(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item()


def mean_iou_pixel(pred, target, eps=1e-6):
    """IoU metric in pixel space for logging (not the loss)."""
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


# ── Task 1: Classification ─────────────────────────────────────────────────────

def train_cls(args):
    from models.vgg11 import VGG11

    wandb.init(project="da6401-assignment2",
               name=f"cls_dp{args.dropout_p}_lr{args.lr}",
               config=vars(args), tags=["classification"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root   = prepare_dataset(args.data_dir)
    train_loader, val_loader, _ = get_dataloaders(root=root, batch_size=args.batch_size,
                                                   num_workers=args.num_workers)

    model     = VGG11(num_classes=37, dropout_p=args.dropout_p).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_acc = 0.0
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss, tr_acc = 0.0, 0.0
        for batch in train_loader:
            imgs   = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item(); tr_acc += cls_accuracy(logits, labels)

        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs   = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(imgs)
                val_loss += criterion(logits, labels).item()
                val_acc  += cls_accuracy(logits, labels)

        n_tr, n_val = len(train_loader), len(val_loader)
        scheduler.step()

        wandb.log({"epoch": epoch,
                   "train/loss": tr_loss/n_tr, "train/accuracy": tr_acc/n_tr,
                   "val/loss": val_loss/n_val, "val/accuracy": val_acc/n_val,
                   "lr": scheduler.get_last_lr()[0]}, step=epoch)

        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"train loss {tr_loss/n_tr:.4f} acc {tr_acc/n_tr:.4f} | "
              f"val loss {val_loss/n_val:.4f} acc {val_acc/n_val:.4f}")

        if val_acc/n_val > best_val_acc:
            best_val_acc = val_acc/n_val
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_acc": best_val_acc},
                       os.path.join(args.ckpt_dir, "classifier.pth"))
            print(f"  → saved classifier.pth (val_acc={best_val_acc:.4f})")

    wandb.finish()


# ── Task 2: Localization ───────────────────────────────────────────────────────

def train_loc(args):
    wandb.init(project="da6401-assignment2",
               name=f"loc_lr{args.lr}",
               config=vars(args), tags=["localization"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root   = prepare_dataset(args.data_dir)
    train_loader, val_loader, _ = get_dataloaders(root=root, batch_size=args.batch_size,
                                                   num_workers=args.num_workers)

    vgg = VGG11(num_classes=37, dropout_p=0.5)
    cls_ckpt = os.path.join(args.ckpt_dir, "classifier.pth")
    if os.path.exists(cls_ckpt):
        state = torch.load(cls_ckpt, map_location="cpu")
        vgg.load_state_dict(state["model_state"])
        print("Loaded classifier.pth backbone")

    model = LocalizationModel(vgg, freeze_backbone=args.freeze_backbone).to(device)

    # MSE + IoU loss (both in pixel space)
    mse_loss = nn.MSELoss()
    iou_loss = IoULoss(reduction="mean")

    # lower lr for backbone, higher for head
    enc_params  = (list(model.block1.parameters()) + list(model.block2.parameters()) +
                   list(model.block3.parameters()) + list(model.block4.parameters()) +
                   list(model.block5.parameters()))
    head_params = list(model.reg_head.parameters())
    optimizer   = optim.Adam([{"params": enc_params,  "lr": args.lr * 0.1},
                               {"params": head_params, "lr": args.lr}],
                              weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_iou = 0.0
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # we need pixel-space targets — scale bbox from normalised to pixel
    IMG_SIZE = 224.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss, tr_iou = 0.0, 0.0
        for batch in train_loader:
            imgs  = batch["image"].to(device)
            # convert normalised [cx,cy,w,h] -> pixel space
            bboxes = batch["bbox"].to(device) * IMG_SIZE

            optimizer.zero_grad()
            pred  = model(imgs)
            loss  = mse_loss(pred, bboxes) + iou_loss(pred, bboxes)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            tr_iou  += mean_iou_pixel(pred.detach(), bboxes)

        model.eval()
        val_loss, val_iou = 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs   = batch["image"].to(device)
                bboxes = batch["bbox"].to(device) * IMG_SIZE
                pred   = model(imgs)
                val_loss += (mse_loss(pred, bboxes) + iou_loss(pred, bboxes)).item()
                val_iou  += mean_iou_pixel(pred, bboxes)

        n_tr, n_val = len(train_loader), len(val_loader)
        scheduler.step()

        wandb.log({"epoch": epoch,
                   "train/loss": tr_loss/n_tr, "train/iou": tr_iou/n_tr,
                   "val/loss": val_loss/n_val, "val/iou": val_iou/n_val,
                   "lr": scheduler.get_last_lr()[0]}, step=epoch)

        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"train loss {tr_loss/n_tr:.4f} iou {tr_iou/n_tr:.4f} | "
              f"val loss {val_loss/n_val:.4f} iou {val_iou/n_val:.4f}")

        if val_iou/n_val > best_val_iou:
            best_val_iou = val_iou/n_val
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_iou": best_val_iou},
                       os.path.join(args.ckpt_dir, "localizer.pth"))
            print(f"  → saved localizer.pth (val_iou={best_val_iou:.4f})")

    wandb.finish()


# ── Task 3: Segmentation ───────────────────────────────────────────────────────

def train_seg(args):
    wandb.init(project="da6401-assignment2",
               name=f"seg_{args.strategy}_lr{args.lr}",
               config=vars(args), tags=["segmentation", args.strategy])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root   = prepare_dataset(args.data_dir)
    train_loader, val_loader, _ = get_dataloaders(root=root, batch_size=args.batch_size,
                                                   num_workers=args.num_workers)

    vgg = VGG11(num_classes=37, dropout_p=0.5)
    cls_ckpt = os.path.join(args.ckpt_dir, "classifier.pth")
    if os.path.exists(cls_ckpt):
        state = torch.load(cls_ckpt, map_location="cpu")
        vgg.load_state_dict(state["model_state"])
        print("Loaded classifier.pth backbone for segmentation")

    model     = SegmentationModel(vgg, num_classes=3, freeze_backbone=False).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.5,1.0,2.0], device=device))

    # optimizer based on strategy
    if args.strategy == "frozen":
        for enc in [model.enc1,model.enc2,model.enc3,model.enc4,model.enc5]:
            for p in enc.parameters(): p.requires_grad = False
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable, lr=args.lr, weight_decay=1e-4)
    elif args.strategy == "partial":
        for enc in [model.enc1,model.enc2,model.enc3]:
            for p in enc.parameters(): p.requires_grad = False
        late = list(model.enc4.parameters()) + list(model.enc5.parameters())
        dec  = [p for n,p in model.named_parameters() if "enc" not in n]
        optimizer = optim.Adam([{"params":late,"lr":args.lr*0.1},
                                 {"params":dec, "lr":args.lr}], weight_decay=1e-4)
    else:  # full
        enc_p = (list(model.enc1.parameters())+list(model.enc2.parameters())+
                 list(model.enc3.parameters())+list(model.enc4.parameters())+
                 list(model.enc5.parameters()))
        dec_p = [p for n,p in model.named_parameters() if "enc" not in n]
        optimizer = optim.Adam([{"params":enc_p,"lr":args.lr*0.1},
                                 {"params":dec_p,"lr":args.lr}], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    best_dice = 0.0
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss, tr_dice = 0.0, 0.0
        for batch in train_loader:
            imgs  = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item(); tr_dice += dice_score(logits.detach(), masks)

        model.eval()
        val_loss, val_dice, val_acc = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs  = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(imgs)
                val_loss += criterion(logits, masks).item()
                val_dice += dice_score(logits, masks)
                val_acc  += (logits.argmax(1)==masks).float().mean().item()

        n_tr, n_val = len(train_loader), len(val_loader)
        scheduler.step()

        wandb.log({"epoch": epoch,
                   "train/loss": tr_loss/n_tr, "train/dice": tr_dice/n_tr,
                   "val/loss": val_loss/n_val, "val/dice": val_dice/n_val,
                   "val/pixel_acc": val_acc/n_val,
                   "lr": scheduler.get_last_lr()[0]}, step=epoch)

        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"train dice {tr_dice/n_tr:.4f} | "
              f"val dice {val_dice/n_val:.4f} acc {val_acc/n_val:.4f}")

        if val_dice/n_val > best_dice:
            best_dice = val_dice/n_val
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_dice": best_dice},
                       os.path.join(args.ckpt_dir, "unet.pth"))
            print(f"  → saved unet.pth (val_dice={best_dice:.4f})")

    wandb.finish()


# ── Task 4: Multi-task ─────────────────────────────────────────────────────────

def train_multi(args):
    from models.multitask import MultiTaskPerceptionModel

    wandb.init(project="da6401-assignment2",
               name=f"multi_lr{args.lr}",
               config=vars(args), tags=["multitask"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root   = prepare_dataset(args.data_dir)
    train_loader, val_loader, test_loader = get_dataloaders(
        root=root, batch_size=args.batch_size, num_workers=args.num_workers)

    # build model manually (skip gdown for training)
    vgg = VGG11(num_classes=37, dropout_p=0.5)
    cls_ckpt = os.path.join(args.ckpt_dir, "classifier.pth")
    if os.path.exists(cls_ckpt):
        state = torch.load(cls_ckpt, map_location="cpu")
        vgg.load_state_dict(state["model_state"])

    # import internal model parts
    from models.segmentation import SegmentationModel
    seg_model = SegmentationModel(vgg, num_classes=3, freeze_backbone=False)

    # build a clean multitask model using the same architecture
    class _MultiTask(nn.Module):
        def __init__(self, vgg):
            super().__init__()
            self.enc1=vgg.block1; self.enc2=vgg.block2; self.enc3=vgg.block3
            self.enc4=vgg.block4; self.enc5=vgg.block5
            self.avgpool=nn.AdaptiveAvgPool2d((7,7))
            flat=512*7*7
            self.cls_head=nn.Sequential(nn.Linear(flat,4096),nn.ReLU(True),
                                         CustomDropout(0.5),nn.Linear(4096,4096),
                                         nn.ReLU(True),CustomDropout(0.5),nn.Linear(4096,37))
            self.loc_head=nn.Sequential(nn.Linear(flat,1024),nn.ReLU(True),
                                         nn.Dropout(0.3),nn.Linear(1024,256),
                                         nn.ReLU(True),nn.Linear(256,4),nn.ReLU(True))
            self.up5=nn.ConvTranspose2d(512,512,2,stride=2); self.dec5=_dec_block(1024,512)
            self.up4=nn.ConvTranspose2d(512,256,2,stride=2); self.dec4=_dec_block(512,256)
            self.up3=nn.ConvTranspose2d(256,128,2,stride=2); self.dec3=_dec_block(256,128)
            self.up2=nn.ConvTranspose2d(128,64,2,stride=2);  self.dec2=_dec_block(128,64)
            self.up1=nn.ConvTranspose2d(64,32,2,stride=2);   self.dec1=_dec_block(32,32)
            self.seg_final=nn.Conv2d(32,3,1)
        def forward(self,x):
            s1=self.enc1(x); s2=self.enc2(s1); s3=self.enc3(s2)
            s4=self.enc4(s3); s5=self.enc5(s4)
            p=self.avgpool(s5); f=torch.flatten(p,1)
            cls=self.cls_head(f); loc=self.loc_head(f)
            d=self.up5(s5);  d=self.dec5(torch.cat([d,s4],1))
            d=self.up4(d);   d=self.dec4(torch.cat([d,s3],1))
            d=self.up3(d);   d=self.dec3(torch.cat([d,s2],1))
            d=self.up2(d);   d=self.dec2(torch.cat([d,s1],1))
            d=self.up1(d);   d=self.dec1(d)
            return cls, loc, self.seg_final(d)

    model     = _MultiTask(vgg).to(device)
    cls_crit  = nn.CrossEntropyLoss(label_smoothing=0.1)
    loc_crit  = nn.MSELoss()
    iou_crit  = IoULoss(reduction="mean")
    seg_crit  = nn.CrossEntropyLoss(weight=torch.tensor([1.5,1.0,2.0],device=device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    IMG_SIZE  = 224.0
    best_dice = 0.0
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr = dict(loss=0,cls=0,loc=0,seg=0,acc=0,iou=0,dice=0)
        for batch in train_loader:
            imgs   = batch["image"].to(device)
            labels = batch["label"].to(device)
            bboxes = batch["bbox"].to(device) * IMG_SIZE
            masks  = batch["mask"].to(device)
            optimizer.zero_grad()
            cls_l, loc_l, seg_l = model(imgs)
            l_cls = cls_crit(cls_l, labels)
            l_loc = loc_crit(loc_l, bboxes) + iou_crit(loc_l, bboxes)
            l_seg = seg_crit(seg_l, masks)
            loss  = args.w_cls*l_cls + args.w_loc*l_loc + args.w_seg*l_seg
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            tr["loss"]+=loss.item(); tr["cls"]+=l_cls.item()
            tr["loc"]+=l_loc.item(); tr["seg"]+=l_seg.item()
            tr["acc"]+=cls_accuracy(cls_l.detach(),labels)
            tr["iou"]+=mean_iou_pixel(loc_l.detach(),bboxes)
            tr["dice"]+=dice_score(seg_l.detach(),masks)

        model.eval()
        val=dict(loss=0,cls=0,loc=0,seg=0,acc=0,iou=0,dice=0)
        all_pred,all_lbl=[],[]
        with torch.no_grad():
            for batch in val_loader:
                imgs   = batch["image"].to(device)
                labels = batch["label"].to(device)
                bboxes = batch["bbox"].to(device) * IMG_SIZE
                masks  = batch["mask"].to(device)
                cls_l, loc_l, seg_l = model(imgs)
                l_cls = cls_crit(cls_l,labels)
                l_loc = loc_crit(loc_l,bboxes)+iou_crit(loc_l,bboxes)
                l_seg = seg_crit(seg_l,masks)
                loss  = args.w_cls*l_cls+args.w_loc*l_loc+args.w_seg*l_seg
                val["loss"]+=loss.item(); val["cls"]+=l_cls.item()
                val["loc"]+=l_loc.item(); val["seg"]+=l_seg.item()
                val["acc"]+=cls_accuracy(cls_l,labels)
                val["iou"]+=mean_iou_pixel(loc_l,bboxes)
                val["dice"]+=dice_score(seg_l,masks)
                all_pred.extend(cls_l.argmax(1).cpu().numpy())
                all_lbl.extend(labels.cpu().numpy())

        n_tr,n_val=len(train_loader),len(val_loader)
        macro_f1=f1_score(all_lbl,all_pred,average="macro",zero_division=0)
        scheduler.step()

        wandb.log({"epoch":epoch,
                   "train/loss":tr["loss"]/n_tr,"train/cls_acc":tr["acc"]/n_tr,
                   "train/iou":tr["iou"]/n_tr,"train/dice":tr["dice"]/n_tr,
                   "val/loss":val["loss"]/n_val,"val/cls_acc":val["acc"]/n_val,
                   "val/iou":val["iou"]/n_val,"val/dice":val["dice"]/n_val,
                   "val/macro_f1":macro_f1,"lr":scheduler.get_last_lr()[0]},step=epoch)

        print(f"Epoch {epoch:03d}/{args.epochs} | loss {tr['loss']/n_tr:.4f} | "
              f"cls {tr['acc']/n_tr:.3f} iou {tr['iou']/n_tr:.3f} dice {tr['dice']/n_tr:.3f} | "
              f"val_f1 {macro_f1:.3f} val_dice {val['dice']/n_val:.3f}")

        if val["dice"]/n_val > best_dice:
            best_dice = val["dice"]/n_val
            torch.save({"epoch":epoch,"model_state":model.state_dict(),
                        "val_dice":best_dice},
                       os.path.join(args.ckpt_dir,"best_multitask.pth"))
            print(f"  → saved best_multitask.pth (dice={best_dice:.4f})")

    wandb.finish()


# ── argument parser ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",           type=str,   default="cls",
                   choices=["cls","loc","seg","multi"])
    p.add_argument("--data_dir",       type=str,   default="./pets_data")
    p.add_argument("--ckpt_dir",       type=str,   default="./checkpoints")
    p.add_argument("--epochs",         type=int,   default=30)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--dropout_p",      type=float, default=0.5)
    p.add_argument("--freeze_backbone",type=lambda x: x.lower()=="true", default=False)
    p.add_argument("--strategy",       type=str,   default="full",
                   choices=["frozen","partial","full"])
    p.add_argument("--w_cls",          type=float, default=1.0)
    p.add_argument("--w_loc",          type=float, default=1.0)
    p.add_argument("--w_seg",          type=float, default=1.0)
    p.add_argument("--num_workers",    type=int,   default=2)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.task == "cls":
        train_cls(args)
    elif args.task == "loc":
        train_loc(args)
    elif args.task == "seg":
        train_seg(args)
    elif args.task == "multi":
        train_multi(args)