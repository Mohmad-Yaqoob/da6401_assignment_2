"""
Task 1 training script — VGG11 classification on Oxford Pets (37 breeds).

Run example:
    python train_cls.py --epochs 30 --lr 1e-3 --batch_size 32 --dropout_p 0.5

W&B logs:
    - train/val loss and accuracy per epoch
    - activation distributions (3rd conv layer) with and without BN
    - dropout ablation curves (no dropout / p=0.2 / p=0.5)
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


# ── helpers ───────────────────────────────────────────────────────────────────

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc  += accuracy(logits, labels)

    n = len(loader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc = 0.0, 0.0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss += loss.item()
        total_acc  += accuracy(logits, labels)

    n = len(loader)
    return total_loss / n, total_acc / n


def log_activation_distribution(model, loader, device, step):
    """
    Pass one batch through, grab activations right after the 3rd conv layer
    (which is the first conv in block2), and log the distribution to W&B.
    This is used for the BN ablation section of the report.
    """
    model.eval()
    activations = {}

    # hook on the ReLU output of block2's first conv-bn-relu triple
    # block2[0] = conv, block2[1] = BN, block2[2] = ReLU
    def hook_fn(module, input, output):
        activations["3rd_conv"] = output.detach().cpu().flatten().numpy()

    handle = model.block2[0][2].register_forward_hook(hook_fn)  # ReLU of block2

    batch = next(iter(loader))
    with torch.no_grad():
        model(batch["image"].to(device))

    handle.remove()

    if "3rd_conv" in activations:
        wandb.log({
            "activations/3rd_conv_layer": wandb.Histogram(activations["3rd_conv"]),
        }, step=step)


# ── main training loop ────────────────────────────────────────────────────────

def train(args):
    # W&B init
    wandb.init(
        project="da6401-assignment2",
        name=f"cls_dropout{args.dropout_p}_lr{args.lr}",
        config=vars(args),
        tags=["classification", "vgg11"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # data
    root = prepare_dataset(args.data_dir)
    train_loader, val_loader, _ = get_dataloaders(
        root=root,
        img_size=224,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # model
    model = VGG11(num_classes=37, dropout_p=args.dropout_p).to(device)
    wandb.watch(model, log="gradients", log_freq=100)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_acc = 0.0
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        # log activation distribution every 5 epochs for the BN analysis
        if epoch % 5 == 0:
            log_activation_distribution(model, val_loader, device, step=epoch)

        wandb.log({
            "epoch":          epoch,
            "train/loss":     train_loss,
            "train/accuracy": train_acc,
            "val/loss":       val_loss,
            "val/accuracy":   val_acc,
            "lr":             scheduler.get_last_lr()[0],
        }, step=epoch)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        # save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.ckpt_dir, "best_cls.pth")
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "val_acc":    val_acc,
                "args":       vars(args),
            }, ckpt_path)
            print(f"  → saved best checkpoint (val_acc={val_acc:.4f})")

    print(f"\nTraining done. Best val accuracy: {best_val_acc:.4f}")
    wandb.finish()
    return model


# ── argument parser ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train VGG11 on Oxford Pets")
    p.add_argument("--data_dir",    type=str,   default="./pets_data")
    p.add_argument("--ckpt_dir",    type=str,   default="./checkpoints")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--dropout_p",   type=float, default=0.5)
    p.add_argument("--num_workers", type=int,   default=2)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)