"""
Custom IoU Loss for bounding box regression.

Input format: [x_center, y_center, width, height] — all normalised to [0,1].

The loss is defined as:  L = 1 - IoU
so it sits in [0, 1] and gradients flow back through the box coords.

Key implementation notes:
  - We convert cx/cy/w/h → x1/y1/x2/y2 for the intersection calculation.
  - A small epsilon guards against division-by-zero when union = 0.
  - We use torch operations throughout so autograd handles the gradient automatically.
"""

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """
    Intersection over Union loss for single-object bounding box regression.

    Args:
        reduction : "mean" averages over the batch, "sum" sums, "none" returns per-sample loss
        eps       : small constant for numerical stability in the denominator
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-6):
        super().__init__()
        assert reduction in ("mean", "sum", "none"), \
            f"reduction must be mean / sum / none, got '{reduction}'"
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   : (N, 4) predicted boxes  [cx, cy, w, h] in [0,1]
            target : (N, 4) ground truth boxes [cx, cy, w, h] in [0,1]

        Returns:
            scalar loss if reduction is mean/sum, else (N,) tensor
        """
        # ── convert [cx, cy, w, h] → [x1, y1, x2, y2] ─────────────────────
        pred_x1 = pred[:, 0] - pred[:, 2] / 2
        pred_y1 = pred[:, 1] - pred[:, 3] / 2
        pred_x2 = pred[:, 0] + pred[:, 2] / 2
        pred_y2 = pred[:, 1] + pred[:, 3] / 2

        tgt_x1 = target[:, 0] - target[:, 2] / 2
        tgt_y1 = target[:, 1] - target[:, 3] / 2
        tgt_x2 = target[:, 0] + target[:, 2] / 2
        tgt_y2 = target[:, 1] + target[:, 3] / 2

        # ── intersection ────────────────────────────────────────────────────
        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)

        # clamp at 0 — if boxes don't overlap the intersection is zero
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        # ── union ────────────────────────────────────────────────────────────
        pred_area   = (pred_x2   - pred_x1).clamp(min=0) * (pred_y2   - pred_y1).clamp(min=0)
        target_area = (tgt_x2    - tgt_x1).clamp(min=0)  * (tgt_y2    - tgt_y1).clamp(min=0)
        union_area  = pred_area + target_area - inter_area

        # ── IoU and loss ─────────────────────────────────────────────────────
        iou  = inter_area / (union_area + self.eps)
        loss = 1.0 - iou   # shape (N,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# ─── sanity checks ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    criterion = IoULoss(reduction="mean")

    # ── test 1: perfect overlap → loss should be 0 ───────────────────────
    box = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
    loss = criterion(box, box)
    print(f"perfect overlap loss (expect ~0): {loss.item():.6f}")
    assert loss.item() < 1e-4

    # ── test 2: no overlap → loss should be 1 ────────────────────────────
    pred   = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
    target = torch.tensor([[0.9, 0.9, 0.1, 0.1]])
    loss   = criterion(pred, target)
    print(f"no overlap loss (expect 1.0):     {loss.item():.6f}")
    assert abs(loss.item() - 1.0) < 1e-4

    # ── test 3: gradients flow through ───────────────────────────────────
    pred_g = torch.tensor([[0.5, 0.5, 0.3, 0.3]], requires_grad=True)
    tgt_g  = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
    loss_g = criterion(pred_g, tgt_g)
    loss_g.backward()
    print(f"gradient on pred: {pred_g.grad}")
    assert pred_g.grad is not None, "no gradient!"

    print("\nAll IoU loss checks passed.")