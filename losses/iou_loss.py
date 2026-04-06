import torch
import torch.nn as nn


class IoULoss(nn.Module):
    # IoU loss for [cx, cy, w, h] boxes — works in any coordinate space
    # loss range [0,1]: 0 = perfect overlap, 1 = no overlap

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        px1 = pred[:,0] - pred[:,2]/2;   py1 = pred[:,1] - pred[:,3]/2
        px2 = pred[:,0] + pred[:,2]/2;   py2 = pred[:,1] + pred[:,3]/2
        tx1 = target[:,0] - target[:,2]/2; ty1 = target[:,1] - target[:,3]/2
        tx2 = target[:,0] + target[:,2]/2; ty2 = target[:,1] + target[:,3]/2

        inter_w = (torch.min(px2,tx2) - torch.max(px1,tx1)).clamp(0)
        inter_h = (torch.min(py2,ty2) - torch.max(py1,ty1)).clamp(0)
        inter   = inter_w * inter_h
        union   = (px2-px1).clamp(0)*(py2-py1).clamp(0) + \
                  (tx2-tx1).clamp(0)*(ty2-ty1).clamp(0) - inter
        loss = 1.0 - inter / (union + self.eps)

        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum":  return loss.sum()
        return loss