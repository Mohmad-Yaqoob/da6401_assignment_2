import torch
import torch.nn as nn


# IoU-based regression loss for bounding boxes in cx, cy, w, h format.

# The loss is defined as 1 - IoU, so it always stays between 0 and 1.

# Supports different reduction types: mean (default), sum, or none.
class IoULoss(nn.Module):

    _VALID = {"mean", "sum", "none"}

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        if reduction not in self._VALID:
            raise ValueError(f"reduction must be one of {self._VALID}, got '{reduction}'")
        self.eps       = eps
        self.reduction = reduction

    @staticmethod
    def _to_xyxy(b: torch.Tensor) -> torch.Tensor:
        cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=1)

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        p = self._to_xyxy(pred_boxes)
        t = self._to_xyxy(target_boxes)

        ix1 = torch.max(p[:, 0], t[:, 0]);  iy1 = torch.max(p[:, 1], t[:, 1])
        ix2 = torch.min(p[:, 2], t[:, 2]);  iy2 = torch.min(p[:, 3], t[:, 3])

        inter_w = (ix2 - ix1).clamp(min=0)
        inter_h = (iy2 - iy1).clamp(min=0)
        inter   = inter_w * inter_h

        pa   = (p[:, 2] - p[:, 0]).clamp(0) * (p[:, 3] - p[:, 1]).clamp(0)
        ta   = (t[:, 2] - t[:, 0]).clamp(0) * (t[:, 3] - t[:, 1]).clamp(0)
        union = pa + ta - inter + self.eps

        loss = 1.0 - inter / union

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def extra_repr(self) -> str:
        return f"eps={self.eps}, reduction='{self.reduction}'"