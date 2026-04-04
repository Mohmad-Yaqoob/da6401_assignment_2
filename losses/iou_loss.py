import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """
    Custom IoU Loss for bounding box regression.

    Inputs: [x_center, y_center, width, height] in pixel space
    Loss range: [0, 1]  — 0 means perfect overlap, 1 means no overlap

    Supported reductions: "mean" (default), "sum", "none"
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-6):
        super().__init__()
        assert reduction in ("mean", "sum", "none"), \
            f"reduction must be mean/sum/none, got '{reduction}'"
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   : (N, 4) predicted   [cx, cy, w, h] pixel space
            target : (N, 4) ground truth [cx, cy, w, h] pixel space
        Returns:
            scalar if reduction mean/sum, else (N,) tensor
        """
        # [cx,cy,w,h] -> [x1,y1,x2,y2]
        pred_x1 = pred[:, 0] - pred[:, 2] / 2
        pred_y1 = pred[:, 1] - pred[:, 3] / 2
        pred_x2 = pred[:, 0] + pred[:, 2] / 2
        pred_y2 = pred[:, 1] + pred[:, 3] / 2

        tgt_x1  = target[:, 0] - target[:, 2] / 2
        tgt_y1  = target[:, 1] - target[:, 3] / 2
        tgt_x2  = target[:, 0] + target[:, 2] / 2
        tgt_y2  = target[:, 1] + target[:, 3] / 2

        # intersection
        inter_w    = (torch.min(pred_x2, tgt_x2) - torch.max(pred_x1, tgt_x1)).clamp(min=0)
        inter_h    = (torch.min(pred_y2, tgt_y2) - torch.max(pred_y1, tgt_y1)).clamp(min=0)
        inter_area = inter_w * inter_h

        # areas
        pred_area   = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
        target_area = (tgt_x2  - tgt_x1).clamp(min=0)  * (tgt_y2  - tgt_y1).clamp(min=0)
        union_area  = pred_area + target_area - inter_area

        iou  = inter_area / (union_area + self.eps)   # [0, 1]
        loss = 1.0 - iou                               # [0, 1]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def extra_repr(self):
        return f"reduction={self.reduction}"