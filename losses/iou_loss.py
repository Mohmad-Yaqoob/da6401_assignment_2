import torch
import torch.nn as nn


class IoULoss(nn.Module):
    # IoU-based regression loss for axis-aligned bounding boxes
    # input format: [cx, cy, w, h] in pixel space
    # output range: [0, 1] — 0 = perfect overlap, 1 = no overlap

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "sum", "none"), \
            f"reduction must be mean/sum/none, got '{reduction}'"
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        # unpack cx,cy,w,h -> corners
        px1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        py1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        px2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        py2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

        tx1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
        ty1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
        tx2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
        ty2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

        inter_w = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(0)
        inter_h = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(0)
        inter   = inter_w * inter_h

        area_p = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
        area_t = (tx2 - tx1).clamp(0) * (ty2 - ty1).clamp(0)
        union  = area_p + area_t - inter

        iou  = inter / (union + self.eps)
        loss = 1.0 - iou

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss