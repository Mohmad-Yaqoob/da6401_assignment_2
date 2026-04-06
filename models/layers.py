import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    # hand-rolled bernoulli mask with inverted scaling
    # no nn.Dropout or F.dropout used anywhere

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"p must be in [0,1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = torch.bernoulli(
            torch.full(x.shape, keep, device=x.device, dtype=x.dtype)
        )
        return x * mask / keep

    def extra_repr(self):
        return f"p={self.p}"