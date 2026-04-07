import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    # standard inverted dropout but built from scratch
    # bernoulli mask sampled fresh every forward call during training
    # at eval time just passes input through unchanged

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"dropout p must be in [0,1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        # fresh bernoulli mask, same device/dtype as input
        mask = torch.zeros_like(x).bernoulli_(keep)
        # scale by 1/keep so expected value matches at test time
        return x * mask / keep

    def extra_repr(self) -> str:
        return f"p={self.p}"