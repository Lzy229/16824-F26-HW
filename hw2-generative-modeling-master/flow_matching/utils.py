
import math
import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    """
    Standard sinusoidal time embedding used to condition the UNet on t
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        # t: shape [B] assumed to be in [0,1] or discrete steps
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

# ---------- Path schedule / interpolant helpers for flow matching training ----------

def linear_interpolant(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
    """
    Linear path from base x1 (Gaussian) to data x0:
      x_t = (1 - t) * x1 + t * x0
    t is broadcast over x0/x1 shape.
    """
    while t.dim() < x0.dim():
        t = t[..., None]
    return (1. - t) * x1 + t * x0

def linear_velocity(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
    """
    Target velocity for the linear path above:
      dx_t/dt = x0 - x1   (independent of t)
    """
    return x0 - x1

def sample_base_like(x: torch.Tensor):
    """
    Sample base distribution N(0, I) with same shape as x
    """
    return torch.randn_like(x)


def normalize_to_minus1_to_1(x):
    # expects x in [0,1]; returns [-1,1]
    return x * 2.0 - 1.0

def unnormalize_to_zero_to_one(x):
    # expects x in [-1,1]; returns [0,1]
    return (x + 1.0) * 0.5