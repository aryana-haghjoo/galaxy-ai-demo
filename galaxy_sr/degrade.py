"""Simulating a smaller, cheaper telescope.

To teach a model to sharpen images, we need pairs: a blurry image and the
sharp truth. We make them by taking a good image and *ruining* it in the
same ways the real world ruins an image:

    1. blur   -- the atmosphere and the optics smear every point of light
                 (astronomers call this the "point spread function")
    2. shrink -- a smaller detector collects fewer pixels
    3. noise  -- photons arrive at random, and the electronics add hiss

The model never sees the sharp version until it is graded on its answer.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def gaussian_kernel(sigma: float, device=None) -> torch.Tensor:
    radius = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur. img: (B, C, H, W)."""
    if sigma <= 0:
        return img
    k = gaussian_kernel(sigma, img.device)
    c = img.shape[1]
    r = (k.numel() - 1) // 2
    kx = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    img = F.conv2d(F.pad(img, (r, r, 0, 0), mode="reflect"), kx, groups=c)
    img = F.conv2d(F.pad(img, (0, 0, r, r), mode="reflect"), ky, groups=c)
    return img


def degrade(hr: torch.Tensor, factor: int = 4, sigma: float = 1.6,
            noise: float = 0.012, generator: torch.Generator | None = None
            ) -> torch.Tensor:
    """Sharp image in, small blurry noisy image out. hr: (B, C, H, W)."""
    lr = blur(hr, sigma)
    lr = F.interpolate(lr, scale_factor=1 / factor, mode="area")
    if noise > 0:
        n = torch.randn(lr.shape, device=lr.device, dtype=lr.dtype,
                        generator=generator)
        lr = lr + noise * n
    return lr.clamp(0, 1)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Peak signal-to-noise ratio in dB. Higher = closer to the truth."""
    mse = F.mse_loss(a.clamp(0, 1), b.clamp(0, 1)).item()
    return 99.0 if mse <= 1e-12 else 10 * math.log10(1.0 / mse)
