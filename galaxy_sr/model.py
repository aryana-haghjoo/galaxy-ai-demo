"""The network itself.

This is a small residual CNN in the EDSR family -- the same idea behind the
"enhance" features in phone cameras, streaming upscalers, and the tools
astronomers use on survey data. It is deliberately small (~1.5M numbers) so
it trains in minutes and runs instantly in front of an audience.

How it works, in one breath: look at the small image through a stack of
learned filters, then rearrange the resulting channels into a bigger grid of
pixels (that last trick is called "pixel shuffle").
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, ch: int, scale: float = 0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
        )
        self.scale = scale

    def forward(self, x):
        return x + self.scale * self.body(x)


class GalaxySR(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 12, factor: int = 4):
        super().__init__()
        self.factor = factor
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.body = nn.Sequential(
            *[ResBlock(channels) for _ in range(blocks)],
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        up = []
        for _ in range(int(torch.log2(torch.tensor(float(factor))).item())):
            up += [nn.Conv2d(channels, channels * 4, 3, padding=1),
                   nn.PixelShuffle(2), nn.ReLU(inplace=True)]
        self.up = nn.Sequential(*up)
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, lr):
        x = self.head(lr)
        x = x + self.body(x)
        x = self.up(x)
        # predict the *difference* from a plain enlargement -- easier to learn
        base = nn.functional.interpolate(
            lr, scale_factor=self.factor, mode="bicubic", align_corners=False)
        return (base + self.tail(x)).clamp(0, 1)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def load_trained(path: str = "weights/galaxy_sr.pt", device="cpu"):
    """Load a checkpoint saved by train.py. Returns (model, info dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = GalaxySR(cfg.get("channels", 64), cfg.get("blocks", 12),
                     cfg.get("factor", 4))
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    return model, ckpt.get("info", {})
