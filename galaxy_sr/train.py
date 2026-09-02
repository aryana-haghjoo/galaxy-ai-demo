"""Training loop. Run this once, ahead of the demo, on the GPU box."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .data import load_images
from .degrade import degrade, bicubic_up, psnr
from .model import GalaxySR


class PatchDataset(Dataset):
    """Random sharp crops of the galaxy images, with flips and rotations."""

    def __init__(self, images: list[np.ndarray], patch: int = 128,
                 length: int = 4000):
        self.images = [im for im in images
                       if im.shape[0] >= patch and im.shape[1] >= patch]
        if not self.images:
            raise ValueError("no images big enough for this patch size")
        self.patch, self.length = patch, length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng = np.random
        im = self.images[rng.randint(len(self.images))]
        p = self.patch
        y = rng.randint(im.shape[0] - p + 1)
        x = rng.randint(im.shape[1] - p + 1)
        crop = im[y:y + p, x:x + p]
        if rng.rand() < 0.5:
            crop = crop[:, ::-1]
        if rng.rand() < 0.5:
            crop = crop[::-1]
        k = rng.randint(4)
        if k:
            crop = np.rot90(crop, k)
        return torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1)


def train(data_dir: str = "data/train", out: str = "weights/galaxy_sr.pt",
          steps: int = 3000, batch: int = 16, patch: int = 128,
          channels: int = 64, blocks: int = 12, factor: int = 4,
          sigma: float = 1.6, noise: float = 0.012, lr: float = 2e-4,
          device: str | None = None, log_every: int = 100) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    images = load_images(data_dir)
    print(f"{len(images)} training images | device: {device}")

    # hold out a few images so the score we quote is on unseen data
    n_val = max(1, len(images) // 10)
    val_images, train_images = images[:n_val], images[n_val:] or images

    ds = PatchDataset(train_images, patch, steps * batch)
    dl = DataLoader(ds, batch_size=batch, num_workers=2, drop_last=True,
                    pin_memory=(device == "cuda"))

    model = GalaxySR(channels, blocks, factor).to(device)
    print(f"model has {model.n_params:,} trainable numbers")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    history, t0 = [], time.time()
    model.train()
    for step, hr in enumerate(dl, 1):
        hr = hr.to(device, non_blocking=True)
        with torch.no_grad():
            lo = degrade(hr, factor, sigma, noise)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            loss = F.l1_loss(model(lo), hr)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % log_every == 0 or step == 1:
            history.append((step, loss.item()))
            print(f"  step {step:5d}/{steps}  loss {loss.item():.4f}"
                  f"  ({time.time() - t0:.0f}s)")

    # score on held-out images: model vs. the classic non-AI method
    model.eval()
    p_model, p_bicubic = [], []
    with torch.no_grad():
        for im in val_images:
            h, w = (im.shape[0] // factor) * factor, (im.shape[1] // factor) * factor
            hr = torch.from_numpy(im[:h, :w]).permute(2, 0, 1)[None].to(device)
            lo = degrade(hr, factor, sigma, noise)
            p_model.append(psnr(model(lo), hr))
            p_bicubic.append(psnr(bicubic_up(lo, factor), hr))

    info = {
        "psnr_model": float(np.mean(p_model)),
        "psnr_bicubic": float(np.mean(p_bicubic)),
        "n_train_images": len(train_images),
        "n_val_images": len(val_images),
        "steps": steps,
        "minutes": (time.time() - t0) / 60,
        "device": device,
        "n_params": model.n_params,
        "history": history,
    }
    cfg = {"channels": channels, "blocks": blocks, "factor": factor,
           "sigma": sigma, "noise": noise, "patch": patch}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg, "info": info},
               out)

    print(f"\nsaved {out}")
    print(f"  held-out PSNR  ->  AI {info['psnr_model']:.2f} dB   vs   "
          f"classic {info['psnr_bicubic']:.2f} dB "
          f"(+{info['psnr_model'] - info['psnr_bicubic']:.2f})")
    print(f"  trained in {info['minutes']:.1f} minutes on {device}")
    return info
