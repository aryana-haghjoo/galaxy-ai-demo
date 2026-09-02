"""Training loop. Run this once, ahead of the demo, on the GPU box.

Weights & Biases logging is optional and off by default, so that anyone can
clone this repo and train without an account. Pass `wandb_project="..."` to
turn it on; the run then records the loss curve, the held-out scores, a
before/after figure, and the checkpoint itself as an artifact.
"""

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
        self._cache = None

    def __len__(self):
        return self.length

    def _rng(self) -> np.random.Generator:
        # Each DataLoader worker is a separate process. Without this, every
        # worker would inherit the same random state and cut the *same*
        # crops, quietly halving the variety the model ever sees.
        info = torch.utils.data.get_worker_info()
        seed = 0 if info is None else info.seed
        if self._cache is None or self._cache[0] != seed:
            self._cache = (seed, np.random.default_rng(seed))
        return self._cache[1]

    def __getitem__(self, idx):
        rng = self._rng()
        im = self.images[rng.integers(len(self.images))]
        p = self.patch
        y = rng.integers(im.shape[0] - p + 1)
        x = rng.integers(im.shape[1] - p + 1)
        crop = im[y:y + p, x:x + p]
        if rng.random() < 0.5:
            crop = crop[:, ::-1]
        if rng.random() < 0.5:
            crop = crop[::-1]
        k = rng.integers(4)
        if k:
            crop = np.rot90(crop, k)
        return torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1)


def train(data_dir: str = "data/train", out: str = "weights/galaxy_sr.pt",
          steps: int = 3000, batch: int = 16, patch: int = 128,
          channels: int = 64, blocks: int = 12, factor: int = 4,
          sigma: float = 1.6, noise: float = 0.012, lr: float = 2e-4,
          device: str | None = None, log_every: int = 100,
          wandb_project: str | None = None, wandb_run: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = {"channels": channels, "blocks": blocks, "factor": factor,
           "sigma": sigma, "noise": noise, "patch": patch, "steps": steps,
           "batch": batch, "lr": lr, "data_dir": data_dir}
    run = _wandb_start(wandb_project, wandb_run, cfg)
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
            if run:
                run.log({"loss": loss.item(),
                         "lr": sched.get_last_lr()[0]}, step=step)

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
        "hardware": (torch.cuda.get_device_name(0) if device == "cuda"
                     else "CPU"),
        "n_params": model.n_params,
        "history": history,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg, "info": info},
               out)

    print(f"\nsaved {out}")
    print(f"  held-out PSNR  ->  AI {info['psnr_model']:.2f} dB   vs   "
          f"classic {info['psnr_bicubic']:.2f} dB "
          f"(+{info['psnr_model'] - info['psnr_bicubic']:.2f})")
    print(f"  trained in {info['minutes']:.1f} minutes on {device}")

    if run:
        _wandb_finish(run, model, info, cfg, out, val_images, factor,
                      sigma, noise, device)
    return info


# --------------------------------------------------------------------------
# optional Weights & Biases logging
# --------------------------------------------------------------------------
def _wandb_start(project: str | None, name: str | None, cfg: dict):
    if not project:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb not installed -- skipping experiment tracking "
              "(pip install wandb)")
        return None
    return wandb.init(project=project, name=name, config=cfg)


def _wandb_finish(run, model, info, cfg, out, val_images, factor, sigma,
                  noise, device):
    """Log the held-out scores, a couple of plots, and the checkpoint."""
    import wandb

    from . import viz

    run.summary.update({k: v for k, v in info.items() if k != "history"})

    figs = {"scoreboard": viz.scoreboard(info),
            "learning_curve": viz.learning_curve(info)}
    if val_images:
        res = viz.run_all(model, val_images[0], factor, sigma, noise)
        figs["validation_example"] = viz.compare(res)
    run.log({k: wandb.Image(f) for k, f in figs.items() if f is not None})

    # the checkpoint plus what is needed to reproduce it
    art = wandb.Artifact("galaxy_sr", type="model", metadata={
        **cfg,
        "psnr_model": info["psnr_model"],
        "psnr_bicubic": info["psnr_bicubic"],
        "n_train_images": info["n_train_images"],
        "n_val_images": info["n_val_images"],
        "device": device,
        "git_commit": _git_commit(),
        "torch": torch.__version__,
    })
    art.add_file(str(out))
    run.log_artifact(art)
    run.finish()


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"
