"""Big, projector-friendly figures for the live demo."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from .degrade import degrade

INK = "#1a1a1a"
MUTED = "#6b6b6b"
AI = "#2f6fb5"       # the model

plt.rcParams.update({
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 13,
    "axes.edgecolor": "#d6d6d6",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
})


def _save(fig, path):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")


# --------------------------------------------------------------------------
def to_tensor(img: np.ndarray, factor: int = 4) -> torch.Tensor:
    """(H, W, 3) float array -> (1, 3, H, W) tensor, cropped to fit."""
    h = (img.shape[0] // factor) * factor
    w = (img.shape[1] // factor) * factor
    return torch.from_numpy(np.ascontiguousarray(img[:h, :w])).permute(2, 0, 1)[None]


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()


@torch.no_grad()
def sharpen(model, low_res: torch.Tensor, device=None) -> torch.Tensor:
    device = device or next(model.parameters()).device
    return model(low_res.to(device)).cpu()


@torch.no_grad()
def time_sharpen(model, low_res: torch.Tensor, repeats: int = 5) -> float:
    """Seconds the network itself takes on one image, measured honestly.

    The GPU runs asynchronously, so we have to wait for it before stopping the
    clock -- otherwise you time how fast Python can queue work, not how fast
    the model is.
    """
    device = next(model.parameters()).device
    lo = low_res.to(device)
    model(lo)                                    # warm-up, not counted
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        model(lo)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


@torch.no_grad()
def run_all(model, image: np.ndarray, factor: int | None = None,
            sigma: float | None = None, noise: float | None = None,
            seed: int = 0) -> dict:
    """Truth -> simulated small telescope -> the model's reconstruction.

    The damage defaults to whatever this model was trained to undo, which is
    recorded in the checkpoint. Override it only to show what happens when
    the test does not match the training.
    """
    cfg = getattr(model, "cfg", {})
    factor = factor if factor is not None else cfg.get("factor", 4)
    sigma = sigma if sigma is not None else cfg.get("sigma", 1.6)
    noise = noise if noise is not None else cfg.get("noise", 0.012)
    hr = to_tensor(image, factor)
    g = torch.Generator().manual_seed(seed)
    lo = degrade(hr, factor, sigma, noise, generator=g)
    return {
        "truth": hr,
        "low": lo,
        "ai": sharpen(model, lo),
        "factor": factor,
    }


# --------------------------------------------------------------------------
def _imshow(ax, t: torch.Tensor, title: str, sub: str = "", nearest=False):
    ax.imshow(to_numpy(t), interpolation="nearest" if nearest else "antialiased")
    ax.set_title(title, fontsize=16, pad=10, color=INK)
    if sub:
        ax.set_xlabel(sub, fontsize=12, color=MUTED, labelpad=8)
    ax.set_xticks([])
    ax.set_yticks([])


def compare(res: dict, zoom: tuple | None = None, save: str | None = None,
            show_truth: bool = True):
    """The main money shot: what the telescope saw vs. what the AI recovered."""
    panels = [
        ("What the small telescope sees", res["low"],
         f"{res['low'].shape[-1]} x {res['low'].shape[-2]} pixels", True),
        ("What the AI reconstructs", res["ai"], "trained on other galaxies", False),
    ]
    if show_truth:
        panels.append(("The real thing", res["truth"], "the original photograph", False))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5.9))
    for ax, (title, t, sub, near) in zip(np.atleast_1d(axes), panels):
        _imshow(ax, t, title, sub, nearest=near)
        if zoom and t.shape[-1] == res["truth"].shape[-1]:
            x, y, s = zoom
            ax.add_patch(Rectangle((x, y), s, s, fill=False, lw=2.0,
                                   edgecolor="#e8b84b"))
    fig.tight_layout()
    # Deliberately no score under this figure. It is the "look at it" beat;
    # the honest measurement gets its own slide later, on 30 held-out
    # galaxies rather than the one that was picked to be shown. A number
    # here just invites the room to argue with it instead of looking.
    _save(fig, save)
    return fig


def zoom_in(res: dict, x: int, y: int, size: int = 96, save: str | None = None):
    """Crop the same square out of every version and blow it up."""
    f = res["factor"]
    crops = [
        ("Small telescope", res["low"][..., y // f:(y + size) // f,
                                       x // f:(x + size) // f], True),
        ("AI", res["ai"][..., y:y + size, x:x + size], False),
        ("Truth", res["truth"][..., y:y + size, x:x + size], False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    for ax, (title, t, near) in zip(axes, crops):
        _imshow(ax, t, title, nearest=near)
    fig.suptitle("Same patch of sky, zoomed in", fontsize=17, y=1.02)
    fig.tight_layout()
    _save(fig, save)
    return fig


def where_it_erred(res: dict, save: str | None = None):
    """The honesty slide: the answer, the truth, and the gap between them."""
    err = (res["ai"] - res["truth"]).abs().mean(1)[0].numpy()
    # clip the scale at the 99.5th percentile so the structure is visible
    vmax = float(np.percentile(err, 99.5))

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.2),
                             constrained_layout=True)
    _imshow(axes[0], res["ai"], "The AI's answer", "looks clean")
    _imshow(axes[1], res["truth"], "The real thing", "the original photograph")

    im = axes[2].imshow(err, cmap="magma", vmin=0, vmax=vmax)
    axes[2].set_title("Where the AI is wrong", fontsize=16, pad=10)
    axes[2].set_xticks([]); axes[2].set_yticks([])
    axes[2].set_xlabel(f"typically off by {255 * err.mean():.1f} shades "
                       "out of 255", fontsize=12, color=MUTED, labelpad=8)
    cb = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.02,
                      label="brighter = further from the truth")
    cb.set_ticks(cb.get_ticks())
    cb.set_ticklabels([f"{255 * t:.0f}" for t in cb.get_ticks()])
    _save(fig, save)
    return fig


def scoreboard(info: dict, save: str | None = None):
    """One bar per held-out galaxy: is it this good on all of them, or was
    the one I showed you a lucky pick?

    Plotted as *how wrong the picture is*, not in decibels -- a shorter bar
    is a better answer, which needs no explaining to anyone. Sorted worst to
    best so the eye lands on the worst case first, which is the honest place
    for it to land.
    """
    each = info.get("psnr_each")
    if not each:
        return _score_summary(info, save)

    # PSNR -> typical error per pixel, on the 0-255 scale everyone knows
    err = np.sort([255 * 10 ** (-p / 20) for p in each])[::-1]
    mean_err = 255 * 10 ** (-info["psnr_model"] / 20)

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.bar(np.arange(len(err)), err, color=AI, width=0.72)
    ax.axhline(mean_err, color=INK, lw=1.4, ls="--")
    # sits at the right-hand end, where the bars are shortest and the space
    # above the mean line is empty
    ax.text(len(err) - 0.6, mean_err + max(err) * 0.03,
            f"typical: {mean_err:.1f}", ha="right", va="bottom",
            fontsize=13, color=INK)

    ax.set_ylim(0, max(err) * 1.3)
    ax.set_xlim(-1, len(err))
    ax.set_xticks([])
    ax.set_xlabel(f"each bar is one of {len(err)} galaxies the model never "
                  "trained on", labelpad=10)
    ax.set_ylabel("how far off the answer is\n"
                  "(shades of brightness, out of 255)")
    ax.set_title("Not a lucky picture: every unseen galaxy, scored",
                 fontsize=15, pad=14)
    ax.annotate("worst one", (0, err[0]), textcoords="offset points",
                xytext=(2, 8), fontsize=12, color=MUTED)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(axis="y", color="#eeeeee", lw=1)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, save)
    return fig


def _score_summary(info: dict, save: str | None = None):
    """Fallback for checkpoints trained before per-galaxy scores were kept."""
    mean_err = 255 * 10 ** (-info["psnr_model"] / 20)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.axis("off")
    ax.text(0.5, 0.62, f"{mean_err:.1f}", ha="center", fontsize=54, color=AI)
    ax.text(0.5, 0.34, "shades of brightness off the truth, out of 255\n"
            f"averaged over {info.get('n_val_images', '?')} galaxies "
            "the model never trained on",
            ha="center", fontsize=13, color=MUTED)
    fig.tight_layout()
    _save(fig, save)
    return fig


def learning_curve(info: dict, save: str | None = None):
    hist = info.get("history", [])
    if not hist:
        return None
    steps = [h[0] for h in hist]
    loss = [h[1] for h in hist]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(steps, loss, color=AI, lw=2)
    ax.set_xlabel("training steps")
    ax.set_ylabel("how wrong the model is")
    ax.set_title("The model teaching itself, one guess at a time",
                 fontsize=15, pad=12)
    ax.annotate(f"{loss[-1]:.3f}", (steps[-1], loss[-1]),
                textcoords="offset points", xytext=(6, 4), color=AI, fontsize=13)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color="#eeeeee", lw=1)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, save)
    return fig
