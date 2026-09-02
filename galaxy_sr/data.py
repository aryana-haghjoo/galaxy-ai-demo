"""Getting galaxy images to train on.

Two sources, tried in order:

1. Real galaxies from SDSS (the Sloan Digital Sky Survey). This is a public
   archive of ~1 million galaxy images. No login, no API key.
2. If there is no internet, simulated galaxies built from the same math
   astronomers use to describe real ones (Sersic profiles + spiral arms).

Everything is cached to disk as PNGs, so the live demo never touches the
network.
"""

from __future__ import annotations

import io
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

SDSS_BASE = "https://skyserver.sdss.org/dr17/SkyServerWS"
CUTOUT_URL = SDSS_BASE + "/ImgCutout/getjpeg"
SQL_URL = SDSS_BASE + "/SearchTools/SqlSearch"

# A few famous galaxies, for the "hero" images in the live demo.
# (name, RA degrees, Dec degrees, arcsec-per-pixel for the cutout)
FAMOUS = [
    ("M51_Whirlpool", 202.4696, 47.1952, 1.2),
    ("M101_Pinwheel", 210.8025, 54.3489, 1.6),
    ("M81_Bodes", 148.8882, 69.0653, 1.6),
    ("M63_Sunflower", 198.9554, 42.0293, 1.0),
    ("M64_BlackEye", 194.1821, 21.6829, 0.8),
    ("M106", 184.7396, 47.3040, 1.4),
    ("M94", 192.7212, 41.1206, 0.8),
    ("M100", 185.7288, 15.8225, 0.8),
    ("M99", 184.7067, 14.4164, 0.7),
    ("M88", 187.9966, 14.4204, 0.7),
    ("M87_Virgo_giant", 187.7059, 12.3911, 0.8),
    ("M74_Phantom", 24.1741, 15.7833, 1.0),
    ("M77", 40.6696, -0.0133, 0.7),
    ("M66", 170.0625, 12.9917, 0.8),
    ("M65", 169.7329, 13.0922, 0.8),
    ("NGC4565_Needle", 189.0865, 25.9877, 1.2),
    ("NGC5907_Splinter", 228.9742, 56.3289, 1.2),
    ("M108", 167.8790, 55.6741, 0.8),
    ("M109", 179.4000, 53.3745, 0.8),
    ("M84", 186.2656, 12.8870, 0.6),
]


# --------------------------------------------------------------------------
# real data
# --------------------------------------------------------------------------
def _get(url: str, timeout: int = 20, tries: int = 3) -> bytes:
    """Fetch a URL, retrying a couple of times.

    The SDSS servers are public and free, which also means they are busy: a
    request times out every fifth try or so. Retrying is not optional.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "galaxy-sr-demo/1.0"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(1.0 + attempt)
    raise RuntimeError("unreachable")


def have_internet(timeout: int = 10) -> bool:
    try:
        _get(SDSS_BASE + "/ImgCutout/getjpeg?ra=202.4696&dec=47.1952"
             "&scale=1.0&width=64&height=64", timeout=timeout, tries=2)
        return True
    except Exception:
        return False


def sdss_cutout(ra: float, dec: float, scale: float = 0.4,
                size: int = 256) -> Image.Image:
    """One colour JPEG cutout of the sky, centred on (ra, dec).

    `scale` is arcseconds per pixel -- bigger scale = wider field of view.
    """
    q = urllib.parse.urlencode(
        {"ra": ra, "dec": dec, "scale": scale, "width": size, "height": size}
    )
    return Image.open(io.BytesIO(_get(f"{CUTOUT_URL}?{q}"))).convert("RGB")


def sdss_galaxy_list(n: int = 400) -> list[tuple[float, float]]:
    """Ask SDSS for coordinates of n reasonably big, bright galaxies."""
    sql = (
        f"SELECT TOP {n} ra, dec FROM Galaxy "
        "WHERE petroR90_r BETWEEN 8 AND 30 AND r BETWEEN 13 AND 16.5 "
        "AND clean = 1"
    )
    q = urllib.parse.urlencode({"cmd": sql, "format": "csv"})
    text = _get(f"{SQL_URL}?{q}", timeout=60, tries=2).decode("utf-8", "replace")
    out = []
    for line in text.splitlines():
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            out.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue  # header / comment lines
    return out


# --------------------------------------------------------------------------
# simulated data (the offline fallback)
# --------------------------------------------------------------------------
def synthetic_galaxy(size: int = 256, rng: np.random.Generator | None = None,
                     force_spiral: bool = False) -> Image.Image:
    """A fake but physically-motivated galaxy: Sersic bulge + disc + arms."""
    rng = rng or np.random.default_rng()
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy = size / 2 + rng.normal(0, size * 0.03, 2)
    x, y = x - cx, y - cy

    # random orientation and inclination
    th = rng.uniform(0, np.pi)
    q = rng.uniform(0.7, 1.0) if force_spiral else rng.uniform(0.3, 1.0)
    xr = x * np.cos(th) + y * np.sin(th)
    yr = (-x * np.sin(th) + y * np.cos(th)) / q
    r = np.sqrt(xr ** 2 + yr ** 2) + 1e-3
    phi = np.arctan2(yr, xr)

    # Sersic bulge (n=4) + exponential disc (n=1)
    re_b = size * rng.uniform(0.02, 0.05)
    re_d = size * rng.uniform(0.10, 0.22)
    bulge = np.exp(-7.669 * ((r / re_b) ** 0.25 - 1))
    disc = np.exp(-1.678 * (r / re_d))
    img = rng.uniform(0.2, 1.2) * bulge + disc

    # spiral arms, sometimes
    if force_spiral or rng.random() < 0.65:
        n_arms = rng.choice([2, 2, 2, 3, 4])
        pitch = rng.uniform(0.15, 0.45)
        arms = np.cos(n_arms * (phi - np.log(r) / pitch))
        img *= 1 + rng.uniform(0.3, 0.9) * arms * np.exp(-r / re_d)

    # dust lanes cutting across the disc -- sharp, dark, hard to fake
    gy, gx = np.mgrid[0:size, 0:size].astype(np.float32)
    for _ in range(rng.integers(0, 3)):
        lw = rng.uniform(0.8, 2.2)
        off = rng.normal(0, re_d * 0.4)
        d = np.abs(yr - off)
        img *= 1 - rng.uniform(0.2, 0.6) * np.exp(-(d ** 2) / (2 * lw ** 2))

    # knots of star formation -- small and bright, i.e. real fine detail
    for _ in range(rng.integers(20, 70)):
        rr = abs(rng.normal(0, re_d))
        pp = rng.uniform(0, 2 * np.pi)
        kx, ky = cx + rr * np.cos(pp), cy + rr * np.sin(pp)
        s = rng.uniform(0.5, 1.3)
        img += rng.uniform(0.05, 0.30) * np.exp(
            -((gx - kx) ** 2 + (gy - ky) ** 2) / (2 * s ** 2))

    # foreground stars
    for _ in range(rng.integers(2, 12)):
        sx, sy = rng.uniform(0, size, 2)
        s = rng.uniform(0.6, 1.4)
        img += rng.uniform(0.2, 1.0) * np.exp(
            -((gx - sx) ** 2 + (gy - sy) ** 2) / (2 * s ** 2))

    # faint background galaxies
    for _ in range(rng.integers(0, 6)):
        sx, sy = rng.uniform(0, size, 2)
        s = rng.uniform(1.5, 4.0)
        img += rng.uniform(0.03, 0.12) * np.exp(
            -((gx - sx) ** 2 + (gy - sy) ** 2) / (2 * s ** 2))

    img = np.arcsinh(img * rng.uniform(4, 20)) / 4.0
    img = np.clip(img / (img.max() + 1e-6), 0, 1)

    # give it a colour: redder core, bluer arms -- like a real galaxy
    warm = np.clip(img ** rng.uniform(0.8, 1.0), 0, 1)
    cool = np.clip(img ** rng.uniform(1.0, 1.4), 0, 1)
    rgb = np.stack([warm, img, cool], -1)
    rgb += rng.normal(0, 0.01, rgb.shape)
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))


# --------------------------------------------------------------------------
# the one function the notebooks call
# --------------------------------------------------------------------------
def build_dataset(root: str | Path = "data", n_train: int = 300,
                  size: int = 256, verbose: bool = True, workers: int = 8,
                  budget_min: float = 8.0) -> dict:
    """Fill `root/train` and `root/demo` with galaxy PNGs. Safe to re-run.

    Downloads run in parallel (`workers` at a time) and give up after
    `budget_min` minutes, whatever they have managed to get. Anything still
    missing is topped up with simulated galaxies, so this function always
    returns a usable dataset -- slow server, flaky wifi, or no wifi at all.
    """
    root = Path(root)
    train_dir, demo_dir = root / "train", root / "demo"
    train_dir.mkdir(parents=True, exist_ok=True)
    demo_dir.mkdir(parents=True, exist_ok=True)

    n_have = len(list(train_dir.glob("*.png")))
    n_demo = len(list(demo_dir.glob("*.png")))
    if n_have >= n_train and n_demo > 0:
        if verbose:
            print(f"Already cached: {n_have} training + {n_demo} demo images.")
        return {"source": "cache", "n_train": n_have, "n_demo": n_demo}

    online = have_internet()
    if verbose:
        print("SDSS reachable." if online else
              "No connection to SDSS -- using simulated galaxies instead.")

    if online:
        deadline = time.time() + budget_min * 60

        def fetch(job) -> bool:
            """Download one cutout to disk. Returns True if the file is there."""
            path, ra, dec, scale, px = job
            if path.exists():
                return True
            if time.time() > deadline:
                return False
            try:
                sdss_cutout(ra, dec, scale, px).save(path)
                return True
            except Exception:
                return False

        def fetch_all(jobs, label) -> int:
            got = 0
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for ok in pool.map(fetch, jobs):
                    got += ok
                    if verbose and got and got % 25 == 0:
                        print(f"  {label}: {got}/{len(jobs)} "
                              f"({(deadline - time.time()) / 60:.1f} min left)")
            return got

        # The hero images for the live demo go first and are never skipped --
        # they are the ones the room will actually look at.
        n_demo = fetch_all([(demo_dir / f"{name}.png", ra, dec, scale, 512)
                            for name, ra, dec, scale in FAMOUS], "demo images")

        # Then the training set: a few hundred ordinary galaxies.
        try:
            coords = sdss_galaxy_list(n_train * 2)
        except Exception as e:
            print(f"  galaxy list failed ({e}); falling back to famous list")
            coords = [(ra, dec) for _, ra, dec, _ in FAMOUS]
        jobs = [(train_dir / f"sdss_{ra:.5f}_{dec:+.5f}.png", ra, dec, 0.4, size)
                for ra, dec in coords][:max(0, n_train - n_have)]
        n_got = fetch_all(jobs, "training images")

        if verbose:
            print(f"  got {n_demo}/{len(FAMOUS)} demo and {n_got}/{len(jobs)} "
                  "training images from SDSS.")
        source = "SDSS"
    else:
        source = "simulated"

    # top up with simulated galaxies if the download was short or offline
    rng = np.random.default_rng(0)
    while len(list(train_dir.glob("*.png"))) < n_train:
        i = len(list(train_dir.glob("*.png")))
        synthetic_galaxy(size, rng).save(train_dir / f"sim_{i:04d}.png")
    if not list(demo_dir.glob("*.png")):
        for i in range(6):
            synthetic_galaxy(512, rng, force_spiral=True).save(
                demo_dir / f"sim_demo_{i}.png")

    n_train_final = len(list(train_dir.glob("*.png")))
    n_demo_final = len(list(demo_dir.glob("*.png")))
    if verbose:
        print(f"Ready: {n_train_final} training images, "
              f"{n_demo_final} demo images (source: {source}).")
    return {"source": source, "n_train": n_train_final, "n_demo": n_demo_final}


def load_images(folder: str | Path) -> list[np.ndarray]:
    """Every PNG in a folder, as float arrays in [0, 1] shaped (H, W, 3)."""
    files = sorted(Path(folder).glob("*.png"))
    return [np.asarray(Image.open(f).convert("RGB"), np.float32) / 255.0
            for f in files]
