# Galaxy super-resolution — a live AI demo for teachers

A small neural network learns to sharpen blurred galaxy images. Built to be
run in front of a room of high school faculty in about 10–15 minutes, with
**no training and no internet during the demo**.

- `01_train.ipynb` — run once, ahead of time, on the GPU box (~15 min)
- `02_demo.ipynb` — the live notebook (loads saved weights, runs instantly)
- `demo_script.md` — what to say, beat by beat, with timings
- `galaxy_sr/` — the actual code (~700 lines, heavily commented for a non-expert reader)

---

## Setup on the GPU box

```bash
cd galaxy-ai-demo
pip install -r requirements.txt        # torch, numpy, pillow, matplotlib
jupyter lab                            # or just open the folder in VS Code
```

Then run `01_train.ipynb` top to bottom. It will:

1. download ~300 real galaxy cutouts from the SDSS public archive into `data/`
2. train the model for a few minutes
3. save `weights/galaxy_sr.pt`
4. print a before/after so you can confirm it worked

**Do this at least a day early**, and then run `02_demo.ipynb` once yourself.
On demo day you are re-running something you have already watched work.

### If the box has no internet

Everything still runs — the data step falls back to simulated galaxies built
from Sérsic profiles, spiral arms, dust lanes and star knots. The result is
less pretty but the demo is identical. You will just want to say "these are
simulated" once, which is a fine thing to say in front of teachers anyway.

---

## On demo day

Open `02_demo.ipynb`, **Restart Kernel**, then run cells one at a time as you
talk. Timings and script are in `demo_script.md`.

Two cells are meant to be edited live:

- `NAME = list(galaxies)[0]` — which galaxy you open with
- `PICK = list(galaxies)[1]` — the one the audience calls out
- `X, Y, SIZE = 190, 190, 130` — where the zoom box sits

Nothing else needs touching.

### If something goes wrong live

Every figure is also saved into `figures/` when you run it. Run the notebook
once the night before, and if the kernel dies in the room, open the PNGs in
`figures/` and keep talking. The story survives without the live run.

---

## Knobs worth knowing about

In `01_train.ipynb`:

| Setting | What it does |
|---|---|
| `steps` | how long it trains. 2000 is fine, 6000 is better |
| `factor` | how much the image shrinks. 4 = throws away 15 of every 16 pixels |
| `sigma` | how badly the simulated small telescope blurs |
| `noise` | how noisy its detector is |
| `blocks`, `channels` | model size. Bigger = slower but sharper |

Turning `sigma` and `noise` **up** makes the demo more dramatic (the AI has
more to fix) but also makes the AI's mistakes more visible — which, for this
particular audience, is arguably the better demo.

---

## What the code does, in one paragraph

Take a good galaxy image. Blur it, shrink it 4×, and add noise — that is a
physically honest simulation of a smaller telescope. Now show a small
residual CNN the damaged version and grade it against the original. Repeat a
few thousand times. The network never memorises galaxies; it learns what
galaxy-shaped detail generally looks like, and uses that to make an educated
guess about what the missing pixels were. That is exactly why it sometimes
invents things — which is the most useful thing in the demo to show teachers.

Data: [SDSS](https://www.sdss.org/) DR17 image cutout service, public, no key.
