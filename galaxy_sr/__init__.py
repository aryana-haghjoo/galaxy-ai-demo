"""A small, honest AI demo: teaching a neural network to sharpen galaxy images.

Built for a live classroom demonstration. Two steps:
    1. `01_train.ipynb`  -- run once ahead of time on a GPU
    2. `02_demo.ipynb`   -- run live; loads the saved weights, no training,
                            no internet
"""

from . import data, degrade, model, viz  # noqa: F401

__all__ = ["data", "degrade", "model", "viz", "train"]
__version__ = "1.0"
