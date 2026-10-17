"""Shared matplotlib styling and save helper for the P5 case figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import fontManager  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = _ROOT / "figures"
DATA_DIR = _ROOT / "data"

_CJK = [f.name for f in fontManager.ttflist if f.name in
        ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC")]

if _CJK:
    plt.rcParams["font.family"] = [_CJK[0], "DejaVu Sans"]
plt.rcParams.update({
    "axes.grid": True,
    "grid.alpha": 0.35,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "axes.prop_cycle": plt.cycler(color=["#1f77b4", "#d62728", "#2ca02c",
                                        "#9467bd", "#8c564b", "#17becf"]),
})


def figure(nrows=1, ncols=1, figsize=None, **kwargs):
    fig, grid = plt.subplots(nrows, ncols,
                             figsize=figsize or (6.4 * ncols, 4.4 * nrows),
                             squeeze=False, **kwargs)
    ax = grid.ravel()
    return fig, (ax[0] if ax.size == 1 else ax)


def save(fig, name: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path


def write_csv(rows, name: str, header: list[str]) -> Path:
    """Write an iterable of sequences as CSV, one row per solver point."""
    import csv

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / name
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path
