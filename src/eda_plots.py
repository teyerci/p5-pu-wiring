"""Plot style and figure helpers for the EDA (static PNGs for the session).

Colours: categorical slots in fixed order, a one-hue sequential ramp for
magnitude, text in ink colours (never the series colour). Values from the
reference palette of the dataviz method, light mode.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#a3a29c"
GRID = "#e6e5e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

FIG_DIR = Path(__file__).resolve().parent.parent / "figures" / "eda"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 2,
        "legend.frameon": False, "legend.fontsize": 8, "font.family": "DejaVu Sans",
    })


def finish(fig, name: str, title: str, caption: str):
    """Title states the answer; caption states dataset, release, threshold and population."""
    width_in = fig.get_figwidth()
    title = textwrap.fill(title, width=int(width_in * 9.5))
    caption = textwrap.fill(caption, width=int(width_in * 16))
    t_lines, c_lines = title.count("\n") + 1, caption.count("\n") + 1
    h = fig.get_figheight()
    top = 1 - (0.12 + 0.22 * t_lines) / h
    bottom = (0.2 + 0.14 * c_lines) / h
    fig.suptitle(title, x=0.01, y=0.995, ha="left", va="top", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.01, 0.01, caption, ha="left", va="bottom", fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=(0, bottom, 1, top))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
