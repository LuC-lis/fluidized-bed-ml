"""Plotting helpers and a consistent house style for the whole project.

All labels are in English on purpose: the repository must render correctly on a
machine without CJK fonts (the original notebooks emitted a wall of
``Glyph 30495 missing from font(s)`` warnings for exactly this reason).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import paths  # noqa: E402

#: One entry per pressure transducer, ordered low -> high.
TAP_COLORS = ["#1b3a6b", "#2f6f9f", "#4fa3a5", "#8cbf6a", "#e0a33e", "#c0392b"]
PALETTE = ["#1b3a6b", "#c0392b", "#2e8b57", "#e0a33e", "#6a4c93", "#4fa3a5"]


def set_style() -> None:
    """Apply the project plotting style (idempotent)."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )


def save(fig, name: str, directory: Optional[Path] = None) -> Path:
    """Save ``fig`` as ``results/figures/<name>.png`` and close it."""
    paths.ensure_results_dirs()
    directory = paths.FIGURES_DIR if directory is None else Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / f"{name}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def write_table(df, name: str, directory: Optional[Path] = None) -> Path:
    """Save a DataFrame as ``results/tables/<name>.csv``."""
    paths.ensure_results_dirs()
    directory = paths.TABLES_DIR if directory is None else Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / f"{name}.csv"
    df.to_csv(out, index=False)
    return out


def annotate_panel(ax, letter: str) -> None:
    """Add an (a)/(b)/(c) panel letter in axes coordinates."""
    ax.text(-0.13, 1.04, f"({letter})", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom")


def bar_with_values(ax, x: Sequence[float], y: Sequence[float], **kwargs) -> None:
    """Simple bar helper that also labels the bars."""
    bars = ax.bar(x, y, **kwargs)
    for rect, value in zip(bars, y):
        ax.annotate(f"{value:.2f}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    ha="center", va="bottom", fontsize=8)
