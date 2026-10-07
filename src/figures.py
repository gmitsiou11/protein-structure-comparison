"""
One visual system for the thesis figures (notebooks 03–05).

Colour by role: the two predictions get the two categorical colours (AF2 blue,
OF3 orange; their mutual comparison aqua); every reference level is grey, from
light (tightest: two copies in one entry) to dark (loosest: different
methods).  Text is never coloured by series.  The categorical colours are the
first three slots of a palette validated for colour-vision deficiency.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = "results/figures"

COLOR = {
    "AF2": "#2a78d6",
    "OF3": "#eb6834",
    "AF2 vs OF3": "#1baf7a",
    # reference levels: light grey = tightest, dark grey = loosest
    "same entry": "#b8b6af",
    "NMR ensemble": "#8a8984",
    "between methods": "#52514e",
    "experimental": "#52514e",
    # ink
    "text": "#0b0b0b",
    "muted": "#52514e",
    "rule": "#8a8984",
}

LABEL = {
    "w_rdist_norm": "w-rdist",
    "b_phipsi": "b-phipsi",
    "t_alpha": "t-alpha",
    "t_alpha_pca": "t-alpha (principal axes)",
    "rmsd": "RMSD (Å)",
}


def apply_style() -> None:
    """Recessive axes and grid, no top/right spines, legends without frames."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": COLOR["rule"],
            "axes.grid": True,
            "grid.color": "#e6e5e1",
            "grid.linewidth": 0.6,
            "axes.titlecolor": COLOR["text"],
            "axes.labelcolor": COLOR["muted"],
            "xtick.color": COLOR["muted"],
            "ytick.color": COLOR["muted"],
            "legend.frameon": False,
            "lines.linewidth": 1.6,
        }
    )


def save(fig, name: str, show: bool = True) -> None:
    """Save a figure as results/figures/<name>.png (and show it in the notebook)."""
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def positive(values, floor: float) -> np.ndarray:
    """For log axes: values of exactly 0 are drawn at the panel's floor."""
    values = np.asarray(values, dtype=float)
    return np.where(values > 0, values, floor)


def log_floor(*arrays) -> float:
    """Half of the smallest positive value over the arrays (the floor of a log axis)."""
    finite = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays if len(a)])
    finite = finite[np.isfinite(finite) & (finite > 0)]
    return 0.5 * finite.min() if len(finite) else 1e-3
