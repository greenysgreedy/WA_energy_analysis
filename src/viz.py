"""Shared plotting style, so every figure in the notebooks reads as one system.

The categorical palette below is validated for colour-vision deficiency separation and
for contrast against the light chart surface. Three slots sit below 3:1 contrast on that
surface (aqua, yellow, magenta), so every chart carries a legend and, where there are
four or fewer series, direct labels as well: identity is never carried by colour alone.

One axis per chart. Never two y-scales.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

#: Categorical slots in fixed order. Assign by series identity, never cycled, and never
#: reordered when a filter changes the series count.
SERIES = (
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
)

#: Single hue for magnitude, light to dark.
SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")

#: Two poles plus a neutral midpoint, for signed quantities such as net export.
DIVERGING_LOW = "#2a78d6"
DIVERGING_MID = "#f0efec"
DIVERGING_HIGH = "#e34948"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

#: Stable colour per battery duration, so the 2-hour case is the same blue everywhere.
DURATION_COLOURS = {1.0: SERIES[0], 2.0: SERIES[1], 4.0: SERIES[2], 8.0: SERIES[3]}

#: Stable colour per strategy.
STRATEGY_COLOURS = {
    "perfect_foresight": INK_PRIMARY,
    "persistence": SERIES[0],
    "yesterday": SERIES[1],
    "last_week": SERIES[2],
}


def use_style() -> None:
    """Apply the project's matplotlib style. Call once at the top of a notebook."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 9.5,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 2.0,
            "lines.markersize": 4.5,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "text.color": INK_PRIMARY,
            "axes.prop_cycle": mpl.cycler(color=list(SERIES)),
        }
    )


def label_last(ax, x, y, text: str, colour: str, *, offset: float = 0.0) -> None:
    """Direct-label the end of a line, so identity is not carried by colour alone."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(6, offset),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=INK_SECONDARY,
    )


def money(ax, axis: str = "y", unit: str = "") -> None:
    """Format an axis as dollars with thousands separators."""
    formatter = mpl.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}{unit}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(formatter)


def percent(ax, axis: str = "y") -> None:
    formatter = mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0%}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(formatter)


def finish(ax, title: str, *, subtitle: str | None = None, ylabel: str | None = None) -> None:
    """Apply the standard title block and axis labels."""
    ax.set_title(title)
    if subtitle:
        ax.text(
            0.0,
            1.02,
            subtitle,
            transform=ax.transAxes,
            fontsize=9,
            color=INK_SECONDARY,
            va="bottom",
        )
    if ylabel:
        ax.set_ylabel(ylabel)


def figure(width: float = 9.0, height: float = 4.0):
    """A single-axis figure at the project's default proportions."""
    return plt.subplots(figsize=(width, height), layout="constrained")
