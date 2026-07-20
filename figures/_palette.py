"""Color-blind-safe palette (Wong 2011) + shared matplotlib rcParams.

Use ``WONG_PALETTE`` for line/bar colors and ``apply_rcparams()`` at the top
of every figure script so output is visually consistent."""
from __future__ import annotations


WONG_PALETTE = [
    "#0072B2",  # blue (primary)
    "#E69F00",  # orange (accent)
    "#009E73",  # green (third)
    "#CC79A7",  # purple
    "#56B4E9",  # light blue
    "#D55E00",  # red-orange
    "#F0E442",  # yellow
]


def apply_rcparams() -> None:
    """Apply consistent rcParams for all figures (run once per script)."""
    import matplotlib as mpl
    mpl.rcParams["font.family"] = ["sans-serif"]
    mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.labelsize"] = 8
    mpl.rcParams["axes.titlesize"] = 9
    mpl.rcParams["xtick.labelsize"] = 7
    mpl.rcParams["ytick.labelsize"] = 7
    mpl.rcParams["legend.fontsize"] = 7
    mpl.rcParams["figure.dpi"] = 150
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["savefig.bbox"] = "tight"


def watermark_preliminary(ax) -> None:
    """Stamp 'PRELIMINARY (synthetic placeholder)' across an axes.

    Use this when figure data is synthetic so the paper PDF visually warns
    that the numbers are placeholders."""
    ax.text(
        0.5, 0.5, "PRELIMINARY\n(synthetic placeholder)",
        transform=ax.transAxes,
        ha="center", va="center",
        rotation=30, fontsize=20, color="#CC79A7", alpha=0.25,
        weight="bold", zorder=100,
    )
