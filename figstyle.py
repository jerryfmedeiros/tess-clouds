"""figstyle.py — the figure conventions of the paper: print sizes, font, colours, line styles and axis style.

Restrained by design: black and greys, with one accent colour for what the figure is about (the classifier, and the
TESS-W sky temperature). Every figure must read in greyscale, so the comparison methods differ by lightness and line
style as well (METHODS), and the accent is darker than the greys it sits among. Sizes follow Copernicus: one column
8.3 cm, two columns 12 cm; text 7 pt at print size, in Liberation Sans (metric-compatible with Arial; SIL Open Font
License, fonts/LICENSE), shipped in fonts/ so the figures do not depend on the fonts of the machine."""
from __future__ import annotations

from pathlib import Path

from matplotlib import font_manager

CM = 1 / 2.54
COL = 8.3 * CM                 # one-column figure width (in)
WIDE = 12.0 * CM               # two-column figure width (in)
DPI = 300

INK = "#1a1a1a"
ACCENT = "#2b5c8a"             # muted dark blue: the classifier, the sky temperature
G1, G2, G3 = "#4d4d4d", "#808080", "#a6a6a6"
MUTED = "#6b6b6b"
GRID = "#ebebeb"
LIGHT = "#ececec"              # shaded ranges

# method -> (label, colour, line style, marker)
METHODS = {
    "two_test":  ("Two-test classifier", ACCENT, "-", "o"),
    "z_brunt":   ("Baseline test alone", G1, (0, (4, 1.5)), "s"),
    "neg_dT":    ("Die minus sky", G2, (0, (6, 1.5, 1, 1.5)), "^"),
    "dashboard": ("Network dashboard rule", G3, (0, (1, 1.2)), "D"),
    "sky_med":   ("Sky temperature", G3, (0, (3, 1, 1, 1, 1, 1)), "v"),   # not the dashboard rule's dotted line
}

for f in sorted((Path(__file__).parent / "fonts").glob("*.ttf")):
    font_manager.fontManager.addfont(str(f))

RC = {"font.family": "Liberation Sans", "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
      "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.5, "lines.linewidth": 1.0,
      "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
      "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK,
      "savefig.dpi": DPI, "mathtext.default": "regular"}


def style(ax, grid="y"):
    """A light grid behind the data on one axis (or none), no top or right spine."""
    if grid:
        ax.grid(axis=grid, lw=0.4, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
