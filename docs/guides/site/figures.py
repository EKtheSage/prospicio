"""Figures and tables on the Python docs site that follow its light and dark modes.

`cargo xtask python` runs this file in a hidden first cell of each guide it
executes. A guide's code is plain matplotlib and great_tables: nothing here is
needed to run it elsewhere.

Figures. Matplotlib draws with placeholder colours (``INK``, ``MUTED``,
``RULE`` and the series in ``SERIES``), each figure left open at the end of a
cell is saved as SVG, and the placeholders are replaced by the site's CSS
variables (``var(--p-ink)`` and so on, defined for both modes in
``python/docs-site.css``). A grey (``color="0.75"``, ``"black"``) is an amount
of ink: it becomes the page's ink mixed into the background in that
proportion, so a light grey stays light on a light page and dark on a dark
one; white becomes the background. The SVG is written into the page, so it
takes the colours of the mode the reader has chosen, and changes with it.

Tables. great_tables' own colours are cleared by the site's CSS (``.gt_table``
in docs-site.css), so a table takes the page's colours too.
"""

import io
import logging
import re
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from cycler import cycler  # noqa: E402
from IPython import get_ipython  # noqa: E402
from IPython.display import HTML, display  # noqa: E402

# The site's face (Jost) is the page's, not the build machine's: the SVG keeps
# its text as text and names the page's font below, so matplotlib's warnings
# about a missing font, and about plt.show() under Agg, say nothing useful.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*non-interactive.*")

# Placeholders: colours no figure uses on purpose, each mapped to a CSS
# variable of the site. Matplotlib writes them as lower-case hex in the SVG.
# Three series at most: the palette (docs-site.css) passes the colour-vision
# checks for three; a fourth series belongs in another panel.
INK, MUTED, RULE = "#010203", "#020304", "#030405"
SERIES = ["#0a0b01", "#0a0b02", "#0a0b03"]
CSS = {
    INK: "var(--p-ink)",
    MUTED: "var(--p-muted)",
    RULE: "var(--p-rule)",
    **{c: f"var(--p-series-{i + 1})" for i, c in enumerate(SERIES)},
}

plt.rcParams.update(
    {
        "figure.figsize": (7.0, 3.6),
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "svg.fonttype": "none",
        "font.family": ["Jost", "Segoe UI", "DejaVu Sans"],
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "axes.edgecolor": RULE,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": RULE,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": MUTED,
        "ytick.labelcolor": MUTED,
        "legend.frameon": False,
        "legend.labelcolor": INK,
        "axes.prop_cycle": cycler(color=SERIES),
        "patch.edgecolor": "none",
    }
)

_HEX = re.compile("|".join(re.escape(c) for c in CSS), re.IGNORECASE)
_GREY = re.compile(r"#([0-9a-f]{2})\1\1\b", re.IGNORECASE)


def _grey(m):
    level = int(m.group(1), 16) / 255
    if level == 1.0:
        return "var(--p-bg)"
    return f"color-mix(in srgb, var(--p-ink) {round(100 * (1 - level))}%, transparent)"


def svg(fig):
    """The figure as inline SVG with the site's colours as CSS variables."""
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    text = buf.getvalue()
    text = text[text.index("<svg") :]
    text = _HEX.sub(lambda m: CSS[m.group(0).lower()], text)
    text = _GREY.sub(_grey, text)
    text = re.sub(r"font-family:[^;\"]*", "font-family: var(--p-font)", text)
    # The figure keeps its own size where the column is wide enough and
    # shrinks to the column on a phone (docs-site.css, figure.p-figure svg).
    return f'<figure class="p-figure">{text}</figure>'


def _figure_hook(plt=plt, display=display, HTML=HTML, svg=svg):
    """After each cell, show the figures it left open, then close them.

    The hook holds what it uses: this file runs in the guide's namespace,
    which Quarto clears (`%reset`) before its last cell, and a hook that then
    failed would print its traceback into that cell's output, which Quarto
    reads as JSON.
    """

    def show_open_figures(*_):
        for num in plt.get_fignums():
            display(HTML(svg(plt.figure(num))))
        plt.close("all")

    return show_open_figures


_ip = get_ipython()
if _ip is not None:
    _ip.events.register("post_execute", _figure_hook())
