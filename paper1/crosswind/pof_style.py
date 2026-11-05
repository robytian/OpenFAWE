"""Figure style of paper 1 (the style of make_draft_figures.py of the journal versions, 2026-10-08): Times New Roman,
ticks inside on all four sides, bold panel letters above the top-left corner, no text inside the plot area apart from
axis labels, legends and letters (author rule); wording goes into the captions. AIP double-column width is 6.69 in."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WIDTH_DOUBLE_IN = 6.69
WIDTH_SINGLE_IN = 3.37


def column_style():
    """Fonts of the single-column figures of the two-column layout (author 2026-10-26 17:13, 17:14)."""
    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 7.5, "lines.markersize": 3.5})
# One palette for the whole paper (author 2026-10-26 17:16), colour-blind safe (Okabe and Ito): blue for the method and the
# production setting (VSM, M600, tube T3, unit A, inner tip), vermilion for the alternative it is compared with (LL,
# model D, outer tip), pink for unit B and the polar-coupled lifting surface, green for the inviscid lifting surface and
# tube T1, sky blue for light variants, orange for rotor rings, grey for references and for runs without rotors.
# Sequential magnitudes use Blues (rings Oranges); signed fields use RdBu_r.
PALETTE = {"blue": "#0072B2", "vermilion": "#D55E00", "pink": "#CC79A7", "green": "#009E73", "sky": "#56B4E9",
           "orange": "#E69F00", "grey": "#7F7F7F", "light grey": "#BDBDBD", "black": "#1A1A1A"}
COL = {"M600": PALETTE["blue"], "V3": PALETTE["vermilion"], "rotors": PALETTE["blue"], "no rotors": PALETTE["light grey"],
       "A": PALETTE["blue"], "B": PALETTE["pink"], "outer": PALETTE["vermilion"], "inner": PALETTE["blue"], "grey": "0.45"}
LAPS = ["#C6DBEF", "#6BAED6", "#2171B5", "#08306B", "#041F45"]     # lap 1 (light) to lap 4 and 5 (dark), Blues
DASH = (0, (3.2, 1.9))


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "TeX Gyre Termes", "DejaVu Serif"],
        "font.size": 10, "axes.labelsize": 11, "axes.linewidth": 0.7, "lines.linewidth": 1.3, "lines.markersize": 4.0,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9, "legend.frameon": False,
        "legend.handlelength": 1.8, "mathtext.fontset": "stix", "savefig.dpi": 300, "figure.dpi": 100})


def letter(ax, text):
    ax.text(0.0, 1.03, text, transform=ax.transAxes, fontweight="bold", fontsize=11, va="bottom", ha="left")


def grid(ax):
    ax.grid(True, color="0.9", linewidth=0.5)
    ax.set_axisbelow(True)


def save(fig, path):
    fig.savefig(str(path)+".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(path)+".pdf", bbox_inches="tight")
    plt.close(fig)
