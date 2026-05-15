"""
Two-stage kriging architecture diagram.
Style: light blue-gray outer container, tan/beige boxes, horizontal two-lane layout.
Reference: academic ML pipeline diagram style.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

# ── palette ────────────────────────────────────────────────────────────────
OUTER_BG   = "#C4D8E8"   # steel-blue container
STAGE1_BG  = "#BDD4BD"   # sage-green lane
STAGE2_BG  = "#ADBDCE"   # slate-blue lane
BOX_FILL   = "#D4B483"   # tan / wood
BOX_EDGE   = "#8A6A40"
INPUT_FILL = "#D8C8A8"   # lighter input
OUT_FILL   = "#EAC96A"   # golden output
ARROW_C    = "#2C2C2C"
TXT_C      = "#1A1A1A"
TITLE1_C   = "#1E4B1E"
TITLE2_C   = "#0F2E4A"

# ── helpers ─────────────────────────────────────────────────────────────────
def rbox(ax, x, y, w, h, fc, ec=BOX_EDGE, lw=1.5, z=4, radius=0.18):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad={radius}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
                       clip_on=False)
    ax.add_patch(p)

def panel(ax, x, y, w, h, fc, ec="none", z=2, radius=0.25):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad={radius}",
                       facecolor=fc, edgecolor=ec, linewidth=1, zorder=z)
    ax.add_patch(p)

def txt(ax, x, y, s, fs=9, ha="center", va="center", bold=False,
        color=TXT_C, z=6, style="normal"):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color,
            fontweight="bold" if bold else "normal",
            fontstyle=style, zorder=z, linespacing=1.35)

def arr(ax, x1, y1, x2, y2, lw=1.5, color=ARROW_C, cs="arc3,rad=0"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                connectionstyle=cs),
                zorder=7)

def line(ax, xs, ys, lw=1.5, color=ARROW_C, z=7):
    ax.plot(xs, ys, color=color, lw=lw, zorder=z, solid_capstyle="round")

# ── figure ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 8.5))
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 20)
ax.set_ylim(0, 8.5)
ax.axis("off")
fig.patch.set_facecolor("white")

# outer container
panel(ax, 0.15, 0.15, 19.7, 8.2, OUTER_BG, ec="#7AAABB", z=1, radius=0.35)

# ── lane panels ─────────────────────────────────────────────────────────────
LX, LW = 2.95, 11.8
# Stage 1 (top)
S1Y, S1H = 4.60, 3.35
panel(ax, LX, S1Y, LW, S1H, STAGE1_BG, ec="#6A9A6A", radius=0.2)
txt(ax, LX + LW/2, S1Y + S1H - 0.30,
    "STAGE 1  —  Indicator Kriging",
    fs=10, bold=True, color=TITLE1_C)

# Stage 2 (bottom)
S2Y, S2H = 0.55, 3.35
panel(ax, LX, S2Y, LW, S2H, STAGE2_BG, ec="#5A7A9A", radius=0.2)
txt(ax, LX + LW/2, S2Y + 0.30,
    "STAGE 2  —  Amount Kriging  (wet stations only)",
    fs=10, bold=True, color=TITLE2_C)

# ── box geometry ─────────────────────────────────────────────────────────────
BW, BH = 2.50, 1.70
# vertical centres
S1CY = S1Y + (S1H - BH) / 2 - 0.12   # shifted down slightly inside panel
S2CY = S2Y + (S2H - BH) / 2 + 0.12   # shifted up slightly

# horizontal positions of 4 boxes
BXS = [3.10, 5.95, 8.80, 11.65]

# ── Stage 1 boxes ────────────────────────────────────────────────────────────
s1_data = [
    ("$I = 1$  if  $Z \\geq 0.5$ mm",   "(wet/dry indicator)"),
    ("Spherical variogram",               "$\\gamma(h)$ fitted per day"),
    ("$\\mathrm{OK}\\;\\rightarrow\\;"
     "\\hat{p}(\\mathbf{s}_0)\\in[0,1]$", ""),
    ("Classify: wet if $\\hat{p} > 0.4$", ""),
]

for i, (bx, (line1, line2)) in enumerate(zip(BXS, s1_data)):
    rbox(ax, bx, S1CY, BW, BH, BOX_FILL)
    cy = S1CY + BH / 2
    if line2:
        txt(ax, bx + BW/2, cy + 0.22, line1, fs=9.0)
        txt(ax, bx + BW/2, cy - 0.22, line2, fs=8.2, color="#444444", style="italic")
    else:
        txt(ax, bx + BW/2, cy, line1, fs=9.0)
    if i < 3:
        arr(ax, bx + BW + 0.17, S1CY + BH/2,
            BXS[i+1] - 0.17,  S1CY + BH/2)

# ── Stage 2 boxes ────────────────────────────────────────────────────────────
s2_data = [
    ("$Q = Z\\,/\\,\\tilde{M}(\\mathbf{s},m)$", "(detrend seasonality)"),
    ("Transform",                                  "none / log / NST"),
    ("OK with exp. variogram",                     "(global, pooled)"),
    ("MC back-transform ($K=100$)",                "$\\rightarrow\\hat{Z}$ in mm"),
]

for i, (bx, (line1, line2)) in enumerate(zip(BXS, s2_data)):
    rbox(ax, bx, S2CY, BW, BH, BOX_FILL)
    cy = S2CY + BH / 2
    if line2:
        txt(ax, bx + BW/2, cy + 0.22, line1, fs=9.0)
        txt(ax, bx + BW/2, cy - 0.22, line2, fs=8.2, color="#444444", style="italic")
    else:
        txt(ax, bx + BW/2, cy, line1, fs=9.0)
    if i < 3:
        arr(ax, bx + BW + 0.17, S2CY + BH/2,
            BXS[i+1] - 0.17,  S2CY + BH/2)

# ── Input box ────────────────────────────────────────────────────────────────
IX, IY, IW, IH = 0.40, 3.40, 2.20, 1.70
rbox(ax, IX, IY, IW, IH, INPUT_FILL, lw=2.0)
txt(ax, IX + IW/2, IY + IH/2 + 0.28,
    "$Z(\\mathbf{s}_i,\\, t)$", fs=10.5, bold=True)
txt(ax, IX + IW/2, IY + IH/2 - 0.05,  "raw daily",   fs=8.5, color="#444")
txt(ax, IX + IW/2, IY + IH/2 - 0.38,  "precipitation", fs=8.5, color="#444")

# ── Output box ───────────────────────────────────────────────────────────────
OX, OY, OW, OH = 15.25, 3.40, 3.20, 1.70
rbox(ax, OX, OY, OW, OH, OUT_FILL, ec="#9A7A20", lw=2.0)
txt(ax, OX + OW/2, OY + OH/2 + 0.25,
    "$\\hat{Z}(\\mathbf{s}_0) = \\hat{p}\\cdot\\hat{Z}_{\\mathrm{amt}}$",
    fs=10.5, bold=True)
txt(ax, OX + OW/2, OY + OH/2 - 0.22, "final prediction", fs=8.5, color="#555")

# ── Wiring: input → fork → stages ───────────────────────────────────────────
FORK_X  = 2.80
INM_X   = IX + IW         # right edge of input box
INM_Y   = IY + IH / 2     # vertical centre of input box
S1_EY   = S1CY + BH / 2   # vertical centre stage-1 lane
S2_EY   = S2CY + BH / 2   # vertical centre stage-2 lane

# input → fork horizontal
line(ax, [INM_X, FORK_X], [INM_Y, INM_Y])
# vertical branch of fork
line(ax, [FORK_X, FORK_X], [S2_EY, S1_EY])
# fork → stage-1 first box
arr(ax, FORK_X, S1_EY, BXS[0] - 0.17, S1_EY)
# fork → stage-2 first box
arr(ax, FORK_X, S2_EY, BXS[0] - 0.17, S2_EY)

# ── Wiring: stages → merge → output ─────────────────────────────────────────
MERGE_X = 14.85
S1_EX   = BXS[-1] + BW    # right edge of last stage-1 box
S2_EX   = BXS[-1] + BW
OUT_MY  = OY + OH / 2      # vertical centre of output box

# stage-1 last box → merge horizontal
line(ax, [S1_EX, MERGE_X], [S1_EY, S1_EY])
# stage-2 last box → merge horizontal
line(ax, [S2_EX, MERGE_X], [S2_EY, S2_EY])
# vertical merge
line(ax, [MERGE_X, MERGE_X], [S2_EY, S1_EY])
# merge → output
arr(ax, MERGE_X, OUT_MY, OX - 0.17, OUT_MY)

# ── Figure label ─────────────────────────────────────────────────────────────
txt(ax, 10, 0.08,
    "Figure: Two-stage kriging pipeline for daily precipitation interpolation",
    fs=8.5, color="#555555", style="italic")

out = ("/Users/etomengoi/Desktop/precip_interpolation_thesis"
       "/images/two_stage_architecture.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
