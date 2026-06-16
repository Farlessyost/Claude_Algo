"""Re-render the fixed alpha decomp and the new fast guard diagram via matplotlib."""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / "state"

DARK_BG = "#0b0e14"; TEXT = "#cdd5e3"; MUTED = "#7a8aa6"

# 1) Alpha decomp — aligned bullish scenario, with the new wider layout
fig, ax = plt.subplots(figsize=(9, 4.4), facecolor=DARK_BG)
ax.set_facecolor(DARK_BG)
components = [
    ("MPC alpha",  +0.350, 1.00, "#7dd3fc", "validated reversion"),
    ("spot lead",  +0.360, 0.30, "#86efac", "Coinbase BTC 3m"),
    ("funding",    -0.075, 0.15, "#c4a8ff", "perp fade"),
    ("OI press.",  +0.060, 0.15, "#f59e0b", "Hyperliquid"),
]
blended = sum(c[1] for c in components)
maxAbs = max(0.05, max(abs(c[1]) for c in components), abs(blended))
ax.set_xlim(-1.2 * maxAbs, 1.2 * maxAbs)
n = len(components)
ax.set_ylim(-0.5, n + 0.5)
ax.invert_yaxis()
ax.axvline(0, color=MUTED, linestyle="--", linewidth=1, alpha=0.55)
ax.text(-maxAbs, -0.4, "SHORT", color=MUTED, fontsize=9, ha="left")
ax.text(0, -0.4, "0", color=MUTED, fontsize=9, ha="center")
ax.text(maxAbs, -0.4, "LONG", color=MUTED, fontsize=9, ha="right")
for i, (label, v, w, color, _note) in enumerate(components):
    dim = abs(v) < 1e-5
    alpha = 0.35 if dim else 1.0
    ax.barh(i, v, height=0.55, color=color, alpha=alpha,
             edgecolor="#0e2236", linewidth=1)
    ax.text(-maxAbs * 1.10, i, label, color=TEXT, fontsize=10, ha="right",
             va="center", fontweight="bold")
    ax.text(-maxAbs * 1.10, i + 0.36, f"w={w:.2f}", color=MUTED, fontsize=8,
             ha="right", va="center")
    sign = "+" if v >= 0 else ""
    ax.text(maxAbs * 1.10, i, f"{sign}{v:.3f}", color=MUTED, fontsize=10, ha="left",
             va="center")
# divider
ax.axhline(n - 0.5, color="#2a3554", linewidth=0.8, alpha=0.75)
# blended row
by = n
ax.barh(by, blended, height=0.75, color="#fbbf24", alpha=0.85,
         edgecolor="#fbbf24", linewidth=1.5)
ax.annotate("", xy=(blended, by), xytext=(0, by),
             arrowprops=dict(arrowstyle="->", color="#fbbf24", lw=2.2, alpha=0.95))
ax.text(-maxAbs * 1.10, by, "BLENDED", color="#fbbf24", fontsize=10,
         ha="right", va="center", fontweight="bold")
ax.text(-maxAbs * 1.10, by + 0.40, "= Σ wᵢ αᵢ", color=MUTED, fontsize=8,
         ha="right", va="center")
sign = "+" if blended >= 0 else ""
ax.text(maxAbs * 1.10, by, f"{sign}{blended:.3f}", color="#fbbf24", fontsize=10,
         ha="left", va="center", fontweight="bold")
ax.set_yticks([]); ax.set_xticks([])
for spine in ax.spines.values():
    spine.set_color("#1f2535")
ax.set_title("Signal Stack · alpha decomposition (fixed layout)", color=TEXT,
              fontsize=11, loc="left", pad=12)
plt.tight_layout()
fig.savefig(OUT / "_diag_alpha_fixed.png", facecolor=DARK_BG, dpi=110,
             bbox_inches="tight")
plt.close(fig)

# 2) Fast guard diagram — 3 scenarios
def fg_diag(move_bps, threshold, polls, fires, running, label, out):
    fig, ax = plt.subplots(figsize=(7, 2.2), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    xMax = threshold * 1.5
    ax.set_xlim(-xMax, xMax)
    ax.set_ylim(-0.7, 1.5)
    # Bands (left to right): danger / warn / safe / warn / danger
    ax.axvspan(-xMax, -threshold, color="#2a0f0f", alpha=0.35)
    ax.axvspan(-threshold, -threshold * 0.5, color="#2a1d0a", alpha=0.30)
    ax.axvspan(-threshold * 0.5, threshold * 0.5, color="#0c2a17", alpha=0.30)
    ax.axvspan(threshold * 0.5, threshold, color="#2a1d0a", alpha=0.30)
    ax.axvspan(threshold, xMax, color="#2a0f0f", alpha=0.35)
    # Threshold lines
    for t in [-threshold, threshold]:
        ax.axvline(t, color="#ef4444", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(0, color="#2a3554", linestyle=":", linewidth=0.5, alpha=0.55)
    # Pointer (triangle)
    color = "#7dd3fc"
    if abs(move_bps) >= threshold: color = "#ff7a7a"
    elif abs(move_bps) >= threshold * 0.5: color = "#fbbf24"
    ax.plot([move_bps, move_bps - 0.06 * xMax, move_bps + 0.06 * xMax, move_bps],
            [0.5, 0.05, 0.05, 0.5], color=color, linewidth=2)
    ax.fill_between([move_bps - 0.06 * xMax, move_bps + 0.06 * xMax],
                     [0.05, 0.05], [0.5, 0.5], color=color, alpha=0.85)
    ax.text(move_bps, 0.95, f"{'+' if move_bps >= 0 else ''}{move_bps:.1f}bps",
             color=TEXT, fontsize=10, ha="center", fontweight="bold")
    # Tick labels
    ax.text(0, -0.45, "0", color=MUTED, fontsize=9, ha="center")
    ax.text(-threshold, -0.45, f"−{int(threshold)}bps", color=MUTED, fontsize=9, ha="center")
    ax.text(threshold, -0.45, f"+{int(threshold)}bps", color=MUTED, fontsize=9, ha="center")
    ax.text(0, 1.35, "intra-cycle mid drift", color=MUTED, fontsize=9, ha="center")
    if running:
        ax.scatter([-xMax * 0.93], [1.30], color="#38bdf8", s=24, alpha=0.7)
    ax.set_yticks([]); ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_color("#1f2535")
    ax.set_title(f"Fast Guard · {label}", color=TEXT, fontsize=11, loc="left", pad=8)
    plt.tight_layout()
    fig.savefig(out, facecolor=DARK_BG, dpi=110, bbox_inches="tight")
    plt.close(fig)

fg_diag(2.1, 10, polls=42, fires=0, running=True,
         label="safe · 3s poll active",
         out=OUT / "_diag_fg_safe.png")
fg_diag(6.4, 10, polls=58, fires=0, running=True,
         label="WARN · drift building",
         out=OUT / "_diag_fg_warn.png")
fg_diag(-12.7, 10, polls=63, fires=4, running=True,
         label="EMERGENCY · cycle fired",
         out=OUT / "_diag_fg_fire.png")

print("Wrote diagrams to", OUT)
