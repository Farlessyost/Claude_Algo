"""Render Python equivalents of the three new SVG diagrams to PNGs so we
can verify the visual design before the user reloads the browser.

Same math + layout as the JS code; saved to state/ for review.
"""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "state"

# Common dark theme settings (mirror the CSS)
DARK_BG = "#0b0e14"
PANEL_BG = "#141925"
LINE = "#1f2535"
MUTED = "#7a8aa6"
TEXT = "#cdd5e3"
ACCENT = "#3b82f6"

def style_axes(ax):
    ax.set_facecolor(DARK_BG)
    for spine in ax.spines.values():
        spine.set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=8)


# ----------------------------------------------------------- 1. RESILIENCE WELL
def render_well(risk: float, skew_now: float, skew_history: list, threshold: float,
                title_suffix: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 3.4), facecolor=DARK_BG)
    style_axes(ax)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.05, 1.0)
    ax.set_aspect("auto")

    # Well shape: U(x) = a x^2; a scales with (1 - risk)^1.5
    a = 0.10 + 0.85 * (max(0, 1 - risk)) ** 1.5
    xs = [i / 50.0 - 1.0 for i in range(101)]
    ys = [a * x * x for x in xs]
    color = "#3b82f6"
    if risk > threshold: color = "#ef4444"
    elif risk > 0.7 * threshold: color = "#f59e0b"
    ax.plot(xs, ys, color=color, linewidth=2.2,
            path_effects=[matplotlib.patheffects.withStroke(linewidth=4, foreground=color, alpha=0.35)])

    # Floor + threshold rim at 0.75 of well max
    well_max = a * 1.0
    rim_y = 0.75 * (well_max if well_max > 0 else 1.0)
    ax.axhline(rim_y, color="#ef4444", linestyle="--", linewidth=1, alpha=0.55)
    ax.text(0.95, rim_y + 0.02, "gate", color=MUTED, fontsize=8, ha="right")

    # Ball: compute z from skew_now vs history
    z = 0.0
    if len(skew_history) >= 8:
        mu = sum(skew_history) / len(skew_history)
        var = sum((v - mu) ** 2 for v in skew_history) / (len(skew_history) - 1)
        std = math.sqrt(var) if var > 0 else 0
        if std > 0:
            z = (skew_now - mu) / std
    ball_x = max(-1, min(1, z / 3))
    ball_y = a * ball_x * ball_x + 0.03
    ball_color = "#86efac"
    if risk > threshold: ball_color = "#ff7a7a"
    elif risk > 0.7 * threshold: ball_color = "#fbbf24"
    ax.scatter([ball_x], [ball_y], color=ball_color, s=140, zorder=5,
                edgecolors="black", linewidths=1.5)

    # Labels
    ax.text(0, -0.025, "deviation x_t", color=MUTED, fontsize=9, ha="center")
    ax.text(-1.05, well_max * 0.9 + 0.02, "U(x)", color=MUTED, fontsize=9, ha="left")
    ax.text(-1.05, -0.03, "-σ", color=MUTED, fontsize=9, ha="left")
    ax.text(1.05, -0.03, "+σ", color=MUTED, fontsize=9, ha="right")
    ax.set_title(f"Resilience Well · {title_suffix}",
                  color=TEXT, fontsize=11, loc="left", pad=10)
    ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(out_path, facecolor=DARK_BG, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------- 2. WINDOW OF VITALITY
def render_vitality(rel_A: float, dist: float, trail: list, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 4.6), facecolor=DARK_BG)
    style_axes(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2)

    # Quadrant fills (subtle)
    ax.axhspan(0.7, 2, xmin=0.5, xmax=1.0, facecolor="#2a0f0f", alpha=0.35)
    ax.axhspan(0.7, 2, xmin=0, xmax=0.5, facecolor="#2a1d0a", alpha=0.35)
    ax.axhspan(0, 0.7, xmin=0, xmax=0.5, facecolor="#1a2238", alpha=0.35)
    ax.axhspan(0, 0.7, xmin=0.5, xmax=1.0, facecolor="#0c2a17", alpha=0.30)

    # Quadrant labels
    ax.text(0.02, 1.93, "DIFFUSE", color="#f59e0b", fontsize=8, fontweight="bold")
    ax.text(0.98, 1.93, "BRITTLE", color="#ff7a7a", fontsize=8, fontweight="bold", ha="right")
    ax.text(0.02, 0.05, "QUIET", color="#94a3b8", fontsize=8, fontweight="bold")
    ax.text(0.98, 0.05, "ORGANIZED", color="#86efac", fontsize=8, fontweight="bold", ha="right")

    # Threshold lines
    ax.axvline(0.5, color="#94a3b8", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(0.7, color="#94a3b8", linestyle="--", linewidth=0.8, alpha=0.5)
    # Grid
    for v in [0.25, 0.75]:
        ax.axvline(v, color="#2a3554", linewidth=0.5, alpha=0.4, linestyle=":")
    for v in [0.5, 1.0, 1.5]:
        ax.axhline(v, color="#2a3554", linewidth=0.5, alpha=0.4, linestyle=":")

    # Trail
    if len(trail) >= 2:
        xs = [p["a"] for p in trail]
        ys = [p["d"] for p in trail]
        ax.plot(xs, ys, color="#38bdf8", alpha=0.25, linewidth=1)
        for i, p in enumerate(trail[:-1]):
            age = (len(trail) - 1 - i) / max(1, len(trail))
            ax.scatter([p["a"]], [p["d"]], s=(1.5 + (1 - age) * 1.5) ** 2 * 8,
                        color="#7dd3fc", alpha=(1 - age) * 0.5)

    # Current point
    pt_color = "#7dd3fc"
    if rel_A >= 0.5 and dist >= 0.7: pt_color = "#ff7a7a"
    elif rel_A < 0.5 and dist >= 0.7: pt_color = "#fbbf24"
    elif rel_A >= 0.5 and dist < 0.7: pt_color = "#86efac"
    else: pt_color = "#cbd5e1"
    ax.scatter([rel_A], [dist], s=220, color=pt_color, edgecolors="#0e1c3a",
                linewidths=2, zorder=5)
    ax.text(rel_A, dist - 0.13, f"{rel_A:.2f} · {dist:.2f}",
             color=MUTED, fontsize=9, ha="center", fontweight="bold")

    ax.set_xlabel("rel ascendancy  A/C", color=MUTED, fontsize=10)
    ax.set_ylabel("disturbance", color=MUTED, fontsize=10)
    ax.set_title("Window of Vitality", color=TEXT, fontsize=11, loc="left", pad=10)
    plt.tight_layout()
    fig.savefig(out_path, facecolor=DARK_BG, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------- 3. ALPHA DECOMPOSITION
def render_alpha_decomp(weights: dict, raw: dict, parts: dict, blended: float,
                          out_path: Path, title_suffix: str):
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=DARK_BG)
    style_axes(ax)

    components = [
        ("mpc",          "MPC alpha",   "validated reversion",  "#7dd3fc"),
        ("spot_lead",    "spot lead",   "Coinbase BTC 3m",      "#86efac"),
        ("funding_fade", "funding",     "perp fade",            "#c4a8ff"),
        ("oi_pressure",  "OI press.",   "Hyperliquid",          "#f59e0b"),
    ]
    all_vals = [blended] + [parts.get(c[0], 0) for c in components] + [raw.get(c[0], 0) for c in components]
    maxAbs = max(0.05, max(abs(v) for v in all_vals))

    ax.set_xlim(-1.1 * maxAbs, 1.1 * maxAbs)
    n = len(components) + 1
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()

    # Zero line
    ax.axvline(0, color="#7a8aa6", linewidth=1, linestyle="--", alpha=0.55)

    # Tick labels
    ax.text(-maxAbs, -0.45, "SHORT", color=MUTED, fontsize=9, ha="left")
    ax.text(0, -0.45, "0", color=MUTED, fontsize=9, ha="center")
    ax.text(maxAbs, -0.45, "LONG", color=MUTED, fontsize=9, ha="right")

    # Component rows
    for i, (key, label, note, color) in enumerate(components):
        w = weights.get(key, 0)
        p = parts.get(key, 0)
        dim = abs(p) < 1e-5
        alpha = 0.35 if dim else 1.0
        # Bar
        ax.barh(i, p, left=0, height=0.55, color=color, alpha=alpha,
                 edgecolor="#0e2236", linewidth=1)
        ax.text(-maxAbs * 1.05, i, f"{label}", color=TEXT, fontsize=10,
                 ha="right", va="center", fontweight="bold", alpha=alpha)
        ax.text(-maxAbs * 1.05, i + 0.32, f"w={w:.2f}", color=MUTED, fontsize=8,
                 ha="right", va="center")
        sign = "+" if p >= 0 else ""
        ax.text(maxAbs * 1.05, i, f"{sign}{p:.3f}", color=MUTED, fontsize=10,
                 ha="left", va="center")

    # Divider
    ax.axhline(len(components) - 0.5, color="#2a3554", linewidth=0.8, alpha=0.6)

    # Blended row
    blend_y = len(components)
    bcolor = "#fbbf24"
    ax.barh(blend_y, blended, left=0, height=0.75, color=bcolor, alpha=0.85,
             edgecolor="#fbbf24", linewidth=1.5)
    # Arrow
    if abs(blended) > 1e-4:
        ax.annotate("", xy=(blended, blend_y), xytext=(0, blend_y),
                     arrowprops=dict(arrowstyle="->", color=bcolor, lw=2.2, alpha=0.95))
    ax.text(-maxAbs * 1.05, blend_y, "BLENDED", color="#fbbf24", fontsize=10,
             ha="right", va="center", fontweight="bold")
    ax.text(-maxAbs * 1.05, blend_y + 0.36, "= Σ wᵢ αᵢ", color=MUTED, fontsize=8,
             ha="right", va="center")
    sign = "+" if blended >= 0 else ""
    ax.text(maxAbs * 1.05, blend_y, f"{sign}{blended:.3f}", color="#fbbf24",
             fontsize=10, ha="left", va="center", fontweight="bold")

    ax.set_yticks([]); ax.set_xticks([])
    ax.set_title(f"Signal Stack · alpha decomposition · {title_suffix}",
                  color=TEXT, fontsize=11, loc="left", pad=10)
    plt.tight_layout()
    fig.savefig(out_path, facecolor=DARK_BG, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------- run
import matplotlib.patheffects  # noqa: E402  (needed by render_well)

if __name__ == "__main__":
    # Scenario A: calm market, MPC firing modestly long
    render_well(risk=0.30, skew_now=0.45, skew_history=[0.4 + i*0.01 for i in range(60)],
                 threshold=0.95, title_suffix="calm regime (risk 0.30)",
                 out_path=OUT / "_diag_well_calm.png")
    # Scenario B: stressed
    render_well(risk=0.82, skew_now=0.72, skew_history=[0.5 + i*0.005 for i in range(60)],
                 threshold=0.95, title_suffix="stressed (risk 0.82)",
                 out_path=OUT / "_diag_well_stressed.png")
    # Scenario C: brittle / over threshold
    render_well(risk=0.97, skew_now=0.9, skew_history=[0.4 + i*0.005 for i in range(60)],
                 threshold=0.95, title_suffix="BRITTLE (risk 0.97 > gate)",
                 out_path=OUT / "_diag_well_brittle.png")

    # Vitality scenarios
    trail = [{"a": 0.10 + i*0.02, "d": 0.5 + i*0.04} for i in range(15)]
    render_vitality(rel_A=0.45, dist=1.1, trail=trail,
                     out_path=OUT / "_diag_vitality_diffuse.png")
    trail2 = [{"a": 0.55 + (i%5)*0.01, "d": 0.6 + i*0.02} for i in range(20)]
    render_vitality(rel_A=0.62, dist=1.3, trail=trail2,
                     out_path=OUT / "_diag_vitality_brittle.png")
    trail3 = [{"a": 0.30, "d": 0.25} for _ in range(8)]
    render_vitality(rel_A=0.30, dist=0.35, trail=trail3,
                     out_path=OUT / "_diag_vitality_quiet.png")

    # Alpha decomposition scenarios
    render_alpha_decomp(
        weights={"mpc": 1.0, "spot_lead": 0.30, "funding_fade": 0.0, "oi_pressure": 0.0},
        raw={"mpc": -0.18, "spot_lead": +0.42, "funding_fade": 0.0, "oi_pressure": 0.0},
        parts={"mpc": -0.18, "spot_lead": +0.126, "funding_fade": 0.0, "oi_pressure": 0.0},
        blended=-0.054,
        title_suffix="MPC fade vs spot follow-through",
        out_path=OUT / "_diag_alpha_mixed.png")
    render_alpha_decomp(
        weights={"mpc": 1.0, "spot_lead": 0.30, "funding_fade": 0.15, "oi_pressure": 0.15},
        raw={"mpc": +0.35, "spot_lead": +1.20, "funding_fade": -0.50, "oi_pressure": +0.40},
        parts={"mpc": +0.35, "spot_lead": +0.36, "funding_fade": -0.075, "oi_pressure": +0.06},
        blended=+0.695,
        title_suffix="aligned bullish — all signals firing long",
        out_path=OUT / "_diag_alpha_aligned.png")

    print("Wrote 8 preview PNGs to", OUT)
