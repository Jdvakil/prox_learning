"""Top-down design overview of the cluttered fume-hood tasks.

Three panels (smallest / middle / largest hood), each showing: the hood shell
at true scale, the object reach span (6-34 cm past the mouth), 200 sampled
clutter placements using the sampler's actual placement + keep-out maths, the
place tray, and the push / pull displacement arrows. Pure matplotlib — runs
anywhere, no simulator needed.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrow, Rectangle

X_FRONT, T = 0.58, 0.012
REACH_SPAN = (0.06, 0.34)
CORRIDOR_R, OBJ_KEEPOUT = 0.14, 0.12
PANELS = [("v00 smallest", 0.32, 0.58), ("v13 middle", 0.45, 0.78), ("v26 largest", 0.58, 1.00)]

rng = np.random.default_rng(7)
fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)

for ax, (label, half_w, depth) in zip(axes, PANELS):
    x0, x1 = X_FRONT, X_FRONT + depth
    # hood shell
    ax.add_patch(Rectangle((x0, -half_w - T), depth + T, T, color="0.35"))
    ax.add_patch(Rectangle((x0, half_w), depth + T, T, color="0.35"))
    ax.add_patch(Rectangle((x1, -half_w - T), T, 2 * half_w + 2 * T, color="0.35"))
    # object reach span on the centreline
    ax.plot([x0 + REACH_SPAN[0], x0 + REACH_SPAN[1]], [0, 0], lw=7, color="tab:red",
            alpha=0.75, solid_capstyle="round", label="object reach span 6–34 cm", zorder=4)
    # place tray (front-right corner, matches gen_fumehood_variants)
    ax.add_patch(Rectangle((X_FRONT + 0.14 - 0.09, -(half_w - 0.12) - 0.09), 0.18, 0.18,
                           facecolor="tab:blue", alpha=0.55, label="place tray"))
    # clutter draws with the sampler's real placement + keep-out maths
    ox, oy = x0 + 0.20, 0.0   # representative object position
    pts = []
    for _ in range(200):
        u, v = rng.uniform(0.12, 0.92), rng.uniform(-0.88, 0.88)
        x = X_FRONT + 0.05 + u * max(depth - 0.10, 0.05)
        y = v * max(half_w - 0.06, 0.05)
        if abs(y - oy) < CORRIDOR_R and X_FRONT - 0.05 <= x <= ox + 0.05:
            continue
        if np.hypot(x - ox, y - oy) < OBJ_KEEPOUT:
            continue
        if np.hypot(x - (X_FRONT + 0.14), y - (-(half_w - 0.12))) < 0.16:
            continue   # tray keep-out, mirrors the sampler
        pts.append((x, y))
    pts = np.array(pts)
    ax.plot(pts[:, 0], pts[:, 1], "s", ms=3.5, color="tab:green", alpha=0.5,
            label="clutter placements (200 draws)")
    ax.plot([ox], [oy], "o", ms=9, color="tab:red", zorder=5)
    # push / pull arrows
    ax.add_patch(FancyArrow(ox, oy, 0.11, 0.0, width=0.008, head_width=0.03,
                            color="tab:purple", zorder=6))
    ax.add_patch(FancyArrow(ox, oy, -0.10, 0.0, width=0.008, head_width=0.03,
                            color="tab:orange", zorder=6))
    ax.annotate("push", (ox + 0.13, 0.02), color="tab:purple", fontsize=9)
    ax.annotate("pull", (ox - 0.16, 0.02), color="tab:orange", fontsize=9)
    ax.axvline(X_FRONT, color="0.6", ls="--", lw=1)
    ax.annotate("mouth", (X_FRONT + 0.005, -half_w - 0.06), fontsize=8, color="0.4")
    ax.set_title(f"{label}   w={2*half_w:.2f} m  d={depth:.2f} m", fontsize=10)
    ax.set_xlim(0.40, X_FRONT + 1.06)
    ax.set_ylim(-0.72, 0.72)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)

axes[0].legend(loc="upper left", fontsize=7.5, framealpha=0.9)
fig.suptitle("Cluttered fume-hood tasks — top-down: clutter keep-outs, reach span, "
             "push/pull directions, place tray", fontsize=12)
fig.tight_layout()
out = Path(__file__).resolve().parent.parent / "figs" / "env_design_topdown.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")
