"""The planar base velocity figure (scripts/plot_states.py plot_velocity) split into one figure per role, saved as
fully vector PDFs: the density is drawn with pcolormesh (one vector quad per bin, not an embedded bitmap). Both
roles share one log color scale, as in the combined figure. No suptitle. A PNG preview is written next to each PDF.

Reads only the rollout's histogram npz (written by plot_states / the W&B follower), so no GPU is needed.

  .venv/bin/python scripts/paper/vel_pdf_split.py NPZ --out OUT_STEM [--frame heading|world] [--no-title]
  -> OUT_STEM_pursuer.pdf/.png, OUT_STEM_evader.pdf/.png
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import plot_states as ps  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

FLOOR = 1e-3  # Densities below this are left blank, as in plot_velocity.


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("npz")
  ap.add_argument("--out", required=True)
  ap.add_argument("--frame", default="heading", choices=("heading", "world"))
  ap.add_argument("--no-title", action="store_true", help="Drop the role name above the plot.")
  a = ap.parse_args()
  h = dict(np.load(a.npz))
  edges = np.linspace(-ps.VEL_MAX, ps.VEL_MAX, ps.VEL_BINS + 1)
  area = np.diff(edges)[0] ** 2
  dens = {r: h[f"{r}_vel_{a.frame}"] / h[f"{r}_vel_{a.frame}"].sum() / area for r in ps.roles(h)}
  norm = LogNorm(FLOOR, max(d.max() for d in dens.values()))
  fwd, lat = ("forward ", "left ") if a.frame == "heading" else ("", "")
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  for r, d in dens.items():
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.pcolormesh(edges, edges, np.ma.masked_less(d.T, FLOOR), cmap=ps.CMAPS[r], norm=norm,
                       shading="flat", edgecolors="face", linewidth=0, rasterized=False)
    ax.axhline(0, color="0.6", lw=0.5)
    ax.axvline(0, color="0.6", lw=0.5)
    if not a.no_title:
      ax.set_title(r.capitalize(), color=ps.COLORS[r], fontweight="bold", fontsize=14)
    ax.set_xlabel(f"$v_x$ {fwd}(m/s)")
    ax.set_ylabel(f"$v_y$ {lat}(m/s)")
    ax.set_aspect("equal")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Probability density (log)")
    cb.solids.set_rasterized(False)  # Matplotlib rasterizes long colorbars by default.
    fig.tight_layout()
    for ext in ("pdf", "png"):
      fig.savefig(f"{a.out}_{r}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {a.out}_{r}.pdf/.png")


if __name__ == "__main__":
  main()
