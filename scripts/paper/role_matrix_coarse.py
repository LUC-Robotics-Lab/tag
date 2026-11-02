"""plot_elo's role matrix (below the diagonal: the row's pursuer win rate vs the column; above: the column's
evader win rate vs the row) on a coarser grid: only checkpoints at multiples of --every, no title.

Every pair of checkpoints in elo.json played each other, so the coarse matrix is the same games, not a resample.

  python scripts/paper/role_matrix_coarse.py RUN [--every 1500] [--out OUT.png] [--transparent]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from plot_elo import role_matrix  # noqa: E402


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("run")
  ap.add_argument("--every", type=int, default=1500)
  ap.add_argument("--out", default=None)
  ap.add_argument("--transparent", action="store_true", help="Transparent figure background.")
  a = ap.parse_args()
  s = json.load(open(os.path.join(a.run, "plots", "elo.json")))
  players = [p for p in s["players"] if p % a.every == 0]
  keep = set(players)
  m = role_matrix(players, [g for g in s["games"] if g[0] in keep and g[1] in keep])
  low = np.tril(np.ones_like(m, dtype=bool), -1)
  fig, (ax, cp, ce) = plt.subplots(1, 3, figsize=(12, 10), gridspec_kw={"width_ratios": (1, 0.03, 0.03), "wspace": 0.25})
  ip = ax.imshow(np.ma.masked_where(~low | np.isnan(m), m), cmap="Reds", vmin=0, vmax=1)
  ie = ax.imshow(np.ma.masked_where(low | np.isnan(m), m), cmap="Blues", vmin=0, vmax=1)
  ticks = list(range(0, len(players), max(1, len(players) // 28)))  # At most ~28 labels, as plot_elo does.
  ax.set_xticks(ticks, [str(players[t]) for t in ticks], rotation=60, ha="right")
  ax.set_yticks(ticks, [str(players[t]) for t in ticks])
  ax.set_xlabel("checkpoint (iteration)")
  ax.set_ylabel("checkpoint (iteration)")
  fig.colorbar(ip, cax=cp, label="below diagonal: row's pursuer win rate (tags the column)")
  fig.colorbar(ie, cax=ce, label="above diagonal: column's evader win rate (not tagged by the row)")
  out = a.out or f"logs/plots/paper/role_matrix_{os.path.basename(os.path.normpath(a.run))}_every{a.every}.png"
  os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
  for ext in (".png", ".pdf"):
    fig.savefig(os.path.splitext(out)[0] + ext, dpi=200, bbox_inches="tight", transparent=a.transparent)
  print(f"saved {out} ({len(players)} checkpoints, {players[0]}-{players[-1]})")


if __name__ == "__main__":
  main()
