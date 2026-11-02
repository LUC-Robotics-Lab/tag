"""Paper version of a run's role heatmap (scripts/plot_elo.py plot_roles, the same figure the W&B follower logs
as model_*_roles.png): no title, figure background transparent outside the axes, 300 dpi PNG plus a PDF.

Reads only <run>/plots/elo.json (the checkpoint tournament the follower already played), so no GPU is needed.

  .venv/bin/python scripts/paper/role_heatmap.py RUN_DIR [RUN_DIR ...] [--outdir logs/plots/paper] [--title]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import plot_elo  # noqa: E402


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("runs", nargs="+")
  ap.add_argument("--outdir", default="logs/plots/paper")
  ap.add_argument("--title", action="store_true", help="Keep plot_elo's title.")
  ap.add_argument("--dpi", type=int, default=300)
  a = ap.parse_args()
  for run in a.runs:
    name = os.path.basename(os.path.normpath(run))
    state = json.load(open(os.path.join(run, "plots", "elo.json")))
    out = os.path.join(a.outdir, f"role_matrix_{name}.png")
    plot_elo.plot_roles(state, out, name if a.title else None, transparent=True, dpi=a.dpi)
    print(f"saved {out} and .pdf ({len(state['players'])} checkpoints, {state['players'][0]}-{state['players'][-1]})")


if __name__ == "__main__":
  main()
