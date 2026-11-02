"""Paper Elo figures (PNG + PDF in logs/plots/paper/):

  elo_seeds       Elo vs checkpoint for the three tag_scratch_lim seeds (from each run's plots/elo.json).
  elo_ru          The same for the RU runs: RU000 alone, and the four arms of the sweep.
  elo_40k         The same to iteration 40000 for RU050 and TC.

  python scripts/paper/paper_figs.py
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from plot_elo import fit  # noqa: E402

OUT = "logs/plots/paper"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # Validated categorical slots 1-3 (dataviz reference palette).
SERIES4 = SERIES + ["#eda100"]  # Slot 4 (yellow); the 4-color set passes the validator (contrast WARN: legend + labels).
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})


def save(fig, name: str) -> None:
  os.makedirs(OUT, exist_ok=True)
  for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
  plt.close(fig)
  print(f"saved {OUT}/{name}.png/.pdf")


def elo_band(arms: dict[str, list[str]], name: str, title: str, xmax: int | None = None) -> None:
  """Per arm: mean Elo over its seeds per checkpoint (line), shaded from the seed min to max. Each run's Elo is
  from its own checkpoint tournament (plots/elo.json, model_0 = 0), so arms share the anchor, not the games."""
  fig, ax = plt.subplots(figsize=(5.5, 3.2))
  for k, (label, runs) in enumerate(arms.items()):
    curves = []
    for r in runs:
      s = json.load(open(os.path.join(r, "plots", "elo.json")))
      curves.append(dict(zip(s["players"], fit(s["players"], s["games"]))))
    its = sorted(i for i in set.intersection(*(set(c) for c in curves)) if xmax is None or i <= xmax)  # Every seed has.
    e = np.array([[c[i] for i in its] for c in curves])  # [seed, checkpoint]
    ax.fill_between(its, e.min(0), e.max(0), color=SERIES4[k], alpha=0.2, lw=0)
    ax.plot(its, e.mean(0), color=SERIES4[k], lw=2, label=label if len(arms) > 1 else None)
  ax.axhline(0, color=INK2, lw=0.8)
  ax.set_xlabel("checkpoint (training iteration)")
  ax.set_ylabel("Elo (model_0 = 0)")
  ax.set_title(title, loc="left", fontsize=10, color=INK)
  if len(arms) > 1:
    ax.legend(loc="lower right", fontsize=8)
  save(fig, name)


def elo_seeds() -> None:
  T = "logs/mofo/tag/tag_scratch_lim"
  elo_band({"tag_scratch_lim": [f"{T}/L_s0", f"{T}/L_s1_repro3", f"{T}/L_s2"]}, "elo_seeds",
           "Tag self-play: Elo of each checkpoint vs all earlier ones\n(mean of 3 seeds; shaded: min to max)")


def elo_40k() -> None:
  C, T = "logs/mofo/tag/tag_chase_adapt", "logs/mofo/tag/tag_confirm"
  elo_band({"RU050": [f"{C}/RU050_s0", f"{C}/RU050_s1"]}, "elo_RU050_40k",
           "Tag self-play (RU050): Elo of each checkpoint vs all earlier ones\n(mean of 2 seeds; shaded: min to max)",
           xmax=40000)
  elo_band({"TC": [f"{T}/TC_s{s}" for s in range(3)]}, "elo_TC_40k",
           "Tag self-play (TC): Elo of each checkpoint vs all earlier ones\n(mean of 3 seeds; shaded: min to max)",
           xmax=40000)


def elo_ru() -> None:
  C = "logs/mofo/tag/tag_chase_adapt"
  elo_band({"RU000 (fixed 50/50 roles)": [f"{C}/RU000_s0", f"{C}/RU000_s1"]}, "elo_RU000",
           "RU000: Elo of each checkpoint vs all earlier ones\n(mean of 2 seeds; shaded: min to max)")
  arms = {"RU000 (fixed roles)": "RU000", "RU025 (adaptive, step 0.025)": "RU025",
          "RU030 (adaptive, step 0.03)": "RU030", "RU050 (adaptive, step 0.05)": "RU050"}
  elo_band({k: [f"{C}/{v}_s0", f"{C}/{v}_s1"] for k, v in arms.items()}, "elo_RU_arms",
           "RU sweep: Elo of each checkpoint vs all earlier ones\n(mean of 2 seeds per arm; shaded: min to max)")


if __name__ == "__main__":
  elo_seeds()
  elo_ru()
  elo_40k()
