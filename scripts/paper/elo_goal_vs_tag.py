"""Pursuer Elo of the dense goal-reaching controller (goal_scratch_lim G2 seeds, no tag reward) against tag
self-play (tag_chase_adapt RU050 seeds), on one scale.

The goal policy cannot play the evader (it never saw role -1 and falls within ~0.2 s), so every checkpoint is
rated as a pursuer: it plays the robot seat with the role fixed to pursuer (rules.learner_role=pursuer) against
a fixed panel of RU050 evader checkpoints, in RU050_s0's game (tag_chase: 16 m arena, 10 s timeout the evader
wins, falls and exits lose). Scores use the game's outcome (1 win, 0.5 draw, 0 loss). Bradley-Terry over
pursuers and panel evaders (plot_elo.fit), anchored at RU050_s0's untrained pursuer (model_0) = 0.
Deterministic actions, reset seed 0, 4096 games per pursuer split over the panel.

  python scripts/paper/elo_goal_vs_tag.py [--plot-only]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib.pyplot as plt  # noqa: E402
from plot_elo import fit, head_to_head  # noqa: E402

C, G = "logs/mofo/tag/tag_chase_adapt", "logs/mofo/goal/goal_scratch_lim"
OUT = "logs/plots/paper/elo_goal_vs_tag"
RUNS = {  # Label -> (dir, checkpoint iterations rated as pursuers).
  "tag RU050 s0": (f"{C}/RU050_s0", list(range(0, 43501, 3000)) + [43500]),
  "tag RU050 s1": (f"{C}/RU050_s1", list(range(0, 43001, 3000)) + [43000]),
  "goal G2 s0": (f"{G}/G2_s0", list(range(0, 18001, 2000)) + [19999]),
  "goal G2 s1": (f"{G}/G2_s1", list(range(0, 18001, 2000)) + [19999]),
  "goal G2 s2": (f"{G}/G2_s2", list(range(0, 18001, 2000)) + [19999]),
}
PANEL = [f"{C}/RU050_s{s}/model_{i}.pt" for s in (0, 1) for i in (3000, 9000, 15000, 21000, 27000, 33000, 39000)]


def load_env(run: str, n: int, overrides: list[str]):
  """plot_states.load_env with the opponent seat's sensors (selfplay), for head_to_head."""
  from mofo.env import GAME_SCHEMAS, game, make_env
  from omegaconf import OmegaConf
  import torch

  cfg = OmegaConf.load(os.path.join(run, ".hydra", "config.yaml"))
  cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
  for k, schema in GAME_SCHEMAS.items():
    node = OmegaConf.structured(schema)
    OmegaConf.set_readonly(node, False)
    cfg[k] = OmegaConf.merge(node, cfg[k])
  return make_env(cfg, n, game(cfg), eval=True, video=False, selfplay=True)


def play() -> dict:
  env = load_env(f"{C}/RU050_s0", 4096, ["rules.learner_role=pursuer"])
  games = []
  for label, (d, its) in RUNS.items():
    for it in sorted(set(its)):
      res = head_to_head(env, f"{d}/model_{it}.pt", PANEL)
      games += [[f"{label}|{it}", f"E|{e}", r[0], r[1]] for e, r in zip(PANEL, res)]
      print(label, it, f"score {sum(r[0] for r in res) / sum(r[1] for r in res):.3f}", flush=True)
  json.dump({"runs": {k: v[1] for k, v in RUNS.items()}, "panel": PANEL, "games": games}, open(OUT + ".json", "w"))
  return json.load(open(OUT + ".json"))


def plot(state: dict) -> None:
  pursuers = [f"{k}|{it}" for k, its in state["runs"].items() for it in sorted(set(its))]
  players = pursuers + [f"E|{e}" for e in state["panel"]]  # players[0] = RU050_s0 model_0: the anchor.
  elo = dict(zip(players, fit(players, state["games"])))
  style = {"tag RU050 s0": ("#2a78d6", "o", "-"), "tag RU050 s1": ("#2a78d6", "s", "--"),
           "goal G2 s0": ("#1baf7a", "o", "-"), "goal G2 s1": ("#1baf7a", "s", "--"), "goal G2 s2": ("#1baf7a", "D", ":")}
  fig, ax = plt.subplots(figsize=(7, 4))
  for k, its in state["runs"].items():
    its = sorted(set(its))
    c, m, ls = style[k]
    ax.plot(its, [elo[f"{k}|{it}"] for it in its], color=c, marker=m, ls=ls, lw=1.8, ms=4, label=k)
  ax.axhline(0, color="#52514e", lw=0.8)
  ax.set_xlabel("checkpoint (training iteration)")
  ax.set_ylabel("pursuer Elo (RU050 s0 model_0 = 0)")
  ax.grid(color="#e4e3df", lw=0.6)
  ax.spines[["top", "right"]].set_visible(False)
  ax.legend(frameon=False, fontsize=8)
  for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}.{ext}", dpi=200, bbox_inches="tight")
  print(f"saved {OUT}.png/.pdf")
  for k, its in state["runs"].items():
    print(k, {it: round(elo[f"{k}|{it}"]) for it in sorted(set(its))})


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--plot-only", action="store_true")
  a = ap.parse_args()
  plot(json.load(open(OUT + ".json")) if a.plot_only else play())
