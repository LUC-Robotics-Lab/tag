"""Extra RU050 (tag_chase_adapt, evader_rate_step 0.05; seeds 0 and 1) results for the paper, written to
logs/plots/paper/ru050/ (PNG + PDF figures, Markdown + CSV tables):

  late_vs_earlier  The 40k checkpoint's head-to-head results against every earlier checkpoint (plots/elo.json:
                   deterministic, ~4096 games per pair split over both role assignments): overall score
                   (win 1, draw 0.5), tag rate as pursuer, not-tagged rate as evader.
  behaviour        Per role over checkpoints, from the follower's self-play state histograms (plots/model_*.npz,
                   4096 worlds, one episode each, deterministic): mean planar speed, share of moving time spent
                   going forward / sideways / backward (heading frame), mean |torque| / limit.
  outcomes         How self-play games end per checkpoint (scripts/paper/ru050_evals.sh: 4096 games, deterministic,
                   reset seed 0), stacked; and the table at 20k / 40k.
  cross            Seed cross-play at 20k / 40k: pursuer-seat checkpoint vs evader-seat checkpoint, all four pairs.
  roles_trace      From the training log (scripts/paper/tb_extract.py): P(learner is evader) in league envs and the
                   evader win rate, and the training reward.
  cost             Training cost to 40k: iterations, env steps, learner transitions, gradient steps, wall-clock.

  .venv/bin/python scripts/paper/ru050_figs.py [NAME ...]   (default: all whose inputs exist)
"""

import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from paper_figs import INK, INK2  # noqa: E402  (also sets the shared rcParams)
import plot_states as ps  # noqa: E402

C = "logs/mofo/tag/tag_chase_adapt"
E = "logs/plots/ru050"  # Evaluation / training-log extracts.
OUT = "logs/plots/paper/ru050"
SEEDS = (0, 1)
LS = {0: "-", 1: "--"}
P, V = ps.COLORS["pursuer"], ps.COLORS["evader"]
LAST = 40000


def save(fig, name: str) -> None:
  os.makedirs(OUT, exist_ok=True)
  for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/{name}.{ext}", dpi=200, bbox_inches="tight")
  plt.close(fig)
  print(f"saved {OUT}/{name}.png/.pdf")


def table(name: str, header: list[str], rows: list[list]) -> None:
  os.makedirs(OUT, exist_ok=True)
  md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]
  open(f"{OUT}/{name}.md", "w").write("\n".join(md) + "\n")
  open(f"{OUT}/{name}.csv", "w").write("\n".join(",".join(map(str, r)) for r in [header] + rows) + "\n")
  print("\n".join(md))
  print(f"saved {OUT}/{name}.md/.csv")


def seed_legend(ax, loc="best") -> None:
  ax.legend(handles=[plt.Line2D([], [], color=INK2, ls=LS[s], label=f"seed {s}") for s in SEEDS], loc=loc, fontsize=8)


def late_vs_earlier() -> None:
  fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
  rows = []
  for s in SEEDS:
    g = sorted((x for x in json.load(open(f"{C}/RU050_s{s}/plots/elo.json"))["games"] if x[0] == LAST), key=lambda x: x[1])
    j = np.array([x[1] for x in g])
    score = np.array([x[2] / x[3] for x in g])
    tag = np.array([x[5] / x[4] for x in g])
    surv = np.array([1 - x[7] / x[6] for x in g])
    axs[0].plot(j, score, color=INK, ls=LS[s], lw=1.6)
    axs[1].plot(j, tag, color=P, ls=LS[s], lw=1.6)
    axs[1].plot(j, surv, color=V, ls=LS[s], lw=1.6)
    rows.append([f"seed {s}", len(g), int((score > 0.5).sum()), f"{score.mean():.2f}", f"{score[j >= 20000].mean():.2f}",
                 f"{tag.mean():.2f}", f"{surv.mean():.2f}", int(np.median([x[3] for x in g]))])
  for ax in axs:
    ax.axhline(0.5, color=INK2, lw=0.8)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_xlabel("earlier checkpoint (training iteration)")
    ax.set_ylim(0, 1)
  axs[0].set_ylabel("40k checkpoint's score (win 1, draw 0.5)")
  axs[0].set_title("Overall", loc="left", fontsize=10, color=INK)
  axs[1].set_title("By role: tag rate as pursuer (red), not tagged as evader (blue)", loc="left", fontsize=10, color=INK)
  seed_legend(axs[0], "lower left")
  save(fig, "late_vs_earlier")
  table("late_vs_earlier", ["run", "earlier ckpts", "beaten (score > 0.5)", "mean score", "mean score vs >= 20k",
                            "mean tag rate (P)", "mean not-tagged (E)", "games per pair"], rows)


def _hist_stats(h: dict, r: str) -> tuple[float, float, float, float, float]:
  speed = float(h[f"{r}_speed_sum"]) / float(h[f"{r}_steps"])
  edges = np.linspace(-ps.VEL_MAX, ps.VEL_MAX, ps.VEL_BINS + 1)
  c = (edges[:-1] + edges[1:]) / 2
  vx, vy = np.meshgrid(c, c, indexing="ij")
  v = h[f"{r}_vel_heading"]
  moving = v * (np.hypot(vx, vy) > 0.3)  # Ignore standing still / swaying. Forward/backward/sideways: 90 deg sectors.
  m = moving.sum()
  fwd = moving[vx > np.abs(vy)].sum() / m
  back = moving[-vx > np.abs(vy)].sum() / m
  side = 1 - fwd - back
  t = h[f"{r}_tau"]
  tc = np.linspace(-1, 1, len(t) + 1)
  tau = float((t * np.abs((tc[:-1] + tc[1:]) / 2)).sum() / t.sum())
  return speed, fwd, side, back, tau


def behaviour() -> None:
  fig, axs = plt.subplots(1, 5, figsize=(18, 3.2))
  for s in SEEDS:
    files = sorted(glob.glob(f"{C}/RU050_s{s}/plots/model_*.npz"), key=lambda f: int(f.rsplit("_", 1)[1][:-4]))
    its = np.array([int(f.rsplit("_", 1)[1][:-4]) for f in files])
    keep = its <= LAST
    stats = {r: np.array([_hist_stats(dict(np.load(f)), r) for f, k in zip(files, keep) if k]) for r in ("pursuer", "evader")}
    its = its[keep]
    for r, col in (("pursuer", P), ("evader", V)):
      st = stats[r]
      axs[0].plot(its, st[:, 0], color=col, ls=LS[s], lw=1.5)
      axs[1].plot(its, st[:, 1], color=col, ls=LS[s], lw=1.5)
      axs[2].plot(its, st[:, 2], color=col, ls=LS[s], lw=1.5)
      axs[3].plot(its, st[:, 3], color=col, ls=LS[s], lw=1.5)
      axs[4].plot(its, st[:, 4], color=col, ls=LS[s], lw=1.5)
  titles = ["mean planar speed (m/s)", "moving forward (share of moving time)", "moving sideways (share of moving time)",
            "moving backward (share of moving time)", "mean |torque| / limit"]
  for ax, t in zip(axs, titles):
    ax.set_title(t, loc="left", fontsize=9, color=INK)
    ax.set_xlabel("checkpoint (training iteration)")
  for ax in axs[1:4]:
    ax.set_ylim(0, 1)
  axs[0].legend(handles=[plt.Line2D([], [], color=P, label="pursuer"), plt.Line2D([], [], color=V, label="evader")]
                + [plt.Line2D([], [], color=INK2, ls=LS[s], label=f"seed {s}") for s in SEEDS], fontsize=8)
  fig.tight_layout()
  save(fig, "behaviour")


ENDS = [("tagged", "tagged", P), ("time_out", "timeout (evader survives)", V), ("pursuer_out", "pursuer out of bounds", "#f2a07b"),
        ("pursuer_fell", "pursuer fell", "#8c1c13"), ("evader_out", "evader out of bounds", "#9cc3e6"),
        ("evader_fell", "evader fell", "#1d4e89")]


def _outcomes(s: int) -> tuple[np.ndarray, dict]:
  its, res = [], {}
  for f in glob.glob(f"{E}/outcomes/RU050_s{s}_*.json"):
    its.append(int(f.rsplit("_", 1)[1][:-5]))
    res[its[-1]] = json.load(open(f))
  return np.array(sorted(its)), res


def outcomes() -> None:
  fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
  rows = []
  for ax, s in zip(axs, SEEDS):
    its, res = _outcomes(s)
    fr = np.array([[res[i]["ended_by"].get(k, 0) / res[i]["worlds"] for k, _, _ in ENDS] for i in its]).T
    ax.stackplot(its, fr, colors=[c for *_, c in ENDS], labels=[l for _, l, _ in ENDS], lw=0)
    ax.set_xlim(its.min(), its.max())
    ax.set_ylim(0, 1)
    ax.set_title(f"seed {s}", loc="left", fontsize=10, color=INK)
    ax.set_xlabel("checkpoint (training iteration)")
    ax.grid(False)
    for i in (20000, 40000):
      if i in res:
        e, n = res[i]["ended_by"], res[i]["worlds"]
        rows.append([f"seed {s}", i] + [f"{100 * e.get(k, 0) / n:.1f}" for k, _, _ in ENDS]
                    + [f"{res[i]['episode_s_median']:.2f}", f"{res[i]['episode_s_mean']:.2f}"])
  axs[0].set_ylabel("share of 4096 self-play games")
  axs[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, reverse=True)
  save(fig, "outcomes")
  table("outcomes_table", ["run", "checkpoint"] + [f"{l} %" for _, l, _ in ENDS] + ["median game s", "mean game s"], rows)


def cross() -> None:
  rows = []
  for f in sorted(glob.glob(f"{E}/cross/*.json")):
    p, _, e, it = os.path.basename(f)[:-5].split("_")
    r = json.load(open(f))
    d, n = r["ended_by"], r["worlds"]
    rows.append([it, f"seed {p[1]}", f"seed {e[1]}", f"{100 * d.get('tagged', 0) / n:.1f}", f"{100 * d.get('time_out', 0) / n:.1f}",
                 f"{100 * (d.get('pursuer_fell', 0) + d.get('pursuer_out', 0)) / n:.1f}",
                 f"{100 * (d.get('evader_fell', 0) + d.get('evader_out', 0)) / n:.1f}", f"{r['episode_s_median']:.2f}"])
  rows.sort(key=lambda r: (int(r[0]), r[1], r[2]))
  table("cross_seed", ["checkpoint", "pursuer", "evader", "tagged %", "timeout %", "pursuer fell/out %", "evader fell/out %",
                       "median game s"], rows)


def _smooth(y: np.ndarray, k: int = 200) -> np.ndarray:
  return np.convolve(y, np.ones(k) / k, mode="valid")


def roles_trace() -> None:
  fig, axs = plt.subplots(1, 3, figsize=(15, 3.2))
  for s in SEEDS:
    tb = json.load(open(f"{E}/tb_RU050_s{s}.json"))
    for ax, tag, col in ((axs[0], "league/p_evader", INK), (axs[1], "league/evader_win_rate/league", V),
                         (axs[1], "league/evader_win_rate/selfplay", "#8a8986"), (axs[2], "Train/mean_reward", INK)):
      x, y = np.array(tb[tag]).T
      m = x <= LAST
      ys = _smooth(y[m])
      ax.plot(x[m][len(x[m]) - len(ys):], ys, color=col, ls=LS[s], lw=1.4)
  axs[0].axhspan(0.95, 1, color="#ecebe7", lw=0)
  axs[0].axhspan(0, 0.05, color="#ecebe7", lw=0)
  axs[0].set_ylim(0, 1)
  axs[0].set_title("P(learner plays evader), league envs", loc="left", fontsize=9, color=INK)
  axs[1].axhline(0.5, color=INK2, lw=0.8)
  axs[1].set_ylim(0, 1)
  axs[1].set_title("evader win rate: league (blue), self-play (grey)", loc="left", fontsize=9, color=INK)
  axs[2].set_title("training reward per episode (all seats)", loc="left", fontsize=9, color=INK)
  for ax in axs:
    ax.set_xlabel("training iteration")
  seed_legend(axs[0], "lower right")
  fig.text(0.01, -0.04, "moving average over 200 iterations; shaded: the sampling bounds [0, 0.05] and [0.95, 1]",
           fontsize=8, color=INK2)
  fig.tight_layout()
  save(fig, "roles_trace")


def evader_sampler(seeds: tuple[int, ...] = (0,)) -> None:
  """Per seed: the adaptive role sampler's P(learner plays evader) in league envs and the league evader win rate it
  steers toward 50% (step 0.05 per iteration, bounds [0.05, 0.95]). Thin: per iteration; thick: 200-iteration mean."""
  fig, axs = plt.subplots(len(seeds), 1, figsize=(9, 2.6 * len(seeds)), sharex=True, squeeze=False)
  axs = axs[:, 0]
  for ax, s in zip(axs, seeds):
    tb = json.load(open(f"{E}/tb_RU050_s{s}.json"))
    for tag, col, label in (("league/p_evader", INK, "P(learner plays evader)"),
                            ("league/evader_win_rate/league", V, "evader win rate (league envs)")):
      x, y = np.array(tb[tag]).T
      m = x <= LAST
      x, y = x[m], y[m]
      ax.plot(x, y, color=col, lw=0.3, alpha=0.25)
      ys = _smooth(y)
      ax.plot(x[len(x) - len(ys):], ys, color=col, lw=1.6, label=label)
    ax.axhspan(0.95, 1, color="#ecebe7", lw=0)
    ax.axhspan(0, 0.05, color="#ecebe7", lw=0)
    ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
    ax.set_ylim(0, 1)
    x, y = np.array(tb["league/p_evader"]).T
    y = y[x <= LAST]
    ax.set_title(f"seed {s}: at the upper bound {100 * (y >= 0.949).mean():.0f}% of iterations, "
                 f"lower bound {100 * (y <= 0.051).mean():.0f}%", loc="left", fontsize=9, color=INK)
  axs[0].legend(loc="lower right", fontsize=8, ncol=2)
  axs[-1].set_xlabel("training iteration")
  fig.text(0.01, -0.02, "thin: every iteration; thick: 200-iteration moving average; shaded: outside the sampling "
           "bounds; dotted: the 50% target", fontsize=8, color=INK2)
  fig.tight_layout()
  save(fig, "evader_sampler")


def cost() -> None:
  from omegaconf import OmegaConf
  rows = []
  for s in SEEDS:
    cfg = OmegaConf.load(f"{C}/RU050_s{s}/.hydra/config.yaml")
    n, T = cfg.num_envs, cfg.agent.num_steps_per_env
    sp = round(cfg.league.selfplay_frac * n)
    seats = 2 * sp + (n - sp)
    grad = cfg.agent.num_learning_epochs * cfg.agent.num_mini_batches
    tb = json.load(open(f"{E}/tb_RU050_s{s}.json"))
    wall = {int(a): b for a, b in tb["_wall"]}
    hours = (wall[max(k for k in wall if k <= LAST)] - wall[min(wall)]) / 3600
    rows.append([f"seed {s}", LAST, n, T, f"{LAST * n * T:.3g}", f"{LAST * seats * T:.3g}", f"{LAST * grad:.3g}",
                 f"{LAST * n * T * 0.02 / 86400 / 365:.1f}", f"{hours:.1f}", "1x RTX PRO 6000"])
  table("cost", ["run", "iterations", "envs", "steps / iter", "env steps", "learner transitions", "gradient steps",
                 "simulated years", "wall-clock h", "GPU"], rows)


ALL = {"late_vs_earlier": late_vs_earlier, "behaviour": behaviour, "outcomes": outcomes, "cross": cross,
       "roles_trace": roles_trace, "evader_sampler": evader_sampler, "cost": cost}

if __name__ == "__main__":
  for name in sys.argv[1:] or ALL:
    try:
      ALL[name]()
    except (FileNotFoundError, KeyError, ValueError) as e:
      print(f"skip {name}: {type(e).__name__}: {e}")
