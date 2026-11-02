"""Copy a few training-log scalars of each run (its tensorboard event file) to logs/plots/ru050/tb_RUN.json:
{tag: [[step, value], ...]}. The event files are GBs, so this is done once and the plots read the JSON.

  .venv/bin/python scripts/paper/tb_extract.py RUN_DIR [RUN_DIR ...]
"""

import json
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = ["league/p_evader", "league/evader_win_rate", "league/evader_win_rate/league", "league/evader_win_rate/selfplay",
        "Train/mean_reward", "Train/mean_episode_length", "Policy/mean_std", "Perf/total_fps",
        "Perf/collection_time", "Perf/learning_time", "Episode_Termination/tagged", "Episode_Termination/time_out",
        "Episode_Termination/fell_over", "Episode_Termination/opponent_fell_over", "Episode_Termination/robot_out",
        "Episode_Termination/opponent_out", "Episode_Reward/tag", "league/size"]

for run in sys.argv[1:]:
  acc = EventAccumulator(run, size_guidance={"scalars": 0, "tensors": 1, "histograms": 1, "images": 1})
  acc.Reload()
  have = set(acc.Tags()["scalars"])
  out = {t: [[e.step, e.value] for e in acc.Scalars(t)] for t in TAGS if t in have}
  out["_wall"] = [[e.step, e.wall_time] for e in acc.Scalars("Perf/total_fps")]
  path = f"logs/plots/ru050/tb_{os.path.basename(os.path.normpath(run))}.json"
  json.dump(out, open(path, "w"))
  print("saved", path, len(out["Train/mean_reward"]), "points", flush=True)
