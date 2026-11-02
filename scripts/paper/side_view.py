"""Low side-view close-ups of a few worlds' first episode (deterministic, reset seed 0), cropped to follow
the robots, at half speed; plus a contact sheet and how every world's episode ended.

  python scripts/paper/side_view.py RUN_DIR CKPT OUT_STEM [--worlds 3] [--steps 300] [--opp CKPT] [--set k=v ...]

--opp: the opponent seat plays this checkpoint instead of CKPT (cross-play); --set: dotlist overrides of the
run's saved config, e.g. rules.learner_role=pursuer to put CKPT in the pursuer seat in every world.

--worlds 0: no camera and no video, only the .json (e.g. --envs 4096 for the preregistered evaluation).
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import plot_states  # noqa: E402  (sets MUJOCO_GL)

import imageio_ffmpeg  # noqa: E402
import mediapy  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

CAM = {"back": 8.0, "up": 1.6, "width": 1280, "height": 720, "fovy": 40.0, "data_types": ["rgb"]}
CROP_H, CROP_W = 420, 560


def pixel(pts: torch.Tensor) -> torch.Tensor:
  """Pixel (row, col) of env-frame points [n, 3] through the camera (as eval.project, nothing dropped)."""
  c = torch.tensor([0.0, -CAM["back"], CAM["up"]])
  fwd = -c / c.norm()
  right = torch.tensor([1.0, 0.0, 0.0])
  up = torch.linalg.cross(right, fwd)
  f = CAM["height"] / 2 / math.tan(math.radians(CAM["fovy"]) / 2)
  p = pts.cpu().float() - c
  z = p @ fwd
  return torch.stack([CAM["height"] / 2 - f * (p @ up) / z, CAM["width"] / 2 + f * (p @ right) / z], -1)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("run")
  ap.add_argument("ckpt")
  ap.add_argument("out")
  ap.add_argument("--worlds", type=int, default=3)
  ap.add_argument("--envs", type=int, default=256)
  ap.add_argument("--steps", type=int, default=300, help="Cap per episode (goal task).")
  ap.add_argument("--opp", default=None, help="Opponent-seat checkpoint (default: CKPT, self-play).")
  ap.add_argument("--set", nargs="*", default=[], help="Dotlist overrides of the run's config.")
  ap.add_argument("--back", type=float, default=CAM["back"], help="Camera distance behind the arena center (m).")
  ap.add_argument("--up", type=float, default=CAM["up"], help="Camera height (m).")
  ap.add_argument("--scale", type=float, default=1.0, help="Multiplies the render resolution and the crop.")
  a = ap.parse_args()
  global CROP_H, CROP_W
  CAM.update(back=a.back, up=a.up, width=round(CAM["width"] * a.scale), height=round(CAM["height"] * a.scale))
  CROP_H, CROP_W = round(CROP_H * a.scale), round(CROP_W * a.scale)
  from mofo.opponent import load_policy
  from mofo.tasks.tag import role

  env = plot_states.load_env(a.run, a.envs, camera=CAM if a.worlds else None, overrides=a.set)
  dev, n = env.device, env.num_envs
  policy = load_policy(a.ckpt, dev)
  opp = load_policy(a.opp, dev) if a.opp else policy
  ents = ["robot"] + (["opponent"] if "opponent" in env.scene.entities else [])
  obs, _ = env.reset(seed=0)
  done = torch.zeros(n, dtype=torch.bool, device=dev)
  length = torch.zeros(n, dtype=torch.long, device=dev)
  ends: dict[str, float] = {}
  frames, centers, pelvis = [], [], []
  steps = min(env.max_episode_length + 1, a.steps) if len(ents) == 1 else env.max_episode_length + 1
  with torch.no_grad():
    for _ in range(steps):
      act = policy(obs["actor"]) if len(ents) == 1 else torch.cat([policy(obs["actor"]), opp(obs["opponent"])], -1)
      is_p = None if len(ents) == 1 else role(env) > 0  # The robot seat is the pursuer (before any reset).
      obs, _, term, trunc, _ = env.step(act)
      new = (term | trunc) & ~done
      tm = env.termination_manager
      for name in tm.active_terms:  # Why each world's first episode ended (per-term flags survive the reset).
        t = tm.get_term(name) & new
        if is_p is not None and name in ("fell_over", "opponent_fell_over", "robot_out", "opponent_out"):
          own = is_p if name in ("fell_over", "robot_out") else ~is_p
          kind = "fell" if "fell" in name else "out"
          for who, m in (("pursuer", own), ("evader", ~own)):
            ends[f"{who}_{kind}"] = ends.get(f"{who}_{kind}", 0) + (t & m).sum().item()
        else:
          ends[name] = ends.get(name, 0) + t.sum().item()
      length += (~done).long()
      done |= term | trunc
      pos = torch.stack([env.scene[e].data.root_link_pos_w[:a.worlds] - env.scene.env_origins[:a.worlds] for e in ents], 1)
      centers.append(pixel(pos.mean(1)).numpy())
      pelvis.append(pos[..., 2].cpu().numpy())
      if a.worlds:
        frames.append(env.scene["world_cam"].data.rgb[:a.worlds].cpu().numpy())
      if done.all():
        break
  ends = {k: v for k, v in ends.items() if v}
  L = length.cpu().numpy()
  stats = {"worlds": n, "episode_s_mean": float(L.mean() * env.step_dt), "episode_s_median": float(np.median(L) * env.step_dt),
           "ended_by": ends, "pelvis_min_m": [float(np.array(pelvis)[: L[w], w].min()) for w in range(a.worlds)]}
  print(json.dumps(stats, indent=2))
  json.dump(stats, open(a.out + ".json", "w"), indent=2)
  if not a.worlds:
    return
  frames, centers = np.array(frames), np.array(centers)  # [T, worlds, H, W, 3], [T, worlds, 2]
  clips, sheet = [], []
  for w in range(a.worlds):
    T = min(int(L[w]) + 5, len(frames))
    c = centers[:T, w].copy()
    for t in range(1, T):  # Smooth the crop window.
      c[t] = 0.7 * c[t - 1] + 0.3 * c[t]
    clip = []
    for t in range(T):
      r = int(np.clip(c[t, 0] - CROP_H * 0.6, 0, CAM["height"] - CROP_H))
      q = int(np.clip(c[t, 1] - CROP_W / 2, 0, CAM["width"] - CROP_W))
      clip.append(frames[t, w, r:r + CROP_H, q:q + CROP_W])
    clips += clip + [np.zeros_like(clip[0])] * 10
    idx = np.linspace(0, T - 1, 6).astype(int)
    sheet.append(np.concatenate([clip[i] for i in idx], 1))
  mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
  mediapy.write_video(a.out + ".mp4", clips, fps=0.5 / env.step_dt)
  mediapy.write_image(a.out + "_sheet.png", np.concatenate(sheet, 0))
  print(f"saved {a.out}.mp4 ({len(clips)} frames, half speed), {a.out}_sheet.png")


if __name__ == "__main__":
  main()
