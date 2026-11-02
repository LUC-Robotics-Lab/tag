"""2 x 4 paper figure: top, motion trails (scripts/paper/motion_trail.py: one world's first episode, deterministic,
reset seed 0, 8 robot poses fading in); bottom, a close-up of the same episode at contact, the frame where the two
robots' bases are closest, from a camera 3.2 m in front of their midpoint. Pursuer red, evader light blue.

The close-up is the first rollout's exact simulator state at that frame (qpos, qvel of the world), copied into a
second env whose camera is moved there and rendered without stepping (re-simulating drifts: GPU physics is not
bit-exact).

  python scripts/paper/contact_figure.py [--out logs/plots/paper/contact_figure] [--ghosts 8]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motion_trail as mt  # noqa: E402  (also sets up scripts/ on the path)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

T, G, C = "logs/mofo/tag", "logs/mofo/goal/goal_scratch_lim", "logs/mofo/tag/tag_chase_adapt"
PANELS = [  # (label, run dir whose game is played, robot-seat checkpoint, opponent-seat checkpoint or None, overrides)
  ("W1F_s0", f"{T}/tag_walk/W1F_s0", f"{T}/tag_walk/W1F_s0/model_31000.pt", None, []),
  ("L_s0", f"{T}/tag_scratch_lim/L_s0", f"{T}/tag_scratch_lim/L_s0/model_19999.pt", None, []),
  ("L_s2", f"{T}/tag_scratch_lim/L_s2", f"{T}/tag_scratch_lim/L_s2/model_19999.pt", None, []),
  ("goal G2_s0 (P) vs RU050_s0 (E)", f"{C}/RU050_s0", f"{G}/G2_s0/model_19999.pt", f"{C}/RU050_s0/model_43500.pt",
   ["rules.learner_role=pursuer"]),
]
CLOSE = {"back": 3.2, "up": 0.9, "fovy": 40.0}  # From 3.2 m in front, 0.9 m above the look-at point (0.6 m high).


def closeup(run: str, sets: list[str], envs: int, qpos: np.ndarray, qvel: np.ndarray, mid: np.ndarray):
  """RGB [H, W, 3] and robot mask [H, W] of world 0 put in state (qpos, qvel), seen from CLOSE in front of mid."""
  import torch

  cam = mt.camera(run, {**CLOSE, "target": [float(mid[0]), float(mid[1]), 0.6]})
  env = mt.plot_states.load_env(run, envs, camera=cam, overrides=sets)
  env.reset(seed=0)  # Roles (and so the robots' colors) as in the first rollout.
  d = env.sim.data
  d.qpos[0] = torch.as_tensor(qpos, device=env.device)
  d.qvel[0] = torch.as_tensor(qvel, device=env.device)
  env.sim.forward()
  env.sim.sense()
  return env.scene["world_cam"].data.rgb[0].cpu().numpy(), mt.robot_mask(env, 0, mt.robot_geoms(env)).cpu().numpy()


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default="logs/plots/paper/contact_figure")
  ap.add_argument("--ghosts", type=int, default=8)
  ap.add_argument("--envs", type=int, default=16)
  a = ap.parse_args()
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  tops, bottoms, titles, subs = [], [], [], []
  for k, (label, run, ckpt, opp, sets) in enumerate(PANELS):
    cache = f"{a.out}_{k}.npz"
    if os.path.exists(cache):
      z = np.load(cache)
      rgb, mask, dt, pos, crgb, cmask = (z[n] for n in ("rgb", "mask", "dt", "pos", "crgb", "cmask"))
      dt, t = float(dt), int(z["t"])
    else:
      rgb, mask, dt, pos, qpos, qvel = mt.episode(run, ckpt, 0, a.envs, opp, sets)
      t = int(np.linalg.norm(pos[:, 0, :2] - pos[:, 1, :2], axis=-1).argmin())  # Contact: bases closest.
      crgb, cmask = closeup(run, sets, a.envs, qpos[t], qvel[t], pos[t, :, :2].mean(0))
      print(f"{label}: contact at step {t} ({t * dt:.2f} s of {len(rgb) * dt:.2f} s)")
      np.savez_compressed(cache, rgb=rgb, mask=mask, dt=dt, pos=pos, crgb=crgb, cmask=cmask, t=t)
    tops.append(mt.crop(mt.trail(rgb, mask, a.ghosts), mask))
    bottoms.append(mt.crop(crgb, cmask[None], pad=0.12))
    titles.append(f"{label} ({len(rgb) * dt:.1f} s)")
    subs.append(f"contact, t = {t * dt:.2f} s")
    for img, kind in ((tops[-1], "trail"), (bottoms[-1], "contact")):
      plt.imsave(f"{a.out}_{k}_{kind}.png", img)
  fig, axes = plt.subplots(2, len(PANELS), figsize=(4 * len(PANELS), 6.4))
  for j in range(len(PANELS)):
    for i, (img, title) in enumerate(((tops[j], titles[j]), (bottoms[j], subs[j]))):
      axes[i, j].imshow(img)
      axes[i, j].set_axis_off()
      axes[i, j].set_title(title, fontsize=10)
  fig.subplots_adjust(wspace=0.03, hspace=0.12, left=0.005, right=0.995, top=0.95, bottom=0)
  for ext in ("png", "pdf"):
    fig.savefig(f"{a.out}.{ext}", dpi=300, bbox_inches="tight", transparent=True)
  print(f"saved {a.out}.png/.pdf and one PNG per panel")


if __name__ == "__main__":
  main()
