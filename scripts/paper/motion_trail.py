"""1 x N figure of motion trails: per panel, one world's first episode of a checkpoint in its own run's game
(self-play, deterministic actions, reset seed 0), with the robots of --ghosts evenly spaced frames overlaid on
one image, fading from translucent (start) to opaque (the final frame). Pursuer red, evader light blue.

Robots are cut out of each frame by the camera's segmentation (geoms of the robot/ and opponent/ bodies); the floor comes
from the frames where nothing covers it. Fixed world camera behind the arena center, scaled to the arena size.
Panels are cropped to the motion (common aspect) and also saved one by one.

  python scripts/paper/motion_trail.py LABEL=RUN_DIR:CKPT [...] --out OUT_STEM [--ghosts 8] [--world 0]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import plot_states  # noqa: E402  (sets MUJOCO_GL)

import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

W, H, ASPECT = 1920, 1080, 4 / 3  # Render size; panel crop aspect (width / height).


def robot_mask(env, world: int, robot_geom: torch.Tensor) -> torch.Tensor:
  """[H, W] bool: world_cam pixels showing a geom of the robot/ or opponent/ bodies."""
  seg = env.scene["world_cam"].data.segmentation[world]
  ids, kinds = seg[..., 0], seg[..., 1]
  return (kinds == int(mujoco.mjtObj.mjOBJ_GEOM)) & (ids >= 0) & robot_geom[ids.clamp_min(0)]


def robot_geoms(env) -> torch.Tensor:
  m = env.sim.mj_model
  body = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in m.geom_bodyid]
  return torch.tensor([n.startswith(("robot/", "opponent/")) for n in body], device=env.device)


def camera(run: str, cam: dict | None = None) -> dict:
  """The default whole-arena camera for the run's arena, with ``cam`` overriding it."""
  half = OmegaConf.load(os.path.join(run, ".hydra", "config.yaml")).rules.arena_half_extent
  back = 1.4 * half + 3.5  # Whole arena in view at fovy 40, seen from ~25 deg above.
  return {"back": back, "up": 0.45 * back, "width": W, "height": H, "fovy": 40.0, "data_types": ["rgb", "segmentation"],
          **(cam or {})}


def episode(run: str, ckpt: str, world: int, envs: int, opp: str | None = None, overrides: list[str] | None = None,
            cam: dict | None = None):
  """RGB frames [T, H, W, 3] and robot masks [T, H, W] of world `world`'s first episode, its step dt, and the robots'
  base positions [T, 2, 3] (robot, opponent; arena frame), and world's sim state [T, nq], [T, nv]. opp: the opponent
  seat's checkpoint (default: ckpt); overrides: dotlist on the run's config; cam: overrides of the default camera."""
  from mofo.opponent import load_policy

  env = plot_states.load_env(run, envs, camera=camera(run, cam), overrides=overrides)
  policy = load_policy(ckpt, env.device)
  opp_policy = load_policy(opp, env.device) if opp else policy
  two = "opponent" in env.scene.entities
  robot_geom = robot_geoms(env)
  obs, _ = env.reset(seed=0)
  rgb, mask, pos, qpos, qvel = [], [], [], [], []
  with torch.no_grad():
    for _ in range(env.max_episode_length + 1):
      rgb.append(env.scene["world_cam"].data.rgb[world].cpu().numpy())
      mask.append(robot_mask(env, world, robot_geom).cpu().numpy())
      qpos.append(env.sim.data.qpos[world].cpu().numpy())
      qvel.append(env.sim.data.qvel[world].cpu().numpy())
      pos.append(torch.stack([env.scene[e].data.root_link_pos_w[world] - env.scene.env_origins[world]
                              for e in (("robot", "opponent") if two else ("robot",))]).cpu().numpy())
      act = torch.cat([policy(obs["actor"]), opp_policy(obs["opponent"])], -1) if two else policy(obs["actor"])
      obs, _, term, trunc, _ = env.step(act)
      if (term | trunc)[world]:
        break
  return np.array(rgb), np.array(mask), env.step_dt, np.array(pos), np.array(qpos), np.array(qvel)


def trail(rgb: np.ndarray, mask: np.ndarray, ghosts: int) -> np.ndarray:
  """Composite: robot-free background, then the sampled frames' robots oldest first, alpha 0.25 -> 1."""
  free = ~mask
  bg = (rgb * free[..., None]).sum(0) / np.maximum(free.sum(0), 1)[..., None]
  bg = np.where(free.any(0)[..., None], bg, rgb[0])  # Never uncovered: first frame.
  out = bg.astype(np.float32)
  idx = np.linspace(0, len(rgb) - 1, ghosts).round().astype(int)
  for k, t in enumerate(idx):
    a = (0.25 + 0.75 * (k / max(ghosts - 1, 1)) ** 1.5) * mask[t][..., None]
    out = out * (1 - a) + rgb[t] * a
  return out.clip(0, 255).astype(np.uint8)


def crop(img: np.ndarray, mask: np.ndarray, pad: float = 0.15) -> np.ndarray:
  """Bounding box of every robot pixel, padded and grown to ASPECT, kept inside the image."""
  ys, xs = np.nonzero(mask.any(0))
  y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
  h, w = (y1 - y0) * (1 + 2 * pad), (x1 - x0) * (1 + 2 * pad)
  h, w = max(h, w / ASPECT), max(w, h * ASPECT)
  h, w = min(h, H), min(w, W)
  w = min(w, h * ASPECT)
  h = w / ASPECT
  cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
  r = int(np.clip(cy - h / 2, 0, H - h))
  c = int(np.clip(cx - w / 2, 0, W - w))
  return img[r:r + int(h), c:c + int(w)]


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("panels", nargs="+", help="LABEL=RUN_DIR:CKPT")
  ap.add_argument("--out", required=True)
  ap.add_argument("--ghosts", type=int, default=8)
  ap.add_argument("--world", type=int, default=0)
  ap.add_argument("--envs", type=int, default=16)
  ap.add_argument("--no-labels", action="store_true")
  a = ap.parse_args()
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  imgs, labels = [], []
  for p in a.panels:
    label, spec = p.split("=", 1)
    run, ckpt = spec.rsplit(":", 1)
    cache = f"{a.out}_{label}_frames.npz"  # Raw frames: re-composite without the GPU.
    if os.path.exists(cache):
      z = np.load(cache)
      rgb, mask, dt = z["rgb"], z["mask"], float(z["dt"])
    else:
      rgb, mask, dt, *_ = episode(run, ckpt, a.world, a.envs)
      np.savez_compressed(cache, rgb=rgb, mask=mask, dt=dt)
    img = crop(trail(rgb, mask, a.ghosts), mask)
    plt.imsave(f"{a.out}_{label}.png", img)
    imgs.append(img)
    labels.append(f"{label} ({len(rgb) * dt:.1f} s)")
    print(label, run, os.path.basename(ckpt), f"{len(rgb)} steps = {len(rgb) * dt:.2f} s", flush=True)
  fig, axes = plt.subplots(1, len(imgs), figsize=(4 * len(imgs), 3.3))
  for ax, img, label in zip(np.atleast_1d(axes), imgs, labels):
    ax.imshow(img)
    ax.set_axis_off()
    if not a.no_labels:
      ax.set_title(label, fontsize=10)
  fig.subplots_adjust(wspace=0.03, left=0.005, right=0.995, top=0.9 if not a.no_labels else 1, bottom=0)
  for ext in ("png", "pdf"):
    fig.savefig(f"{a.out}.{ext}", dpi=300, bbox_inches="tight", transparent=True)
  print(f"saved {a.out}.png/.pdf and one PNG per panel")


if __name__ == "__main__":
  main()
