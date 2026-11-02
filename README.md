# motionforge

Two Unitree G1 humanoids learn to play tag in simulation: whole-body control policies trained with PPO against a
league of past selves, built on [mjlab](https://github.com/mujocolab/mjlab) and rsl-rl. A single-robot goal-reaching
task serves as a dense-reward baseline.

## Install

Requires [uv](https://docs.astral.sh/uv/), Python 3.11 and a recent NVIDIA GPU: the locked PyTorch build has no kernels
for older cards such as the V100 or RTX 2080. Tested on the L40S; training used the RTX PRO 6000.

```bash
uv sync
uv run scripts/mofo_train.py check   # Environment sanity tests.
uv run -m mofo.league_test           # League self-test (CPU).
```

## Train

Configuration uses [Hydra](https://hydra.cc): `config/` holds the defaults, and `config/experiment/` the experiments.

```bash
uv run scripts/mofo_train.py tag +experiment=tag_chase_adapt seed=0 cmd.logdir=logs/mofo/tag/tag_chase_adapt/run_s0
uv run scripts/mofo_train.py tag +experiment=goal_scratch_lim seed=0 cmd.logdir=logs/mofo/goal/goal_scratch_lim/run_s0
```

Training logs to Weights & Biases by default; add `wandb.mode=disabled` to log to TensorBoard only. Checkpoints go to
`cmd.logdir` as `model_<iteration>.pt`. Resuming a run is not supported.

| Experiment | Game |
|---|---|
| `tag_scratch_lim` | Tag from scratch: 8 x 8 m arena, a timeout is a draw |
| `tag_chase` | A game the evader can win: 16 x 16 m arena, the evader wins at the 10 s timeout, falls and exits lose |
| `tag_chase_adapt` | `tag_chase` with role sampling adapted toward a 50% win rate |
| `goal_scratch_lim` | One G1 reaching goals under a dense reward, no opponent |

Other experiments are variants of these; each file's header describes it. Trained checkpoints are not included, so
`ru025_goalopp`, `s17_curr` and `s17_both`, which use one as an opponent, need their paths pointed at your own.

## Evaluate and plot

```bash
uv run scripts/mofo_train.py eval num_envs=512 cmd.policy=CKPT   # Win rates and episode statistics.
uv run scripts/eval.py policy=CKPT                               # Self-play video with every world overlaid.
uv run scripts/plot_elo.py RUN_DIR                               # Elo ratings of a run's checkpoints.
uv run scripts/plot_states.py CKPT                               # State-visitation distributions.
```

`scripts/wandb_plots.py` logs these per checkpoint to W&B, and can follow a run as it trains.

## Paper figures

`scripts/paper/` regenerates the paper's figures and tables, writing to `logs/plots/paper/`. The scripts expect the
training runs and their evaluations under `logs/`, at the paths named in each script, so they need your own runs.
Each file's header gives its inputs and usage.

| Script | Figure |
|---|---|
| `contact_figure.py` (with `motion_trail.py`) | Motion trails and close-ups at contact |
| `paper_figs.py` | Elo over training: mean of seeds, shaded from min to max |
| `elo_goal_vs_tag.py` | Pursuer Elo of the goal policy against tag self-play |
| `role_heatmap.py`, `role_matrix_coarse.py` | Pursuer and evader win rates between checkpoints |
| `vel_pdf_split.py` | Planar velocity density, one vector PDF per role |
| `ru050_figs.py` (with `ru050_evals.sh`, `side_view.py`, `tb_extract.py`) | RU050 outcomes, behaviour, role sampling and cost |

## Layout

```
src/mofo/tasks/   Tag and goal environments
src/mofo/         League (PFSP), opponents, self-play, env construction, rolling evaluator
scripts/          Training, evaluation and plotting entry points; paper/ has the paper's figures, slurm/ example jobs
config/           Hydra configs
docs/             Architecture notes; tag_chase preregistration
reports/          17 h sprint feasibility audit and preregistration
```

The preregistrations and reports are kept as written and refer to some analysis tools and results that are not in
this repository.
