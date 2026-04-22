# Gray-Scott Toy Workflow

This toy experiment is intentionally minimal and is meant for convergence checks as model size grows.

- shared defaults: `base.yaml`
- config variants: `configs/`
- shared dataset generation: `_gen_data.py`
- Phi-model training: `run_train.py`

Run outputs go under:

```text
runs/gray-scott-toy/<variant>/seed_<seed>/
```

Shared datasets are reused across configs for the same seed under:

```text
runs/gray-scott-toy/shared_datasets/default/seed_<seed>/data/
```

Each run directory contains:

- checkpoint: `psnn_phi.pt`
- `training_summary.json` with final train/validation errors

## Configs

The configs in `configs/` share the same dataset settings by default, so comparisons within a seed isolate architecture and optimization changes. The intended knobs for this toy are under `training.phi.model`, especially `embed_dim`, `width`, and `depth`.

## Test Setting Summary

The architecture sweeps in `configs/test_1/` and `configs/test_2/` do not use the small default random-sampling dataset from `base.yaml`. Instead, each sweep has its own shared dataset variant (`shared_dataset.variant: test_1` or `test_2`) and both sweep bases use the same grid-generated train/test data:

- parameter domain `theta=(f,k)`: `[0.0, 0.3] x [0.0, 0.08]`
- state domain `u=(u,v)`: `[0,1] x [0,1]`
- train dataset:
  - `N_obs: [60, 60]` with `method_theta: grid`, so there are `60 x 60 = 3600` parameter locations
  - `N_random: [20, 20]` with `method_u: grid`, so there are `20 x 20 = 400` sampled state points for each parameter location
  - the generator also appends the exact Gray-Scott steady states `U(theta)` for each parameter when they exist, then evaluates the target `Phi(theta, u)` on the combined set
- test dataset:
  - identical grid settings to the train dataset (`[60,60]` for `theta`, `[20,20]` for `u`)
- approximation metric:
  - after training, `approximation_eval` is enabled and evaluates on a denser reference grid with `theta_grid: [100,100]` and `u_grid: [50,50]`

## Architecture Sweeps

### `test_1`

`test_1` varies model depth and embedding dimension while keeping the two hidden widths fixed:

- fixed width: `width: [30, 30]`
- varied embedding dimension: `embed_dim in {2, 4, 8, 16, 32}`
- varied depth: `depth: [d, d]` with `d in {1, 2, 4, 8}`

So `test_1` is a symmetric sweep where both PSNN branches use the same number of hidden layers, and the variant name `test_1_d<depth>_n<embed_dim>` directly encodes those two knobs.

### `test_2`

`test_2` fixes `embed_dim: 8` and varies width plus an asymmetric depth assignment between the two PSNN branches:

- fixed embedding dimension: `embed_dim: 8`
- varied width: `width: [w, w]` with `w in {20, 30, 40}`
- varied layer count parameter: `d in {1, 2, 4, 8}`
- subtest `s1`: `depth: [4, d]`
- subtest `s2`: `depth: [d, 4]`

In the aggregation script, these correspond to:

- `s1`: PNN-side depth fixed at `4`, SNN-side depth varied by `d`
- `s2`: SNN-side depth fixed at `4`, PNN-side depth varied by `d`

So `test_2` measures how approximation changes when capacity is moved between the two branches, while also sweeping the shared hidden width.

## Commands

Generate data only:

```bash
python exps/gray-scott-toy/_gen_data.py --config configs/complete.yaml --seed 123
```

Generate data and train:

```bash
python exps/gray-scott-toy/run_train.py --config configs/complete.yaml --seed 123
```

Train from existing run data:

```bash
python exps/gray-scott-toy/run_train.py --config configs/complete.yaml --seed 123 --skip-data
```

## Practical usage

- Keep the dataset fixed within a seed and only change `training.phi.*` when comparing convergence.
- For the `test_1` sweep, use the configs under `configs/test_1/`.
- For the `test_2` sweep, use the configs under `configs/test_2/`.
- `training_summary.json` records train/validation MSE, and `approximation_metrics.json` records the reference-grid approximation error including `l2_error`.
- For model-comparison runs, each config at the same seed reuses the same shared dataset automatically.
