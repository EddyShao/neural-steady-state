# Gray-Scott workflow

This experiment now follows the same structure as `feedback_loop`:

- configs: `configs/`
- shared defaults: `base.yaml`
- data generation: `_gen_data.py`
- training: `run_train.py`
- bifurcation plotting: `run_bifur.py`
- test-observation evaluation: `run_eval.py`

All outputs go under:

```text
runs/gray_scott/<variant>/seed_<seed>/
```

That directory contains:

- checkpoints: `psnn_phi.pt`, `psnn_numsol.pt`, `psnn_stability_cls.pt`
- generated data in `data/`
- bifurcation outputs in `bifur_strict_runs/` or `bifur_flexible_runs/`
- evaluation outputs in `test_observation_eval_strict/` or `test_observation_eval_flexible/`

## Configs

Only the complete-data variant is scaffolded right now:

- `configs/complete.yaml`

`base.yaml` keeps the shared defaults.

## Commands

Generate data only:

```bash
python exps/gray_scott/_gen_data.py --config configs/complete.yaml --seed 123
```

Generate data and train:

```bash
python exps/gray_scott/run_train.py --config configs/complete.yaml --seed 123
```

Train from existing run data:

```bash
python exps/gray_scott/run_train.py --config configs/complete.yaml --seed 123 --skip-data
```

Run strict bifurcation plotting:

```bash
python exps/gray_scott/run_bifur.py --config configs/complete.yaml --seed 123 --mode strict -- --k 0.05
```

Run flexible bifurcation plotting:

```bash
python exps/gray_scott/run_bifur.py --config configs/complete.yaml --seed 123 --mode flexible -- --k 0.05
```

Evaluate the test-observation set with the strict method:

```bash
python exps/gray_scott/run_eval.py --config configs/complete.yaml --seed 123 --mode strict -- --device cpu
```

Evaluate the test-observation set with the flexible method:

```bash
python exps/gray_scott/run_eval.py --config configs/complete.yaml --seed 123 --mode flexible -- --device cpu
```

## Practical usage

- `--mode strict` uses the count classifier plus the strict locator.
- `--mode flexible` uses only the learned landscape geometry to infer the centers.
- The bifurcation scripts sweep `f` while holding `k` fixed.
