# Feedback-loop workflow

This experiment now has one canonical directory:

- configs: [configs/](/root/neural-steady-state/exps/feedback_loop/configs)
- shared defaults: [base.yaml](/root/neural-steady-state/exps/feedback_loop/base.yaml)
- data generation: [_gen_data.py](/root/neural-steady-state/exps/feedback_loop/_gen_data.py)
- training: [run_train.py](/root/neural-steady-state/exps/feedback_loop/run_train.py)
- bifurcation plotting: [run_bifur.py](/root/neural-steady-state/exps/feedback_loop/run_bifur.py)
- test-observation evaluation: [run_eval.py](/root/neural-steady-state/exps/feedback_loop/run_eval.py)

All outputs go under:

- [runs/feedback_loop](/root/neural-steady-state/runs/feedback_loop)

Each run is isolated as:

```text
runs/feedback_loop/<variant>/seed_<seed>/
```

That directory contains:

- checkpoints: `psnn_phi.pt`, `psnn_numsol.pt`, `psnn_stability_cls.pt`
- generated data in `data/`
- bifurcation outputs in `bifur_strict_runs/` or `bifur_flexible_runs/`
- evaluation outputs in `test_observation_eval_strict/` or `test_observation_eval_flexible/`

## Configs

Edit [base.yaml](/root/neural-steady-state/exps/feedback_loop/base.yaml) for shared defaults.

Edit the variant files in [configs/](/root/neural-steady-state/exps/feedback_loop/configs) for differences:

- [baseline.yaml](/root/neural-steady-state/exps/feedback_loop/configs/baseline.yaml)
- [per_center.yaml](/root/neural-steady-state/exps/feedback_loop/configs/per_center.yaml)
- [minor.yaml](/root/neural-steady-state/exps/feedback_loop/configs/minor.yaml)
- [major.yaml](/root/neural-steady-state/exps/feedback_loop/configs/major.yaml)
- [minor_per_center.yaml](/root/neural-steady-state/exps/feedback_loop/configs/minor_per_center.yaml)
- [major_per_center.yaml](/root/neural-steady-state/exps/feedback_loop/configs/major_per_center.yaml)

## Commands

Generate data only:

```bash
python exps/feedback_loop/_gen_data.py --config configs/major.yaml --seed 42
```

Generate data and train:

```bash
python exps/feedback_loop/run_train.py --config configs/major.yaml --seed 42
```

Train from existing run data:

```bash
python exps/feedback_loop/run_train.py --config configs/major.yaml --seed 42 --skip-data
```

Run strict bifurcation plotting:

```bash
python exps/feedback_loop/run_bifur.py --config configs/major.yaml --seed 42 --mode strict -- --L-cut 0.35
```

Run flexible bifurcation plotting:

```bash
python exps/feedback_loop/run_bifur.py --config configs/major.yaml --seed 42 --mode flexible -- --L-cut 0.35
```

Evaluate the test-observation set with the strict method:

```bash
python exps/feedback_loop/run_eval.py --config configs/major.yaml --seed 42 --mode strict -- --device cpu
```

Evaluate the test-observation set with the flexible method:

```bash
python exps/feedback_loop/run_eval.py --config configs/major.yaml --seed 42 --mode flexible -- --device cpu
```

Evaluate a random subset of the test-observation set:

```bash
python exps/feedback_loop/run_eval.py --config configs/major.yaml --seed 42 --mode strict -- --sample-size 100 --sample-seed 7 --device cpu
```

Run all configured variants for one or more seeds:

```bash
bash runall.sh 42
bash runall.sh 42 43 44
```

## Practical usage

To try another random seed, change only `--seed`:

```bash
python exps/feedback_loop/run_train.py --config configs/baseline.yaml --seed 43
```

To try another experiment setting, change only `--config`:

```bash
python exps/feedback_loop/run_train.py --config configs/per_center.yaml --seed 42
```

Use CLI args for bifurcation settings such as `--L-cut`, `--alpha1-min`, `--alpha1-max`, `--alpha1-steps`, and similar plotting/sweep parameters.

For evaluation:

- `--mode strict` uses the count classifier plus the strict locator in [_eval_strict.py](/root/neural-steady-state/exps/feedback_loop/_eval_strict.py)
- `--mode flexible` uses the flexible locator in [_eval_flexible.py](/root/neural-steady-state/exps/feedback_loop/_eval_flexible.py)
- `--limit N` evaluates the first `N` observations
- `--sample-size N --sample-seed S` evaluates a random subset of `N` observations without replacement

Each evaluation writes:

- count accuracy for the evaluated thetas
- mean `L^2` error between lexicographically sorted predicted and true solutions over correctly counted thetas
- per-theta details as an `.npz` alongside a JSON metrics summary
