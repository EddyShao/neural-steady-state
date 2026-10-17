# Data-Driven Discovery of Steady States in Parametric ODE Systems

This repository implements a three-stage pipeline for learning and locating steady states of parametric ODE systems from data. The codebase is organized around two example systems:

- Gray-Scott
- Feedback loop

The learned workflow has three parts:

1. Data generation and preprocessing
2. Auxiliary networks
3. Locating solutions

The repository corresponds to the paper: https://arxiv.org/abs/2312.10315

## Pipeline Overview

### 1. Data Generation and Preprocessing

For each parameter value `theta`, the experiment code first generates observations of steady states `U(theta)`. Those observations are then converted into training data for the PSNN by sampling state-space points `u` and evaluating a target field `Phi(theta, u)` built from the observed steady states.

This stage also supports synthetic incompleteness: some true solutions can be removed from the observation set before preprocessing. That makes it possible to study how the downstream models and locaters behave when observations are incomplete.

Outputs of this stage include:

- observation files such as `*_obs_train.pkl`, `*_obs_test.pkl`, and sometimes `*_obs_missing.pkl`
- PSNN regression data such as `*_data_train.npz` and `*_data_test.npz`

### 2. Main and Auxiliary Networks

The central learned model is the PSNN, and it is supported by two auxiliary networks:

- PSNN: the main network, which learns the surrogate field `Phi(theta, u)` and defines the landscape used for solution search
- Counting classifier: an auxiliary network that predicts how many steady states exist for a given `theta`
- Stability classifier: an auxiliary network that predicts whether a candidate `(theta, u)` is stable

These networks are later combined during the locating stage.

### 3. Locating Solutions

Given a parameter `theta`, the locator uses the learned PSNN field to search the state domain `D` for peaks.

The basic logic is:

1. Sample points in the state domain.
2. Evaluate the learned field `Phi(theta, u)`.
3. Keep points with large values relative to the current maximum.
4. Cluster the retained points to obtain candidate steady states.
5. Refine locally around those candidates with adaptive resampling.
6. Apply the stability classifier to the final centers.

There are two locator modes:

- Flexible locator: does not use the counting classifier. It infers the number of clusters from the sampled high-value region, then merges centers if there is no valley between them.
- Strict locator: uses the counting classifier first, then reclusters with a fixed number of centers during adaptive refinement.

In both modes, the PSNN gives the search landscape and the stability classifier labels the stability of located solutions.

## Repository Structure

### Core Learning Modules

- `psnn/nets.py`: neural network definitions for PSNN, the theta-to-count classifier, and the stability classifier.
- `psnn/datasets.py`: dataset and dataloader wrappers for PSNN regression data and observation-derived classification data.
- `psnn/trainer.py`: training loops for the three models.
- `psnn/run_train.py`: shared training entrypoint that reads config, loads generated data, trains enabled models, and saves checkpoints.
- `psnn/loaders.py`: checkpoint loading helpers and callable wrappers for PSNN, count prediction, and stability prediction.
- `psnn/config.py`: YAML config loading, merging, and path resolution utilities.
- `psnn/utils.py`: model metadata helpers used when saving and reloading checkpoints.

### Locating Modules

- `locater/flexible.py`: adaptive locator without the count classifier. It samples the PSNN landscape, chooses cluster count from the data, merges centers based on valley checks, and refines candidates locally.
- `locater/strict.py`: adaptive locator with the count classifier. It fixes the number of clusters using the predicted count, merges overlapping regions only for sampling, and reclusters back to the prescribed number during refinement.

## Experiment Layout

Each system has its own experiment directory under `exps/`.

### Stage 1: Data Generation and Preprocessing

Gray-Scott:

- `exps/gray_scott/_gen_data.py`: defines the Gray-Scott parameter box, state domain, steady-state solver, target `Phi` construction, synthetic incompleteness, and export of training/test datasets.
- `exps/gray_scott/configs/*.yaml`: experiment variants such as complete and incomplete-data settings.
- `exps/gray_scott/base.yaml`: shared defaults for domain bounds, sampling, model training, and output paths.

Feedback loop:

- `exps/feedback_loop/_gen_data.py`: defines the feedback-loop parameter box, state domain, steady-state solver, target `Phi` construction, synthetic incompleteness, and export of training/test datasets.
- `exps/feedback_loop/configs/*.yaml`: experiment variants such as complete and incomplete-data settings.
- `exps/feedback_loop/base.yaml`: shared defaults for domain bounds, sampling, model training, and output paths.

### Stage 2: Auxiliary Network Training

Gray-Scott:

- `exps/gray_scott/run_train.py`: convenience entrypoint that generates data for one config/seed pair and then trains PSNN, counting, and stability models using the shared trainer.

Feedback loop:

- `exps/feedback_loop/run_train.py`: same orchestration as above for the feedback-loop system.

Saved checkpoints typically include:

- `psnn_phi.pt`
- `psnn_numsol.pt`
- `psnn_stability_cls.pt`

### Stage 3: Locating, Evaluation, and Bifurcation Runs

Gray-Scott:

- `exps/gray_scott/run_eval.py`: evaluates the learned locator on held-out observations, in either `flexible` or `strict` mode.
- `exps/gray_scott/_eval_common.py`: shared evaluation utilities.
- `exps/gray_scott/_eval_flexible.py`: evaluation logic for the flexible locator.
- `exps/gray_scott/_eval_strict.py`: evaluation logic for the strict locator.
- `exps/gray_scott/run_bifur.py`: wrapper for bifurcation-style sweeps and plots.
- `exps/gray_scott/_bifur_flexible.py`: flexible bifurcation runner.
- `exps/gray_scott/_bifur_strict.py`: strict bifurcation runner.

Feedback loop:

- `exps/feedback_loop/run_eval.py`: evaluates the learned locator on held-out observations, in either `flexible` or `strict` mode.
- `exps/feedback_loop/_eval_common.py`: shared evaluation utilities.
- `exps/feedback_loop/_eval_flexible.py`: evaluation logic for the flexible locator.
- `exps/feedback_loop/_eval_strict.py`: evaluation logic for the strict locator.
- `exps/feedback_loop/run_bifur.py`: wrapper for bifurcation-style sweeps and plots.
- `exps/feedback_loop/_bifur_flexible.py`: flexible bifurcation runner.
- `exps/feedback_loop/_bifur_strict.py`: strict bifurcation runner.

## Typical Usage

### Train a Full Pipeline Run

Gray-Scott:

```bash
python exps/gray_scott/run_train.py --config configs/complete.yaml --seed 123
```

Feedback loop:

```bash
python exps/feedback_loop/run_train.py --config configs/complete.yaml --seed 42
```

These commands:

1. Generate observations
2. Preprocess them into PSNN training data
3. Train PSNN
4. Train the counting classifier
5. Train the stability classifier

### Evaluate Locating on Test Observations

Strict mode:

```bash
python exps/gray_scott/run_eval.py --config configs/complete.yaml --seed 123 --mode strict -- --device cpu
python exps/feedback_loop/run_eval.py --config configs/complete.yaml --seed 42 --mode strict -- --device cpu
```

Flexible mode:

```bash
python exps/gray_scott/run_eval.py --config configs/complete.yaml --seed 123 --mode flexible -- --device cpu
python exps/feedback_loop/run_eval.py --config configs/complete.yaml --seed 42 --mode flexible -- --device cpu
```

### Run Bifurcation-Style Sweeps

```bash
python exps/gray_scott/run_bifur.py --config configs/complete.yaml --seed 123 --mode strict -- --device cpu
python exps/feedback_loop/run_bifur.py --config configs/complete.yaml --seed 42 --mode flexible -- --device cpu
```

## Outputs

A typical run directory contains:

- `data/`: generated observations and PSNN regression datasets
- `psnn_phi.pt`: learned PSNN field
- `psnn_numsol.pt`: learned count classifier
- `psnn_stability_cls.pt`: learned stability classifier
- `test_observation_eval_strict/` or `test_observation_eval_flexible/`: evaluation outputs
- `bifur_strict_runs/` or `bifur_flexible_runs/`: bifurcation outputs

## Notes

- The flexible and strict locaters share the same learned PSNN and stability models; the main difference is whether the count classifier is used to fix the number of centers.
- Synthetic incompleteness is introduced during observation generation, before preprocessing into training targets.
- YAML configs control data generation, incompleteness, training hyperparameters, and output locations.
