# Data-Driven discovery of steady states in ODE systesms

This code base provides a workflow for learning and locating steady-state solutions of parametric ODE systems.
It centers on three learned components: a solution-parameter matching network (PSNN), a stability classifier,
and a solution-count classifier. Example pipelines are included for the Gray-Scott and feedback-loop systems.
This repository corresponds to the paper: https://arxiv.org/abs/2312.10315

## Networks
The following 3 networks are trained independently. 

### 1) Solution-count (classifier)
- Given theta, predicts the number of steady states.
- Implemented in `psnn/nets.py` as `ThetaCountClassifier`.
- Trained with (class-weighted) cross-entropy to address label imbalance.

### 2) PSNN (solution-parameter matching)
- Trains a neural (probabilistics) field `Phi(theta, u)` that approximates target function $\Phi$.
- Implemented in `psnn/nets.py` as `PSNN`.
- Trained via regression on generated data: `Theta`, `U`, and `Phi` stored in `*_data_*.npz`.

### 3) Stability classifier
- Given (theta, u), predicts whether a steady state is stable.
- Note: The input is not necessary a parameter-solution pair; Therefore, the network's output will be $\mathbb{P}(\text{stable} \ | \ (\theta,u)\text{ is a solution})$. Very much similar to the multi-proecess training part.
- Implemented as a PSNN-style inner product with a logistic output in `psnn/nets.py` as `StabilityClassifier`.
- Trained with (class-weighted) binary cross-entropy to address label imbalance.



## Pipeline for locating solutions (single theta)
1) Use the solution-count classifier to estimate the number of solutions.
2) Use PSNN Phi(theta, u) to evaluate a grid in state space and build a landscape of Gaussian bumps.
3) Apply a cutoff L to keep points with Phi >= L.
4) Cluster those points using the predicted count as the cluster number.
5) Run the stability classifier on each cluster center to obtain stability.

## Examples

### Gray-Scott
- Data generation: `python exps/grey_scott/data/gen_gray_scott.py`
- Training (PSNN + stability + count): `python exps/grey_scott/train.py`
- Solution map with stability: `python exps/grey_scott/locate_sol_classifier.py --make_map`

### Feedback loop
- Data generation: `python exps/feedback_loop/data/gen_feedback_loop.py`
- Training (PSNN + stability + count): `python exps/feedback_loop/train.py`

## Configuration (YAML)
All experiment hyperparameters are collected in YAML config files:
- `exps/grey_scott/config.yaml`
- `exps/feedback_loop/config.yaml`

Each config separates:
- `data_generation`: data sampling and output files.
- `training`: model hyperparameters and checkpoints.
- `postprocessing`: locator/bifurcation settings.

All scripts accept `--config path/to/config.yaml`. If omitted, they will use the
default config in the experiment directory when present.
## Outputs
- `psnn_phi.pt`: PSNN weights (Phi landscape).
- `psnn_numsol.pt`: solution-count classifier.
- `psnn_stability_cls.pt`: stability classifier.

## Notes
- If you change model widths/depths during training, load scripts infer shapes from checkpoints when needed.
- Class imbalance is handled with weighted losses for both classifiers.
