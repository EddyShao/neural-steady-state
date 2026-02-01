# Feedback-loop experiment

This folder is the cleaned, reproducible version of the collaborator notebook `NN_feedback_loop.ipynb`.

## What the model learns

We train a `psnn.nets.PSNN` surrogate for

- `Phi(theta, u)`: a “solution likelihood” field over state space `u=(p1,p2)` for each parameter `theta=(alpha1,alpha2,gamma1,gamma2)`.

The target is built as a sum of Gaussian RBF bumps centered at the homogeneous steady states (the notebook’s `Z`).

## Data generation

- Generator: [exps/feedback_loop/data/gen_feedback_loop.py](exps/feedback_loop/data/gen_feedback_loop.py)
- Outputs (written into `exps/feedback_loop/data/` if you run it from that folder):
  - `feedback_loop_data_train.npz`, `feedback_loop_data_test.npz` (keys: `Theta`, `U`, `Phi`)
  - `feedback_loop_obs_train.pkl`, `feedback_loop_obs_test.pkl` (used for evaluation / cutoff tuning)

Run:

- `python exps/feedback_loop/data/gen_feedback_loop.py`

## Training

Entry point: [exps/feedback_loop/train.py](exps/feedback_loop/train.py)

- `python exps/feedback_loop/train.py`

This trains:

- `psnn_phi.pt` (from `feedback_loop_data_*.npz`)

## Locating steady states

Script: [exps/feedback_loop/locate_sol.py](exps/feedback_loop/locate_sol.py)

- Evaluates the trained PSNN on a grid over `[0,5]^2`
- Keeps points where `Phi >= L_cut`
- Clusters them via K-means with the known number of solutions

Run:

- `python exps/feedback_loop/locate_sol.py`
