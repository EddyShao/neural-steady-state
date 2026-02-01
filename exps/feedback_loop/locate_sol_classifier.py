import argparse
import os
import sys
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

try:
	import tqdm
except Exception:  # pragma: no cover
	tqdm = None


# --- make repo root importable (so `import psnn` works no matter where you run from) ---
exp_dir = os.path.dirname(os.path.abspath(__file__))          # .../exps/feedback_loop
repo_root = os.path.abspath(os.path.join(exp_dir, "../.."))   # .../ (repo root)
if repo_root not in sys.path:
	sys.path.insert(0, repo_root)

from psnn import datasets, nets
from psnn.config import cfg_get, load_yaml, resolve_path
from exps.feedback_loop.data.gen_feedback_loop import U as true_U


def make_u_grid(m: int = 200, lo: float = 0.0, hi: float = 5.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Construct a uniform m x m grid on [lo,hi]^2 for u=(p1,p2)."""
	p1_axis = np.linspace(lo, hi, m, endpoint=False)
	p2_axis = np.linspace(lo, hi, m, endpoint=False)
	P1, P2 = np.meshgrid(p1_axis, p2_axis, indexing="ij")
	U_all = np.stack([P1.ravel(), P2.ravel()], axis=1)
	return U_all, p1_axis, p2_axis


def infer_eta_from_npz(train_npz: str, device: torch.device) -> float:
	train_loader, _ = datasets.make_loaders(
		train_npz,
		train_npz,
		batch_size=1024,
		num_workers=0,
		device=device,
	)
	phi_max = float(train_loader.dataset.Phi.max().item())
	eta = min(0.01, 1.5 * (phi_max - 1.0))
	print(f"Inferred eta={eta:.2e} from training data (phi_max={phi_max:.2e})")
	return float(eta)


def _infer_mlp_shape(state_dict: dict, prefix: Optional[str] = None) -> Tuple[int, int, int, int]:
	if prefix is None:
		linear_keys = [k for k in state_dict.keys() if k.endswith(".weight") and ".net." in k]
	else:
		linear_keys = [k for k in state_dict.keys() if k.startswith(prefix) and k.endswith(".weight")]
	linear_keys = sorted(linear_keys, key=lambda k: int(k.split(".")[-2]))
	if not linear_keys:
		raise ValueError(f"No linear weights found for prefix={prefix}")

	first_w = state_dict[linear_keys[0]]
	last_w = state_dict[linear_keys[-1]]
	width = int(first_w.shape[0])
	in_dim = int(first_w.shape[1])
	out_dim = int(last_w.shape[0])
	num_linear_layers = len(linear_keys)
	depth = max(0, num_linear_layers - 1)
	return in_dim, width, depth, out_dim


def load_phi_model(phi_ckpt: str, train_npz: str, device: torch.device) -> torch.nn.Module:
	eta = infer_eta_from_npz(train_npz, device)
	state_dict = torch.load(phi_ckpt, map_location=device)
	if isinstance(state_dict, dict) and "state_dict" in state_dict:
		state_dict = state_dict["state_dict"]

	p_in, p_w, p_d, p_out = _infer_mlp_shape(state_dict, "pnn.net")
	s_in, s_w, s_d, s_out = _infer_mlp_shape(state_dict, "snn.net")
	if p_out != s_out:
		raise ValueError(f"PSNN embed_dim mismatch: pnn_out={p_out}, snn_out={s_out}")

	model = nets.PSNN(
		dim_theta=p_in,
		dim_u=s_in,
		embed_dim=p_out,
		width=[p_w, s_w],
		depth=[p_d, s_d],
		eta=eta,
	).to(device)
	model.load_state_dict(state_dict)
	model.eval()
	return model


def load_count_classifier(count_ckpt: str, device: torch.device) -> Tuple[torch.nn.Module, np.ndarray, int]:
	ckpt = torch.load(count_ckpt, map_location=device)
	if isinstance(ckpt, dict) and "state_dict" in ckpt:
		state_dict = ckpt["state_dict"]
		class_values = ckpt.get("class_values", None)
	else:
		state_dict = ckpt
		class_values = None

	in_dim, width, depth, out_dim = _infer_mlp_shape(state_dict, "net.net")
	model = nets.ThetaCountClassifier(dim_theta=in_dim, num_classes=out_dim, width=width, depth=depth).to(device)
	model.load_state_dict(state_dict)
	model.eval()

	if class_values is None:
		class_values_np = np.arange(out_dim, dtype=np.int64)
	else:
		if isinstance(class_values, torch.Tensor):
			class_values_np = class_values.detach().cpu().numpy().astype(np.int64)
		else:
			class_values_np = np.asarray(class_values, dtype=np.int64)
		# Safety: if malformed, fall back.
		if class_values_np.shape[0] != out_dim:
			class_values_np = np.arange(out_dim, dtype=np.int64)

	print(f"Loaded count classifier: in_dim={in_dim}, width={width}, depth={depth}, out_dim={out_dim}")
	print(f"Count class_values (decode): {class_values_np.tolist()}")
	return model, class_values_np, out_dim


def load_stability_classifier(stab_ckpt: str, device: torch.device) -> torch.nn.Module:
	state_dict = torch.load(stab_ckpt, map_location=device)
	if isinstance(state_dict, dict) and "state_dict" in state_dict:
		state_dict = state_dict["state_dict"]

	theta_in, theta_w, theta_d, theta_out = _infer_mlp_shape(state_dict, "theta_net.net")
	u_in, u_w, u_d, u_out = _infer_mlp_shape(state_dict, "u_net.net")
	if theta_out != u_out:
		raise ValueError(f"Stability embed_dim mismatch: theta_out={theta_out}, u_out={u_out}")

	model = nets.StabilityClassifier(
		dim_theta=theta_in,
		dim_u=u_in,
		embed_dim=theta_out,
		width=[theta_w, u_w],
		depth=[theta_d, u_d],
	).to(device)
	model.load_state_dict(state_dict)
	model.eval()
	return model


def predict_num_solutions(
	count_model: torch.nn.Module,
	class_values: np.ndarray,
	theta: np.ndarray,
	device: torch.device,
) -> int:
	theta_t = torch.from_numpy(theta.astype(np.float32)).unsqueeze(0).to(device)
	with torch.no_grad():
		logits = count_model(theta_t).squeeze(0)
		label = int(torch.argmax(torch.softmax(logits, dim=0)).item())
	label = max(0, min(label, int(class_values.shape[0]) - 1))
	return int(class_values[label])


def enforce_cluster_count(U_collected: np.ndarray, expected: int) -> np.ndarray:
	if expected <= 0 or U_collected.size == 0:
		return np.empty((0, U_collected.shape[1] if U_collected.ndim == 2 else 0))
	if expected == 1:
		return U_collected.mean(axis=0, keepdims=True)
	n_clusters = min(int(expected), int(U_collected.shape[0]))
	km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(U_collected)
	return km.cluster_centers_


def locate_solutions_for_theta(
	phi_model: torch.nn.Module,
	theta: np.ndarray,
	U_all: np.ndarray,
	L_cut: float,
	expected_count: int,
	*,
	return_scores: bool = True,
	device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Classifier-assisted locator: force k=expected_count clusters after thresholding.
	"""
	if expected_count <= 0:
		if return_scores:
			return np.empty((0, U_all.shape[1])), np.empty((U_all.shape[0],))
		return np.empty((0, U_all.shape[1])), np.empty((0,))

	U_all_t = torch.from_numpy(U_all.astype(np.float32)).to(device)
	theta_t = torch.from_numpy(theta.astype(np.float32)).to(device).unsqueeze(0)
	theta_t = theta_t.repeat(U_all_t.shape[0], 1)

	with torch.no_grad():
		scores_t = phi_model(U_all_t, theta_t).flatten()
	scores = scores_t.detach().cpu().numpy()

	U_collected = U_all[scores >= float(L_cut), :]
	if U_collected.size == 0:
		if return_scores:
			return np.empty((0, U_all.shape[1])), scores
		return np.empty((0, U_all.shape[1])), np.empty((0,))

	# Enforce the classifier-predicted number of solutions (clusters).
	centers = enforce_cluster_count(U_collected, int(expected_count))

	if return_scores:
		return centers, scores
	return centers, np.empty((0,))


def _bifur_data(par: np.ndarray, sol: Sequence[np.ndarray], stb: Sequence[np.ndarray]):
	"""Mirror the notebook's bifur_data() helper.

	par: (N,) parameter values (alpha2)
	sol: list length N, each element is (k_i, 2) centers
	stb: list length N, each element is (k_i,) stability labels (+1 stable, -1 unstable)
	"""
	bifur_par = []
	bifur_sol1 = []
	bifur_sol2 = []
	bifur_col = []
	for i in range(len(par)):
		for j in range(int(sol[i].shape[0])):
			bifur_par.append(float(par[i]))
			bifur_sol1.append(float(sol[i][j][0]))
			bifur_sol2.append(float(sol[i][j][1]))
			bifur_col.append("royalblue" if float(stb[i][j]) > 0 else "hotpink")
	return np.asarray(bifur_par), np.asarray(bifur_sol1), np.asarray(bifur_sol2), np.asarray(bifur_col)


def _bifur_data_true(par: np.ndarray, sol: Sequence[Sequence[dict]]):
	"""Build bifurcation data from analytic solver outputs."""
	bifur_par = []
	bifur_sol1 = []
	bifur_sol2 = []
	bifur_col = []
	for i in range(len(par)):
		for s in sol[i]:
			u = np.asarray(s["u"], dtype=float)
			bifur_par.append(float(par[i]))
			bifur_sol1.append(float(u[0]))
			bifur_sol2.append(float(u[1]))
			bifur_col.append("royalblue" if bool(s.get("stable", False)) else "hotpink")
	return np.asarray(bifur_par), np.asarray(bifur_sol1), np.asarray(bifur_sol2), np.asarray(bifur_col)


def build_bifurcation_diagram(
	*,
	phi_model: torch.nn.Module,
	count_model: torch.nn.Module,
	class_values: np.ndarray,
	stab_model: torch.nn.Module,
	U_all: np.ndarray,
	alpha1: float,
	gamma1: float,
	gamma2: float,
	alpha2_min: float,
	alpha2_max: float,
	steps: int,
	L_cut: float,
	device: torch.device,
):
	alpha2_grid = np.linspace(float(alpha2_min), float(alpha2_max), int(steps), endpoint=True)
	Sol = []
	Stb = []

	iterable = range(alpha2_grid.shape[0])
	if tqdm is not None:
		iterable = tqdm.tqdm(iterable, desc="Bifurcation sweep")

	for i in iterable:
		alpha2 = float(alpha2_grid[i])
		theta = np.asarray([alpha1, alpha2, gamma1, gamma2], dtype=np.float32)
		expected = int(predict_num_solutions(count_model, class_values, theta, device))
		centers, _ = locate_solutions_for_theta(
			phi_model=phi_model,
			theta=theta,
			U_all=U_all,
			L_cut=L_cut,
			expected_count=expected,
			return_scores=False,
			device=device,
		)
		probs = classify_stability(stab_model, theta, centers, device)
		stb = np.where(probs >= 0.5, 1.0, -1.0).astype(np.float32)
		Sol.append(centers)
		Stb.append(stb)

	return alpha2_grid, Sol, Stb


def build_true_bifurcation(
	*,
	alpha1: float,
	gamma1: float,
	gamma2: float,
	alpha2_min: float,
	alpha2_max: float,
	steps: int,
):
	alpha2_grid = np.linspace(float(alpha2_min), float(alpha2_max), int(steps), endpoint=True)
	Sol_true = []
	for i in range(alpha2_grid.shape[0]):
		alpha2 = float(alpha2_grid[i])
		theta = np.asarray([alpha1, alpha2, gamma1, gamma2], dtype=np.float32)
		Sol_true.append(true_U(theta))
	return alpha2_grid, Sol_true


def classify_stability(
	stab_model: torch.nn.Module,
	theta: np.ndarray,
	centers: np.ndarray,
	device: torch.device,
) -> np.ndarray:
	if centers is None or centers.size == 0:
		return np.empty((0,), dtype=float)
	centers_t = torch.from_numpy(centers.astype(np.float32)).to(device)
	theta_t = torch.from_numpy(theta.astype(np.float32)).to(device).unsqueeze(0)
	theta_t = theta_t.repeat(centers_t.shape[0], 1)
	with torch.no_grad():
		probs = stab_model(centers_t, theta_t).view(-1).detach().cpu().numpy()
	return probs


def parse_theta(theta_args: Sequence[float]) -> np.ndarray:
	if len(theta_args) != 4:
		raise ValueError("Expected 4 floats for theta (dim_theta=4).")
	return np.asarray([float(x) for x in theta_args], dtype=np.float32)


def main():
	pre = argparse.ArgumentParser(add_help=False)
	pre.add_argument("--config", type=str, default=None, help="Path to YAML config.")
	pre_args, remaining = pre.parse_known_args()

	default_cfg = os.path.join(exp_dir, "config.yaml")
	cfg = {}
	cfg_path = pre_args.config or (default_cfg if os.path.exists(default_cfg) else None)
	if cfg_path:
		cfg = load_yaml(cfg_path)

	loc_cfg = cfg_get(cfg, "postprocessing.locate", {})
	path_cfg = cfg_get(cfg, "postprocessing.paths", {})

	parser = argparse.ArgumentParser(
		description="Feedback-loop solution locator (count classifier -> forced clustering -> stability)",
		parents=[pre],
	)
	parser.add_argument("--m", type=int, default=cfg_get(loc_cfg, "m", 200), help="Grid resolution per axis")
	parser.add_argument("--lo", type=float, default=cfg_get(loc_cfg, "lo", 0.0), help="Grid lower bound")
	parser.add_argument("--hi", type=float, default=cfg_get(loc_cfg, "hi", 5.0), help="Grid upper bound")
	parser.add_argument("--L-cut", dest="L_cut", type=float, default=cfg_get(loc_cfg, "L_cut", None), help="Threshold for collecting points")
	parser.add_argument("--theta", nargs=4, type=float, default=cfg_get(loc_cfg, "theta", None), metavar=("t1", "t2", "t3", "t4"), help="Explicit theta")
	parser.add_argument("--bifurcation", action="store_true", default=bool(cfg_get(loc_cfg, "bifurcation", False)), help="Plot bifurcation diagram by sweeping alpha2 (like the notebook)")
	parser.add_argument("--true-bifurcation", action="store_true", default=bool(cfg_get(loc_cfg, "true_bifurcation", False)), help="Overlay analytic bifurcation from U(theta)")
	parser.add_argument("--alpha1", type=float, default=cfg_get(loc_cfg, "alpha1", 4.0), help="Fixed alpha1 for bifurcation sweep")
	parser.add_argument("--gamma1", type=float, default=cfg_get(loc_cfg, "gamma1", 0.2), help="Fixed gamma1 for bifurcation sweep")
	parser.add_argument("--gamma2", type=float, default=cfg_get(loc_cfg, "gamma2", 0.2), help="Fixed gamma2 for bifurcation sweep")
	parser.add_argument("--alpha2-min", type=float, default=cfg_get(loc_cfg, "alpha2_min", 2.0), help="alpha2 sweep start")
	parser.add_argument("--alpha2-max", type=float, default=cfg_get(loc_cfg, "alpha2_max", 3.0), help="alpha2 sweep end")
	parser.add_argument("--alpha2-steps", type=int, default=cfg_get(loc_cfg, "alpha2_steps", 101), help="# of alpha2 points")
	parser.add_argument("--obs", type=str, default=cfg_get(loc_cfg, "obs", None), help="Optional obs .pkl for theta selection")
	parser.add_argument("--obs-index", type=int, default=cfg_get(loc_cfg, "obs_index", 0), help="Index into obs list if --theta not provided")
	parser.add_argument("--no-plot", action="store_true", default=bool(cfg_get(loc_cfg, "no_plot", False)), help="Disable matplotlib plotting")
	parser.add_argument("--save-fig", type=str, default=cfg_get(loc_cfg, "save_fig", None), help="Path to save the figure")

	args = parser.parse_args(remaining)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	data_dir = resolve_path(exp_dir, cfg_get(path_cfg, "data_dir", "data"))
	train_npz = resolve_path(exp_dir, cfg_get(path_cfg, "train_npz", data_dir / "feedback_loop_data_train.npz"))
	phi_ckpt = resolve_path(exp_dir, cfg_get(path_cfg, "phi_ckpt", "psnn_phi.pt"))
	count_ckpt = resolve_path(exp_dir, cfg_get(path_cfg, "count_ckpt", "psnn_numsol.pt"))
	stab_ckpt = resolve_path(exp_dir, cfg_get(path_cfg, "stability_ckpt", "psnn_stability_cls.pt"))

	if not os.path.exists(train_npz):
		raise FileNotFoundError(f"Missing training npz (for eta inference): {train_npz}")
	if not os.path.exists(phi_ckpt):
		raise FileNotFoundError(f"Missing Phi model checkpoint: {phi_ckpt}")
	if not os.path.exists(count_ckpt):
		raise FileNotFoundError(f"Missing count classifier checkpoint: {count_ckpt}")
	if not os.path.exists(stab_ckpt):
		raise FileNotFoundError(f"Missing stability classifier checkpoint: {stab_ckpt}")

	obs_list = None
	if args.obs is not None:
		import joblib

		if not os.path.exists(args.obs):
			raise FileNotFoundError(f"Missing obs file: {args.obs}")
		obs_list = joblib.load(args.obs)

	phi_model = load_phi_model(phi_ckpt, train_npz, device)
	count_model, class_values, _num_classes = load_count_classifier(count_ckpt, device)
	stab_model = load_stability_classifier(stab_ckpt, device)

	if args.L_cut is None:
		raise ValueError("Provide --L-cut (no automatic cutoff scan in this locator).")

	U_all, p1_axis, p2_axis = make_u_grid(m=400, lo=args.lo, hi=args.hi)
	L_cut = float(args.L_cut)

	# --- bifurcation diagram mode (mirrors NN_feedback_loop.ipynb) ---
	if args.true_bifurcation and not args.bifurcation:
		args.bifurcation = True

	if args.bifurcation:
		alpha2_grid, Sol, Stb = build_bifurcation_diagram(
			phi_model=phi_model,
			count_model=count_model,
			class_values=class_values,
			stab_model=stab_model,
			U_all=U_all,
			alpha1=args.alpha1,
			gamma1=args.gamma1,
			gamma2=args.gamma2,
			alpha2_min=args.alpha2_min,
			alpha2_max=args.alpha2_max,
			steps=args.alpha2_steps,
			L_cut=L_cut,
			device=device,
		)

		b_par, b_sol1, b_sol2, b_col = _bifur_data(alpha2_grid, Sol, Stb)
		true_par = true_sol1 = true_sol2 = true_col = None
		if args.true_bifurcation:
			alpha2_true, Sol_true = build_true_bifurcation(
				alpha1=args.alpha1,
				gamma1=args.gamma1,
				gamma2=args.gamma2,
				alpha2_min=args.alpha2_min,
				alpha2_max=args.alpha2_max,
				steps=args.alpha2_steps,
			)
			true_par, true_sol1, true_sol2, true_col = _bifur_data_true(alpha2_true, Sol_true)

		if args.no_plot:
			print("Fixed alpha1=4,gamm1=0.2,gamm2=0.2.")
			print(f"Generated {b_par.shape[0]} solution points from alpha2 sweep.")
			if args.true_bifurcation:
				print(f"Generated {true_par.shape[0]} true solution points from alpha2 sweep.")
			return

		plt.figure(figsize=(6.0, 4.0))
		plt.scatter(b_par, b_sol1, color=b_col, s=12)
		if args.true_bifurcation and true_par is not None:
			plt.scatter(true_par, true_sol1, color=true_col, s=24, marker="x", label="true")
		plt.xlabel("alpha2")
		plt.ylabel("sol1")
		plt.title("Bifurcation diagram (sol1) | stable=royalblue, unstable=hotpink")
		plt.tight_layout()
		if args.true_bifurcation and true_par is not None:
			plt.legend(loc="best", fontsize=8)
		if args.save_fig:
			root, ext = os.path.splitext(args.save_fig)
			out1 = f"{root}_sol1{ext or '.png'}"
			plt.savefig(out1, dpi=200)
			print(f"Saved: {out1}")
		else:
			plt.show()

		print("Fixed alpha1=4,gamm1=0.2,gamm2=0.2.")
		plt.figure(figsize=(6.0, 4.0))
		plt.scatter(b_par, b_sol2, color=b_col, s=12)
		if args.true_bifurcation and true_par is not None:
			plt.scatter(true_par, true_sol2, color=true_col, s=24, marker="x", label="true")
		plt.xlabel("alpha2")
		plt.ylabel("sol2")
		plt.title("Bifurcation diagram (sol2) | stable=royalblue, unstable=hotpink")
		plt.tight_layout()
		if args.true_bifurcation and true_par is not None:
			plt.legend(loc="best", fontsize=8)
		if args.save_fig:
			root, ext = os.path.splitext(args.save_fig)
			out2 = f"{root}_sol2{ext or '.png'}"
			plt.savefig(out2, dpi=200)
			print(f"Saved: {out2}")
		else:
			plt.show()
		return

	# allow config-provided theta if CLI omitted it
	if args.theta is None and cfg_get(loc_cfg, "theta", None) is not None:
		args.theta = cfg_get(loc_cfg, "theta")

	if args.theta is not None:
		theta = parse_theta(args.theta)
	elif obs_list is not None and len(obs_list) > 0:
		theta = np.asarray(obs_list[int(args.obs_index)]["Theta"], dtype=np.float32)
	else:
		raise ValueError("Provide --theta t1 t2 t3 t4 or --obs path.pkl")

	expected = predict_num_solutions(count_model, class_values, theta, device)
	centers, scores = locate_solutions_for_theta(
		phi_model=phi_model,
		theta=theta,
		U_all=U_all,
		L_cut=L_cut,
		expected_count=expected,
		device=device,
	)

	stab_probs = classify_stability(stab_model, theta, centers, device)

	print(f"Theta: {theta.tolist()}")
	print(f"Predicted #solutions (count classifier): {expected}")
	print(f"Located centers: {centers.shape[0]}")
	if centers.shape[0] > 0:
		for i, c in enumerate(centers):
			p = float(stab_probs[i]) if i < stab_probs.shape[0] else float('nan')
			print(f"  center[{i}]=[{c[0]:.6f}, {c[1]:.6f}]  stable_prob={p:.4f}")

	if args.no_plot:
		return

	Phi_grid = scores.reshape(args.m, args.m)
	P1, P2 = np.meshgrid(p1_axis, p2_axis, indexing="ij")

	plt.figure(figsize=(6.5, 5.2))
	plt.contourf(P1, P2, Phi_grid, levels=50)
	cbar = plt.colorbar()
	cbar.set_label(r"$\\Phi$")

	if centers.shape[0] > 0:
		is_stable = (stab_probs >= 0.5) if stab_probs.size == centers.shape[0] else np.zeros((centers.shape[0],), dtype=bool)
		stable_pts = centers[is_stable]
		unstable_pts = centers[~is_stable]
		if stable_pts.size > 0:
			plt.scatter(stable_pts[:, 0], stable_pts[:, 1], c="limegreen", s=70, marker="o", label="stable")
		if unstable_pts.size > 0:
			plt.scatter(unstable_pts[:, 0], unstable_pts[:, 1], c="crimson", s=70, marker="x", label="unstable")
		plt.legend(loc="best")

	theta_str = ",".join([f"{t:.3f}" for t in theta.tolist()])
	plt.title(f"Feedback-loop locator | theta=[{theta_str}] | expected={expected} | L={L_cut:.3f}")
	plt.xlabel("p1")
	plt.ylabel("p2")	
	plt.tight_layout()

	if args.save_fig:
		plt.savefig(args.save_fig, dpi=200)
		print(f"Saved figure to: {args.save_fig}")
	else:
		plt.show()


if __name__ == "__main__":
	main()
