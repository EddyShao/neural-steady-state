import argparse
import os
import sys
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import OrderedDict

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


def _normalize_state_dict_keys(state_dict: dict) -> dict:
	"""Normalize state_dict keys across common wrappers.

	- torch.compile() prefixes parameters with "_orig_mod."
	- DistributedDataParallel prefixes with "module."
	- Some training code may nest under "model."
	"""
	if not isinstance(state_dict, dict):
		return state_dict
	prefixes = ("_orig_mod.", "module.", "model.")

	changed = False
	out = OrderedDict()
	for k, v in state_dict.items():
		k2 = k
		for p in prefixes:
			if k2.startswith(p):
				k2 = k2[len(p) :]
				changed = True
		out[k2] = v
	return out if changed else state_dict


def load_phi_model(phi_ckpt: str, train_npz: str, device: torch.device) -> torch.nn.Module:
	eta = infer_eta_from_npz(train_npz, device)
	try:
		state_dict = torch.load(phi_ckpt, map_location=device, weights_only=True)
	except TypeError:  # older torch
		state_dict = torch.load(phi_ckpt, map_location=device)
	if isinstance(state_dict, dict) and "state_dict" in state_dict:
		state_dict = state_dict["state_dict"]
	state_dict = _normalize_state_dict_keys(state_dict)

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
	try:
		ckpt = torch.load(count_ckpt, map_location=device, weights_only=True)
	except TypeError:  # older torch
		ckpt = torch.load(count_ckpt, map_location=device)
	if isinstance(ckpt, dict) and "state_dict" in ckpt:
		state_dict = _normalize_state_dict_keys(ckpt["state_dict"])
		class_values = ckpt.get("class_values", None)
	else:
		state_dict = _normalize_state_dict_keys(ckpt)
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
	try:
		state_dict = torch.load(stab_ckpt, map_location=device, weights_only=True)
	except TypeError:  # older torch
		state_dict = torch.load(stab_ckpt, map_location=device)
	if isinstance(state_dict, dict) and "state_dict" in state_dict:
		state_dict = state_dict["state_dict"]
	state_dict = _normalize_state_dict_keys(state_dict)

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

	scores = score_phi_points(phi_model, theta, U_all, device)

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


def score_phi_points(
	phi_model: torch.nn.Module,
	theta: np.ndarray,
	U_points: np.ndarray,
	device: torch.device,
) -> np.ndarray:
	U_all_t = torch.from_numpy(U_points.astype(np.float32)).to(device)
	theta_t = torch.from_numpy(theta.astype(np.float32)).to(device).unsqueeze(0)
	theta_t = theta_t.repeat(U_all_t.shape[0], 1)
	with torch.no_grad():
		scores_t = phi_model(U_all_t, theta_t).flatten()
	return scores_t.detach().cpu().numpy()


def _amr_sort_centers(centers: np.ndarray) -> np.ndarray:
	return centers[np.lexsort((centers[:, 1], centers[:, 0]))]


def _amr_match_centers(prev_centers: np.ndarray, new_centers: np.ndarray) -> np.ndarray:
	"""Greedy center matching to avoid depending on scipy or external AMR modules."""
	if prev_centers.shape != new_centers.shape:
		return new_centers
	matched = np.zeros_like(new_centers)
	used = set()
	for i in range(prev_centers.shape[0]):
		dists = np.linalg.norm(new_centers - prev_centers[i], axis=1)
		for j in np.argsort(dists).tolist():
			if j not in used:
				matched[i] = new_centers[j]
				used.add(j)
				break
	return matched


def _amr_uniform_sampling(D, N):
	low = np.array([D[i][0] for i in range(len(D))], dtype=float)
	high = np.array([D[i][1] for i in range(len(D))], dtype=float)
	return np.random.uniform(low=low, high=high, size=(int(N), len(D)))


def _amr_uniform_ball_sampling(center: np.ndarray, radius: float, N: int) -> np.ndarray:
	dim = len(center)
	directions = np.random.randn(int(N), dim)
	norm = np.linalg.norm(directions, axis=1, keepdims=True)
	norm = np.maximum(norm, 1e-12)
	directions = directions / norm
	radii = np.random.rand(int(N), 1) ** (1.0 / dim)
	return center + directions * (radii * float(radius))


def _amr_clamp_to_domain(points: np.ndarray, D) -> np.ndarray:
	P = points.copy()
	for j in range(P.shape[1]):
		P[:, j] = np.clip(P[:, j], D[j][0], D[j][1])
	return P


def _amr_make_grid(D, m: int) -> np.ndarray:
	x = np.linspace(D[0][0], D[0][1], int(m))
	y = np.linspace(D[1][0], D[1][1], int(m))
	xx, yy = np.meshgrid(x, y)
	return np.vstack([xx.ravel(), yy.ravel()]).T


def _amr_build_boxes(centers: np.ndarray, base_radius: float):
	return [
		[c[0] - base_radius, c[0] + base_radius, c[1] - base_radius, c[1] + base_radius]
		for c in centers
	]


def _amr_merge_boxes(boxes):
	merged = []
	for box in boxes:
		merged_flag = False
		for i, cur in enumerate(merged):
			if not (box[1] < cur[0] or box[0] > cur[1] or box[3] < cur[2] or box[2] > cur[3]):
				merged[i] = [
					min(box[0], cur[0]),
					max(box[1], cur[1]),
					min(box[2], cur[2]),
					max(box[3], cur[3]),
				]
				merged_flag = True
				break
		if not merged_flag:
			merged.append(list(box))
	return merged


def _amr_refine_boxes(boxes, m_local: int, D):
	pts_all = []
	for box in boxes:
		D_local = [
			[max(box[0], D[0][0]), min(box[1], D[0][1])],
			[max(box[2], D[1][0]), min(box[3], D[1][1])],
		]
		pts_all.append(_amr_make_grid(D_local, m_local))
	return np.vstack(pts_all) if pts_all else np.empty((0, 2))


def _amr_compute_ball_radii(centers: np.ndarray, r_default: float, radius_scale: float) -> np.ndarray:
	if len(centers) <= 1:
		return np.array([r_default], dtype=float)
	radii = np.zeros(len(centers), dtype=float)
	for i in range(len(centers)):
		dists = np.linalg.norm(centers - centers[i], axis=1)
		dists[i] = np.inf
		radii[i] = radius_scale * np.min(dists)
		if not np.isfinite(radii[i]) or radii[i] <= 0:
			radii[i] = r_default
	return radii


def adaptive_refinement(
	f,
	D,
	expected_k,
	method="grid",
	L_cut=0.3,
	max_iter=12,
	tol=1e-4,
	plot_each_iter=False,
	random_state=0,
	m0=12,
	m_growth=1.5,
	m_local=60,
	base_radius=0.4,
	N_global=2000,
	N_local=1000,
	r_default=0.3,
	radius_scale=0.45,
	global_refill_factor=1.5,
):
	"""ALGORITHM-DEPENDENT: local copy of the adaptive refinement peak finder.

	This block is intentionally kept in this file so the locator does not depend
	on `amr_wc.py`. If the refinement strategy changes, update this function and
	its `_amr_*` helpers together.
	"""
	np.random.seed(int(random_state))
	method = str(method).lower().strip()
	if method not in ("grid", "random"):
		raise ValueError("method must be 'grid' or 'random'.")

	centers_prev = None
	if method == "grid":
		m_global = int(m0)
		pts = _amr_make_grid(D, m_global)
		scores = f(pts)
	else:
		pts = _amr_uniform_sampling(D, N_global)
		scores = f(pts)

	for _ in range(int(max_iter)):
		collected = pts[scores >= float(L_cut)]

		if len(collected) < int(expected_k):
			if method == "grid":
				m_global = max(m_global + 1, int(np.ceil(m_global * float(m_growth))))
				pts = _amr_make_grid(D, m_global)
				scores = f(pts)
				centers_prev = None
			else:
				addN = max(200, int(np.ceil(len(pts) * (float(global_refill_factor) - 1.0))))
				new_pts = _amr_uniform_sampling(D, addN)
				new_scores = f(new_pts)
				pts = np.vstack([pts, new_pts])
				scores = np.concatenate([scores, new_scores])
			continue

		km = KMeans(n_clusters=int(expected_k), n_init=20, random_state=int(random_state)).fit(collected)
		centers = _amr_sort_centers(km.cluster_centers_)

		if centers_prev is not None:
			centers = _amr_match_centers(centers_prev, centers)
			movement = np.max(np.linalg.norm(centers - centers_prev, axis=1))
			if movement < float(tol):
				return centers

		if plot_each_iter:
			pass

		if method == "grid":
			boxes = _amr_build_boxes(centers, float(base_radius))
			refined_pts = _amr_refine_boxes(_amr_merge_boxes(boxes), int(m_local), D)
		else:
			radii = _amr_compute_ball_radii(centers, float(r_default), float(radius_scale))
			refined_pts_list = []
			for c, r in zip(centers, radii):
				p = _amr_uniform_ball_sampling(c, r, int(N_local))
				refined_pts_list.append(_amr_clamp_to_domain(p, D))
			refined_pts = np.vstack(refined_pts_list) if refined_pts_list else np.empty((0, 2))

		if len(refined_pts) > 0:
			refined_scores = f(refined_pts)
			pts = np.vstack([pts, refined_pts])
			scores = np.concatenate([scores, refined_scores])

		centers_prev = centers

	return centers_prev if centers_prev is not None else np.empty((0, 2))


def locate_solutions_for_theta_adaptive(
	phi_model: torch.nn.Module,
	theta: np.ndarray,
	L_cut: float,
	expected_count: int,
	*,
	lo: float,
	hi: float,
	device: torch.device,
	amr_method: str = "grid",
	amr_random_state: int = 0,
	amr_max_iter: int = 12,
	amr_tol: float = 1e-4,
	amr_m0: int = 12,
	amr_m_growth: float = 1.5,
	amr_m_local: int = 60,
	amr_base_radius: float = 0.4,
	amr_N_global: int = 2000,
	amr_N_local: int = 1000,
	amr_r_default: float = 0.3,
	amr_radius_scale: float = 0.45,
) -> np.ndarray:
	"""ALGORITHM-DEPENDENT: adaptive solution localization for the feedback-loop locator."""
	if expected_count <= 0:
		return np.empty((0, 2), dtype=float)

	D = [[float(lo), float(hi)], [float(lo), float(hi)]]

	def _phi_fn(U_points: np.ndarray) -> np.ndarray:
		return score_phi_points(phi_model, theta, U_points, device)

	centers = adaptive_refinement(
		_phi_fn,
		D,
		expected_k=int(expected_count),
		method=str(amr_method),
		L_cut=float(L_cut),
		max_iter=int(amr_max_iter),
		tol=float(amr_tol),
		plot_each_iter=False,
		random_state=int(amr_random_state),
		m0=int(amr_m0),
		m_growth=float(amr_m_growth),
		m_local=int(amr_m_local),
		base_radius=float(amr_base_radius),
		N_global=int(amr_N_global),
		N_local=int(amr_N_local),
		r_default=float(amr_r_default),
		radius_scale=float(amr_radius_scale),
	)
	return np.asarray(centers, dtype=float)


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


def _build_branches_from_centers(
	alpha2_grid: np.ndarray,
	Sol: Sequence[np.ndarray],
	Stb: Sequence[np.ndarray],
	*,
	max_match_dist: float = 0.6,
):
	"""Assign solution centers across alpha2 into continuous branches.

	This is a lightweight nearest-neighbor tracker meant for k in {1,3}.
	It matches centers between consecutive alpha2 values by minimizing Euclidean
	distance in u-space.

	Returns a list of branches, each with arrays: alpha2, u, stb.
	"""
	branches = []
	active = []  # list of dicts: {"last_u", "alpha2", "u_list", "a_list", "stb_list"}

	for i, a2 in enumerate(alpha2_grid):
		centers = np.asarray(Sol[i], dtype=float)
		stb = np.asarray(Stb[i], dtype=float) if Stb is not None else np.empty((0,), dtype=float)
		if centers.size == 0:
			# close all active branches
			branches.extend(active)
			active = []
			continue

		# normalize stability labels to +1/-1
		if stb.size == 0:
			stb = np.full((centers.shape[0],), 1.0, dtype=float)
		else:
			stb = np.where(stb >= 0.0, 1.0, -1.0).astype(float)

		# First frame: create branches
		if len(active) == 0:
			for j in range(centers.shape[0]):
				active.append(
					{
						"last_u": centers[j],
						"a_list": [float(a2)],
						"u_list": [centers[j].copy()],
						"stb_list": [float(stb[j])],
					}
				)
			continue

		# Greedy matching: for small k, this is fine.
		used_centers = set()
		used_branches = set()
		pairs = []
		for bi, br in enumerate(active):
			for cj in range(centers.shape[0]):
				d = float(np.linalg.norm(br["last_u"] - centers[cj]))
				pairs.append((d, bi, cj))
		pairs.sort(key=lambda t: t[0])

		new_active = []
		# Start by carrying over all branches; we will update matched ones.
		for br in active:
			new_active.append(br)

		for d, bi, cj in pairs:
			if bi in used_branches or cj in used_centers:
				continue
			if d > float(max_match_dist):
				continue
			used_branches.add(bi)
			used_centers.add(cj)
			br = new_active[bi]
			br["last_u"] = centers[cj]
			br["a_list"].append(float(a2))
			br["u_list"].append(centers[cj].copy())
			br["stb_list"].append(float(stb[cj]))

		# Any unmatched existing branches are considered ended at previous step.
		ended = []
		still_active = []
		for bi, br in enumerate(new_active):
			if bi in used_branches:
				still_active.append(br)
			else:
				ended.append(br)
		branches.extend(ended)

		# Any unmatched centers spawn new branches
		for cj in range(centers.shape[0]):
			if cj in used_centers:
				continue
			still_active.append(
				{
					"last_u": centers[cj],
					"a_list": [float(a2)],
					"u_list": [centers[cj].copy()],
					"stb_list": [float(stb[cj])],
				}
			)

		active = still_active

	# Close remaining
	branches.extend(active)

	# Convert lists to arrays
	out = []
	for br in branches:
		a = np.asarray(br["a_list"], dtype=float)
		u = np.asarray(br["u_list"], dtype=float)
		s = np.asarray(br["stb_list"], dtype=float)
		if a.size == 0 or u.size == 0:
			continue
		out.append({"alpha2": a, "u": u, "stb": s})
	return out


def _build_true_centers_and_stb(alpha2_grid: np.ndarray, Sol_true: Sequence[Sequence[dict]]):
	Sol = []
	Stb = []
	for i in range(len(alpha2_grid)):
		sols_i = Sol_true[i]
		if sols_i is None or len(sols_i) == 0:
			Sol.append(np.zeros((0, 2), dtype=np.float32))
			Stb.append(np.zeros((0,), dtype=np.float32))
			continue
		centers = np.asarray([np.asarray(s["u"], dtype=float) for s in sols_i], dtype=float)
		stb = np.asarray([1.0 if bool(s.get("stable", False)) else -1.0 for s in sols_i], dtype=float)
		Sol.append(centers)
		Stb.append(stb)
	return Sol, Stb


def plot_bifurcation_2d_u_space(
	*,
	alpha2_grid: np.ndarray,
	Sol: Sequence[np.ndarray],
	Stb: Sequence[np.ndarray],
	alpha2_true: Optional[np.ndarray] = None,
	Sol_true: Optional[Sequence[Sequence[dict]]] = None,
	max_match_dist: float = 0.6,
):
	"""2D u-space bifurcation visualization.

	Left: predicted centers (with stability colors). Right: analytic (true) centers.
	Each branch is tracked across alpha2 and drawn as a curve.
	"""
	markers = ["o", "s", "^", "D", "P", "X", "v", ">", "<"]
	stable_c = "royalblue"
	unstable_c = "hotpink"

	branches_pred = _build_branches_from_centers(alpha2_grid, Sol, Stb, max_match_dist=max_match_dist)
	branches_true = []
	if alpha2_true is not None and Sol_true is not None:
		Sol_t, Stb_t = _build_true_centers_and_stb(alpha2_true, Sol_true)
		branches_true = _build_branches_from_centers(alpha2_true, Sol_t, Stb_t, max_match_dist=max_match_dist)

	fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharex=True, sharey=True)
	(ax_pred, ax_true) = axes

	def _draw(ax, branches, *, title: str):
		for bi, br in enumerate(branches):
			a = br["alpha2"]
			u = br["u"]
			s = br["stb"]
			mk = markers[bi % len(markers)]

			# light connecting line to suggest a curve
			ax.plot(u[:, 0], u[:, 1], color="0.75", linewidth=1.0, zorder=1)

			colors = [stable_c if float(si) > 0 else unstable_c for si in s]
			ax.scatter(u[:, 0], u[:, 1], c=colors, s=35, marker=mk, edgecolor="none", zorder=2)

			# mark start/end with larger marker + alpha2 annotations
			start_u, end_u = u[0], u[-1]
			start_a, end_a = float(a[0]), float(a[-1])
			ax.scatter([start_u[0]], [start_u[1]], s=140, marker=mk, facecolor="none", edgecolor="black", linewidth=1.5, zorder=3)
			ax.scatter([end_u[0]], [end_u[1]], s=140, marker=mk, facecolor="none", edgecolor="black", linewidth=1.5, zorder=3)
			ax.annotate(fr"$\alpha_{2}={start_a:.2f}$", xy=(start_u[0], start_u[1]), xytext=(6, 6), textcoords="offset points", fontsize=8)
			ax.annotate(fr"$\alpha_{2}={end_a:.2f}$", xy=(end_u[0], end_u[1]), xytext=(6, -10), textcoords="offset points", fontsize=8)

		ax.set_title(title)
		ax.set_xlabel("p1")
		ax.set_ylabel("p2")
		# legend handles
		ax.scatter([], [], c=stable_c, s=35, marker="o", label="stable")
		ax.scatter([], [], c=unstable_c, s=35, marker="o", label="unstable")
		ax.legend(loc="best", fontsize=8)

	_draw(ax_pred, branches_pred, title="Predicted bifurcation in u-space")
	if branches_true:
		_draw(ax_true, branches_true, title="True bifurcation in u-space")
	else:
		ax_true.axis("off")
		ax_true.text(0.5, 0.5, "Enable --true-bifurcation\nfor analytic curve", ha="center", va="center", transform=ax_true.transAxes)

	fig.tight_layout()
	return fig


def build_bifurcation_diagram(
	*,
	phi_model: torch.nn.Module,
	count_model: torch.nn.Module,
	class_values: np.ndarray,
	stab_model: torch.nn.Module,
	U_all: Optional[np.ndarray],
	alpha1: float,
	gamma1: float,
	gamma2: float,
	alpha2_min: float,
	alpha2_max: float,
	steps: int,
	L_cut: float,
	device: torch.device,
	locate_mode: str = "fixed",
	lo: float = 0.0,
	hi: float = 5.0,
	amr_method: str = "grid",
	amr_random_state: int = 0,
	amr_max_iter: int = 12,
	amr_tol: float = 1e-4,
	amr_m0: int = 12,
	amr_m_growth: float = 1.5,
	amr_m_local: int = 60,
	amr_base_radius: float = 0.4,
	amr_N_global: int = 2000,
	amr_N_local: int = 1000,
	amr_r_default: float = 0.3,
	amr_radius_scale: float = 0.45,
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
		if str(locate_mode).lower() == "adaptive":
			centers = locate_solutions_for_theta_adaptive(
				phi_model=phi_model,
				theta=theta,
				L_cut=L_cut,
				expected_count=expected,
				lo=lo,
				hi=hi,
				device=device,
				amr_method=amr_method,
				amr_random_state=amr_random_state,
				amr_max_iter=amr_max_iter,
				amr_tol=amr_tol,
				amr_m0=amr_m0,
				amr_m_growth=amr_m_growth,
				amr_m_local=amr_m_local,
				amr_base_radius=amr_base_radius,
				amr_N_global=amr_N_global,
				amr_N_local=amr_N_local,
				amr_r_default=amr_r_default,
				amr_radius_scale=amr_radius_scale,
			)
		else:
			if U_all is None:
				raise ValueError("U_all must be provided for locate_mode='fixed'.")
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
	parser.add_argument("--locate-mode", choices=["fixed", "adaptive"], default=cfg_get(loc_cfg, "locate_mode", "fixed"), help="Locator mode: fixed grid or adaptive refinement")
	parser.add_argument("--L-cut", dest="L_cut", type=float, default=cfg_get(loc_cfg, "L_cut", None), help="Threshold for collecting points")
	parser.add_argument("--amr-method", choices=["grid", "random"], default=cfg_get(loc_cfg, "amr.method", "grid"), help="Adaptive refinement sampling method")
	parser.add_argument("--amr-random-state", type=int, default=cfg_get(loc_cfg, "amr.random_state", 0), help="Random state for adaptive clustering")
	parser.add_argument("--amr-max-iter", type=int, default=cfg_get(loc_cfg, "amr.max_iter", 12), help="Max adaptive refinement iterations")
	parser.add_argument("--amr-tol", type=float, default=cfg_get(loc_cfg, "amr.tol", 1e-4), help="Convergence tolerance for adaptive refinement")
	parser.add_argument("--amr-m0", type=int, default=cfg_get(loc_cfg, "amr.m0", 12), help="Initial global grid size for adaptive grid mode")
	parser.add_argument("--amr-m-growth", type=float, default=cfg_get(loc_cfg, "amr.m_growth", 1.5), help="Global grid growth factor when adaptive grid has too few points")
	parser.add_argument("--amr-m-local", type=int, default=cfg_get(loc_cfg, "amr.m_local", 60), help="Local refinement grid size in adaptive grid mode")
	parser.add_argument("--amr-base-radius", type=float, default=cfg_get(loc_cfg, "amr.base_radius", 0.4), help="Local refinement box half-width for adaptive grid mode")
	parser.add_argument("--amr-N-global", type=int, default=cfg_get(loc_cfg, "amr.N_global", 2000), help="Global sample count for adaptive random mode")
	parser.add_argument("--amr-N-local", type=int, default=cfg_get(loc_cfg, "amr.N_local", 1000), help="Per-center local sample count for adaptive random mode")
	parser.add_argument("--amr-r-default", type=float, default=cfg_get(loc_cfg, "amr.r_default", 0.3), help="Default local radius for adaptive random mode")
	parser.add_argument("--amr-radius-scale", type=float, default=cfg_get(loc_cfg, "amr.radius_scale", 0.45), help="Neighbor-distance scaling for adaptive random radii")
	parser.add_argument("--theta", nargs=4, type=float, default=cfg_get(loc_cfg, "theta", None), metavar=("t1", "t2", "t3", "t4"), help="Explicit theta")
	parser.add_argument("--bifurcation", action="store_true", default=bool(cfg_get(loc_cfg, "bifurcation", False)), help="Plot bifurcation diagram by sweeping alpha2 (like the notebook)")
	parser.add_argument("--true-bifurcation", action="store_true", default=bool(cfg_get(loc_cfg, "true_bifurcation", False)), help="Overlay analytic bifurcation from U(theta)")
	parser.add_argument("--alpha1", type=float, default=cfg_get(loc_cfg, "alpha1", 4.0), help="Fixed alpha1 for bifurcation sweep")
	parser.add_argument("--gamma1", type=float, default=cfg_get(loc_cfg, "gamma1", 0.2), help="Fixed gamma1 for bifurcation sweep")
	parser.add_argument("--gamma2", type=float, default=cfg_get(loc_cfg, "gamma2", 0.2), help="Fixed gamma2 for bifurcation sweep")
	parser.add_argument("--alpha2-min", type=float, default=cfg_get(loc_cfg, "alpha2_min", 1.2), help="alpha2 sweep start")
	parser.add_argument("--alpha2-max", type=float, default=cfg_get(loc_cfg, "alpha2_max", 4.8), help="alpha2 sweep end")
	parser.add_argument("--alpha2-steps", type=int, default=cfg_get(loc_cfg, "alpha2_steps", 401), help="# of alpha2 points")
	parser.add_argument("--obs", type=str, default=cfg_get(loc_cfg, "obs", None), help="Optional obs .pkl for theta selection")
	parser.add_argument("--obs-index", type=int, default=cfg_get(loc_cfg, "obs_index", 0), help="Index into obs list if --theta not provided")
	parser.add_argument("--no-plot", action="store_true", default=bool(cfg_get(loc_cfg, "no_plot", False)), help="Disable matplotlib plotting")
	parser.add_argument("--save-fig", type=str, default=cfg_get(loc_cfg, "save_fig", None), help="Path to save the figure")
	parser.add_argument("--bifurcation-2d", action="store_true", default=bool(cfg_get(loc_cfg, "bifurcation_2d", False)), help="Also plot bifurcation curves in 2D u-space (predicted vs true)")

	args = parser.parse_args(remaining)

	# If the user asks for the true bifurcation overlay, default to saving the
	# resulting plots (common in headless runs) unless they explicitly opted out.
	out_dir = resolve_path(exp_dir, cfg_get(path_cfg, "out_dir", "."))
	if args.true_bifurcation and (not args.no_plot) and (not args.save_fig):
		args.save_fig = str(out_dir / "bifurcation_true.png")
		print(f"--true-bifurcation set; defaulting to --save-fig={args.save_fig}")

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
	L_cut = float(args.L_cut)
	locate_mode = str(args.locate_mode).lower()
	U_all = None
	p1_axis = None
	p2_axis = None
	if locate_mode == "fixed":
		U_all, p1_axis, p2_axis = make_u_grid(m=int(args.m), lo=args.lo, hi=args.hi)

	# --- bifurcation diagram mode (mirrors NN_feedback_loop.ipynb) ---
	if args.true_bifurcation and not args.bifurcation:
		args.bifurcation = True
	if args.true_bifurcation and not args.bifurcation_2d:
		args.bifurcation_2d = True

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
			locate_mode=locate_mode,
			lo=args.lo,
			hi=args.hi,
			amr_method=args.amr_method,
			amr_random_state=args.amr_random_state,
			amr_max_iter=args.amr_max_iter,
			amr_tol=args.amr_tol,
			amr_m0=args.amr_m0,
			amr_m_growth=args.amr_m_growth,
			amr_m_local=args.amr_m_local,
			amr_base_radius=args.amr_base_radius,
			amr_N_global=args.amr_N_global,
			amr_N_local=args.amr_N_local,
			amr_r_default=args.amr_r_default,
			amr_radius_scale=args.amr_radius_scale,
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
			out1_dir = os.path.dirname(out1)
			if out1_dir:
				os.makedirs(out1_dir, exist_ok=True)
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
			out2_dir = os.path.dirname(out2)
			if out2_dir:
				os.makedirs(out2_dir, exist_ok=True)
			plt.savefig(out2, dpi=200)
			print(f"Saved: {out2}")
		else:
			plt.show()

		# Additional 2D u-space bifurcation visualization (predicted vs true side-by-side)
		if args.bifurcation_2d:
			fig_u = plot_bifurcation_2d_u_space(
				alpha2_grid=alpha2_grid,
				Sol=Sol,
				Stb=Stb,
				alpha2_true=alpha2_true if args.true_bifurcation else None,
				Sol_true=Sol_true if args.true_bifurcation else None,
			)
			if args.save_fig:
				root, ext = os.path.splitext(args.save_fig)
				out3 = f"{root}_u2d{ext or '.png'}"
				out3_dir = os.path.dirname(out3)
				if out3_dir:
					os.makedirs(out3_dir, exist_ok=True)
				fig_u.savefig(out3, dpi=200)
				print(f"Saved: {out3}")
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
	if locate_mode == "adaptive":
		centers = locate_solutions_for_theta_adaptive(
			phi_model=phi_model,
			theta=theta,
			L_cut=L_cut,
			expected_count=expected,
			lo=args.lo,
			hi=args.hi,
			device=device,
			amr_method=args.amr_method,
			amr_random_state=args.amr_random_state,
			amr_max_iter=args.amr_max_iter,
			amr_tol=args.amr_tol,
			amr_m0=args.amr_m0,
			amr_m_growth=args.amr_m_growth,
			amr_m_local=args.amr_m_local,
			amr_base_radius=args.amr_base_radius,
			amr_N_global=args.amr_N_global,
			amr_N_local=args.amr_N_local,
			amr_r_default=args.amr_r_default,
			amr_radius_scale=args.amr_radius_scale,
		)
		U_plot, p1_axis, p2_axis = make_u_grid(m=int(args.m), lo=args.lo, hi=args.hi)
		scores = score_phi_points(phi_model, theta, U_plot, device)
		U_all = U_plot
	else:
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
	cbar.set_label(r"$\Phi$")

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
		out_dirname = os.path.dirname(args.save_fig)
		if out_dirname:
			os.makedirs(out_dirname, exist_ok=True)
		plt.savefig(args.save_fig, dpi=200)
		print(f"Saved figure to: {args.save_fig}")
	else:
		plt.show()


if __name__ == "__main__":
	main()
