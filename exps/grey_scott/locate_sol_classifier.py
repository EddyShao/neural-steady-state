import argparse
import os
import sys

import numpy as np
import torch
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

try:
    import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

# --- make repo root importable (so `import psnn` works no matter where you run from) ---
exp_dir = os.path.dirname(os.path.abspath(__file__))          # .../exps/grey_scott
repo_root = os.path.abspath(os.path.join(exp_dir, "../.."))   # .../ (repo root)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from psnn import datasets, nets
from psnn.config import cfg_get, load_yaml, resolve_path


def make_u_grid(m: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a uniform m x m grid on [0,1]^2."""
    v_axis = np.arange(0.0, 1.0, 1.0 / m)
    u_axis = np.arange(0.0, 1.0, 1.0 / m)
    V, U = np.meshgrid(v_axis, u_axis, indexing="ij")
    U_all = np.stack([V.ravel(), U.ravel()], axis=1)
    return U_all, v_axis, u_axis


def infer_eta_from_npz(train_npz: str, device: torch.device) -> float:
    train_loader, _ = datasets.make_loaders(
        train_npz,
        train_npz,
        batch_size=1024,
        num_workers=0,
        device=device,
    )
    eta = 1.5 * (train_loader.dataset.Phi.max() - 1.0)
    return min(float(eta), 0.01)


def load_phi_model(phi_ckpt: str, train_npz: str, device: torch.device) -> torch.nn.Module:
    eta = infer_eta_from_npz(train_npz, device)
    model = nets.PSNN(
        dim_theta=2,
        dim_u=2,
        embed_dim=8,
        width=[30, 20],
        depth=[4, 3],
        eta=eta,
    ).to(device)
    model.load_state_dict(torch.load(phi_ckpt, map_location=device))
    model.eval()
    return model


def _infer_mlp_shape(state_dict: dict) -> tuple[int, int, int]:
    linear_keys = [k for k in state_dict.keys() if k.endswith(".weight") and ".net." in k]
    linear_keys = sorted(linear_keys, key=lambda k: int(k.split(".")[-2]))
    if not linear_keys:
        raise ValueError("No linear weights found in count classifier state_dict.")

    first_w = state_dict[linear_keys[0]]
    last_w = state_dict[linear_keys[-1]]
    width = int(first_w.shape[0])
    in_dim = int(first_w.shape[1])
    out_dim = int(last_w.shape[0])
    num_linear_layers = len(linear_keys)
    depth = max(0, num_linear_layers - 1)
    return in_dim, width, depth, out_dim


def load_count_classifier(count_ckpt: str, device: torch.device) -> tuple[torch.nn.Module, int]:
    ckpt = torch.load(count_ckpt, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        num_classes = int(ckpt.get("num_classes", 2))
    else:
        state_dict = ckpt
        num_classes = 2
    in_dim, width, depth, out_dim = _infer_mlp_shape(state_dict)
    print(f"Loaded count classifier: in_dim={in_dim}, width={width}, depth={depth}, out_dim={out_dim}")
    num_classes = out_dim
    model = nets.ThetaCountClassifier(dim_theta=in_dim, num_classes=num_classes, width=width, depth=depth).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, num_classes


def load_stability_classifier(stab_ckpt: str, device: torch.device) -> torch.nn.Module:
    model = nets.StabilityClassifier(
        dim_theta=2,
        dim_u=2,
        embed_dim=8,
        width=[64, 64],
        depth=[2, 2],
    ).to(device)
    model.load_state_dict(torch.load(stab_ckpt, map_location=device))
    model.eval()
    return model


def predict_num_solutions(model: torch.nn.Module, num_classes: int, theta: np.ndarray, device: torch.device) -> int:
    theta_t = torch.from_numpy(theta.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(theta_t).squeeze(0)
        probs = torch.softmax(logits, dim=0)
        label = int(torch.argmax(probs).item())
    if num_classes == 2:
        return 0 if label == 0 else 2
    return label


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
    device: torch.device = torch.device("cpu"),
):
    if expected_count == 0:
        return np.empty((0, U_all.shape[1])), np.empty((U_all.shape[0],))

    U_all_t = torch.from_numpy(U_all.astype(np.float32)).to(device)
    theta_t = torch.from_numpy(theta.astype(np.float32)).to(device).unsqueeze(0)
    theta_t = theta_t.repeat(U_all_t.shape[0], 1)

    with torch.no_grad():
        scores_t = phi_model(U_all_t, theta_t).flatten()

    scores = scores_t.cpu().numpy()
    U_collected = U_all[scores >= L_cut, :]
    if U_collected.size == 0:
        return np.empty((0, U_all.shape[1])), scores

    centers = enforce_cluster_count(U_collected, expected_count)

    return centers, scores


def classify_stability(
    stab_model: torch.nn.Module, theta: np.ndarray, centers: np.ndarray, device: torch.device
) -> np.ndarray:
    if centers is None or centers.size == 0:
        return np.empty((0,), dtype=float)
    centers_t = torch.from_numpy(centers.astype(np.float32)).to(device)
    theta_t = torch.from_numpy(theta.astype(np.float32)).to(device).unsqueeze(0)
    theta_t = theta_t.repeat(centers_t.shape[0], 1)
    with torch.no_grad():
        probs = stab_model(centers_t, theta_t).view(-1).cpu().numpy()
    return probs


def parse_thetas(theta_args):
    if not theta_args:
        return [np.array([0.02, 0.05], dtype=np.float32)]
    return [np.array([float(f), float(k)], dtype=np.float32) for f, k in theta_args]


def build_solution_map_with_stability(
    phi_model: torch.nn.Module,
    count_model: torch.nn.Module,
    num_classes: int,
    stab_model: torch.nn.Module,
    *,
    m: int = 100,
    L_cut: float = 0.48,
    device: torch.device,
    out_path: str,
    grid_n: int = 100,
):
    U_all, _v_axis, _u_axis = make_u_grid(m=m)
    Theta_f_space = np.linspace(0.00, 0.30, grid_n)
    Theta_k_space = np.linspace(0.00, 0.08, grid_n)
    Theta_grid = np.array(np.meshgrid(Theta_f_space, Theta_k_space)).T.reshape(-1, 2)

    pts_zero = []
    pts_two_both_stable = []
    pts_two_mixed = []
    pts_two_both_unstable = []
    pts_other = []

    iterable = range(Theta_grid.shape[0])
    if tqdm is not None:
        iterable = tqdm.tqdm(iterable, desc="Building stability solution map")

    for i in iterable:
        theta_i = Theta_grid[i]
        expected = predict_num_solutions(count_model, num_classes, theta_i, device)
        centers, _scores = locate_solutions_for_theta(
            phi_model=phi_model,
            theta=theta_i,
            U_all=U_all,
            L_cut=L_cut,
            expected_count=expected,
            device=device,
        )

        n = int(centers.shape[0])
        if n == 0:
            pts_zero.append(theta_i)
            continue

        if n == 2:
            stab_probs = classify_stability(stab_model, theta_i, centers, device)
            stable_count = int(np.sum(stab_probs >= 0.5))
            if stable_count == 2:
                pts_two_both_stable.append(theta_i)
            elif stable_count == 1:
                pts_two_mixed.append(theta_i)
            else:
                pts_two_both_unstable.append(theta_i)
        else:
            pts_other.append(theta_i)

    def _scatter(points, *, color, label, marker='o', s=18):
        if len(points) == 0:
            return
        arr = np.asarray(points)
        plt.scatter(arr[:, 0], arr[:, 1], c=color, s=s, marker=marker, label=label)

    plt.figure(figsize=(7, 4.5))
    _scatter(pts_zero, color='royalblue', label='0 solutions')
    _scatter(pts_two_both_stable, color='limegreen', label='2 solutions: both stable')
    _scatter(pts_two_mixed, color='darkorange', label='2 solutions: 1 stable + 1 unstable', marker='s')
    _scatter(pts_two_both_unstable, color='crimson', label='2 solutions: both unstable', marker='x')
    _scatter(pts_other, color='black', label='other (#solutions != 0 or 2)', marker='.')

    f_space = np.linspace(0.00, 0.255, 200)
    k_dividing = np.sqrt(f_space) / 2.0 - f_space
    plt.plot(f_space, k_dividing, c='red', linewidth=1.5, label='analytic divider')

    plt.xlabel('f')
    plt.ylabel('k')
    plt.title('Solution map with stability (classifier + PSNN)')
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"plot saved to <{out_path}>")


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
        description="Locate Gray-Scott solutions using classifiers + PSNN.",
        parents=[pre],
    )
    parser.add_argument("--theta", action="append", nargs=2, metavar=("F", "K"), help="Add a theta pair.")
    parser.add_argument("--m", type=int, default=cfg_get(loc_cfg, "m", 100), help="Grid resolution.")
    parser.add_argument("--L_cut", type=float, default=cfg_get(loc_cfg, "L_cut", 0.48), help="Phi cutoff.")
    parser.add_argument("--make_map", action="store_true", default=bool(cfg_get(loc_cfg, "make_map", False)), help="Generate solution map with stability info.")
    parser.add_argument("--grid_n", type=int, default=cfg_get(loc_cfg, "grid_n", 100), help="Parameter grid resolution for map.")
    args = parser.parse_args(remaining)

    data_dir = resolve_path(exp_dir, cfg_get(path_cfg, "data_dir", "data"))
    train_npz = resolve_path(exp_dir, cfg_get(path_cfg, "train_npz", data_dir / "gray_scott_data_train.npz"))
    phi_ckpt = resolve_path(exp_dir, cfg_get(path_cfg, "phi_ckpt", "psnn_phi.pt"))
    compat_ckpt = resolve_path(exp_dir, cfg_get(path_cfg, "compat_phi_ckpt", "psnn_final.pt"))
    count_ckpt = resolve_path(exp_dir, cfg_get(path_cfg, "count_ckpt", "psnn_numsol.pt"))
    stab_ckpt = resolve_path(exp_dir, cfg_get(path_cfg, "stability_ckpt", "psnn_stability_cls.pt"))

    if not os.path.exists(phi_ckpt) and os.path.exists(compat_ckpt):
        phi_ckpt = compat_ckpt

    for path in (train_npz, phi_ckpt, count_ckpt, stab_ckpt):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required file: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phi_model = load_phi_model(phi_ckpt, train_npz, device)
    count_model, num_classes = load_count_classifier(count_ckpt, device)
    stab_model = load_stability_classifier(stab_ckpt, device)

    U_all, _v_axis, _u_axis = make_u_grid(m=args.m)
    if not args.theta and cfg_get(loc_cfg, "thetas", None):
        args.theta = [tuple(t) for t in cfg_get(loc_cfg, "thetas", [])]
    thetas = parse_thetas(args.theta)

    if args.make_map:
        out_path = resolve_path(exp_dir, cfg_get(path_cfg, "solution_map", "solution_map_with_stability_info.png"))
        build_solution_map_with_stability(
            phi_model=phi_model,
            count_model=count_model,
            num_classes=num_classes,
            stab_model=stab_model,
            m=args.m,
            L_cut=args.L_cut,
            device=device,
            out_path=out_path,
            grid_n=args.grid_n,
        )
        return

    for theta in thetas:
        expected = predict_num_solutions(count_model, num_classes, theta, device)
        centers, _scores = locate_solutions_for_theta(
            phi_model=phi_model,
            theta=theta,
            U_all=U_all,
            L_cut=args.L_cut,
            expected_count=expected,
            device=device,
        )
        stab_probs = classify_stability(stab_model, theta, centers, device)

        print(f"Theta={theta.tolist()} | predicted solutions={expected}")
        if centers.size == 0:
            print("  No solutions detected.")
            continue
        for i, (center, prob) in enumerate(zip(centers, stab_probs), start=1):
            stable = bool(prob >= 0.5)
            print(f"  Sol {i}: u={center.tolist()} | stable_prob={prob:.4f} | stable={stable}")


if __name__ == "__main__":
    main()
