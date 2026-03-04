from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from psnn import datasets, nets
from psnn.config import cfg_get, load_yaml, resolve_path
from psnn.trainer import train_count_classifier, train_phi_model, train_stability_classifier
from psnn.utils import infer_psnn_arch, infer_stability_arch, infer_theta_count_arch


def _set_all_seeds(seed: int) -> None:
	torch.manual_seed(seed)
	np.random.seed(seed)
	try:
		import random

		random.seed(seed)
	except Exception:
		pass
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def _device_from_cfg(device_cfg: str) -> torch.device:
	device_cfg = str(device_cfg or "auto").lower()
	if device_cfg == "cpu":
		return torch.device("cpu")
	if device_cfg == "cuda":
		return torch.device("cuda")
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _maybe_compile(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
	if not enabled:
		return model
	try:
		return torch.compile(model)  # type: ignore[attr-defined]
	except Exception:
		return model


def run_from_loaded_config(cfg: dict[str, Any], base_dir: str | Path) -> None:
	base_dir = Path(base_dir).expanduser().resolve()

	tr = cfg_get(cfg, "training", {})
	paths = cfg_get(tr, "paths", {})
	data_outputs = cfg_get(cfg, "data_generation.outputs", {})

	global_seed = int(cfg_get(cfg, "seed", cfg_get(tr, "seed", 123)))

	def seed_for(section: str) -> int:
		return int(cfg_get(tr, f"{section}.seed", global_seed))

	device = _device_from_cfg(cfg_get(tr, "device", "auto"))
	num_workers = int(cfg_get(tr, "num_workers", 0))
	compile_enabled = bool(cfg_get(tr, "compile", False))

	data_dir = resolve_path(base_dir, cfg_get(paths, "data_dir", cfg_get(data_outputs, "out_dir", "data")))
	out_dir = resolve_path(base_dir, cfg_get(paths, "out_dir", "."))
	out_dir.mkdir(parents=True, exist_ok=True)

	train_npz = resolve_path(
		base_dir,
		cfg_get(
			paths,
			"train_npz",
			data_dir / cfg_get(data_outputs, "data_train_npz", "data_train.npz"),
		),
	)
	test_npz = resolve_path(
		base_dir,
		cfg_get(
			paths,
			"test_npz",
			data_dir / cfg_get(data_outputs, "data_test_npz", "data_test.npz"),
		),
	)
	obs_train_pkl = resolve_path(
		base_dir,
		cfg_get(
			paths,
			"obs_train_pkl",
			data_dir / cfg_get(data_outputs, "obs_train_pkl", "obs_train.pkl"),
		),
	)
	obs_test_pkl = resolve_path(
		base_dir,
		cfg_get(
			paths,
			"obs_test_pkl",
			data_dir / cfg_get(data_outputs, "obs_test_pkl", "obs_test.pkl"),
		),
	)

	out_phi = resolve_path(out_dir, cfg_get(paths, "phi_ckpt", "psnn_phi.pt"))
	out_count = resolve_path(out_dir, cfg_get(paths, "count_ckpt", "psnn_numsol.pt"))
	out_stability = resolve_path(out_dir, cfg_get(paths, "stability_ckpt", "psnn_stability_cls.pt"))
	out_compat_phi = cfg_get(paths, "compat_phi_ckpt", None)
	out_compat_phi = resolve_path(out_dir, out_compat_phi) if out_compat_phi else None

	print(f"Base dir: {base_dir}")
	print(f"Device: {device}")

	# --- Phi model ---
	if bool(cfg_get(tr, "phi.enabled", False)):
		if not train_npz.exists():
			raise FileNotFoundError(f"Missing train file: {train_npz}")
		if not test_npz.exists():
			raise FileNotFoundError(f"Missing test file: {test_npz}")

		print("Training Phi model...")
		phi_seed = seed_for("phi")
		_set_all_seeds(phi_seed)

		phi_batch_size = int(cfg_get(tr, "phi.batch_size", 256))
		phi_epochs = int(cfg_get(tr, "phi.epochs", 100))
		phi_lr = float(cfg_get(tr, "phi.lr", 1e-3))
		eta_scale = float(cfg_get(tr, "phi.eta.scale", 1.5))
		eta_cap = float(cfg_get(tr, "phi.eta.cap", 0.01))
		phi_compile = bool(cfg_get(tr, "phi.compile", compile_enabled))

		train_loader, test_loader = datasets.make_loaders(
			str(train_npz),
			str(test_npz),
			batch_size=phi_batch_size,
			num_workers=num_workers,
			device=device,
		)

		dim_theta = int(train_loader.dataset.Theta.shape[1])
		dim_u = int(train_loader.dataset.U.shape[1])
		phi_max = float(train_loader.dataset.Phi.max().item())
		eta = min(float(eta_cap), float(eta_scale) * (phi_max - 1.0))
		print(f"Using eta={eta:.3e} (Phi.max={phi_max:.3e})")

		model_phi = nets.PSNN(
			dim_theta=dim_theta,
			dim_u=dim_u,
			embed_dim=int(cfg_get(tr, "phi.model.embed_dim", 8)),
			width=cfg_get(tr, "phi.model.width", [30, 20]),
			depth=cfg_get(tr, "phi.model.depth", [4, 3]),
			eta=eta,
		).to(device)
		model_phi = _maybe_compile(model_phi, phi_compile)

		train_phi_model(
			model_phi,
			train_loader,
			val_loader=test_loader,
			epochs=phi_epochs,
			lr=phi_lr,
			device=device,
		)

		torch.save(
			{
				"format_version": 1,
				"kind": "phi",
				"state_dict": model_phi.state_dict(),
				"model": infer_psnn_arch(model_phi),
				"data": {
					"train_npz": str(train_npz),
					"test_npz": str(test_npz),
					"phi_max": float(phi_max),
				},
				"train": {
					"epochs": int(phi_epochs),
					"lr": float(phi_lr),
					"eta_scale": float(eta_scale),
					"eta_cap": float(eta_cap),
					"batch_size": int(phi_batch_size),
					"device": str(device),
					"seed": int(phi_seed),
				},
			},
			out_phi,
		)
		if out_compat_phi is not None:
			torch.save(model_phi.state_dict(), out_compat_phi)
		print(f"Saved Phi model to: {out_phi}")

	# --- Count classifier ---
	if bool(cfg_get(tr, "count.enabled", True)):
		if obs_train_pkl.exists() and obs_test_pkl.exists():
			print("Training count classifier (Theta -> #solutions)...")
			count_seed = seed_for("count")
			_set_all_seeds(count_seed)
			count_batch_size = int(cfg_get(tr, "count.batch_size", 256))
			count_epochs = int(cfg_get(tr, "count.epochs", 100))
			count_lr = float(cfg_get(tr, "count.lr", 1e-3))

			count_train_loader, count_test_loader = datasets.make_obs_loaders(
				str(obs_train_pkl),
				str(obs_test_pkl),
				batch_size=count_batch_size,
				num_workers=num_workers,
				device=device,
				mode="count",
			)
			if len(count_train_loader.dataset) > 0:
				dim_theta = int(count_train_loader.dataset.Theta.shape[1])
				num_classes = int(getattr(count_train_loader.dataset, "num_classes", 0))

				model_count = nets.ThetaCountClassifier(
					dim_theta=dim_theta,
					num_classes=num_classes,
					width=int(cfg_get(tr, "count.model.width", 64)),
					depth=int(cfg_get(tr, "count.model.depth", 3)),
				).to(device)

				model_count, num_classes = train_count_classifier(
					model_count,
					count_train_loader,
					val_loader=count_test_loader,
					epochs=count_epochs,
					lr=count_lr,
					device=device,
				)

				class_values = getattr(count_train_loader.dataset, "ClassValues", None)
				if isinstance(class_values, torch.Tensor):
					class_values = class_values.detach().cpu().numpy().astype(np.int64).tolist()
				label_counts = (
					torch.bincount(count_train_loader.dataset.Labels.detach().cpu(), minlength=int(num_classes))
					.to(torch.int64)
					.tolist()
				)

				torch.save(
					{
						"format_version": 1,
						"kind": "count_classifier",
						"state_dict": model_count.state_dict(),
						"model": infer_theta_count_arch(model_count),
						"data": {
							"obs_train_pkl": str(obs_train_pkl),
							"obs_test_pkl": str(obs_test_pkl),
							"class_values": class_values,
							"label_counts": label_counts,
						},
						"train": {
							"epochs": int(count_epochs),
							"lr": float(count_lr),
							"batch_size": int(count_batch_size),
							"device": str(device),
							"seed": int(count_seed),
						},
					},
					out_count,
				)
				print(f"Saved count classifier to: {out_count}")
			else:
				print("Skipping count classifier: no observation data found.")
		else:
			print("Skipping count classifier: observation .pkl not found.")

	# --- Stability classifier ---
	if bool(cfg_get(tr, "stability_classifier.enabled", True)):
		if obs_train_pkl.exists() and obs_test_pkl.exists():
			print("Training stability classifier (Theta, U -> stable)...")
			stab_seed = seed_for("stability_classifier")
			_set_all_seeds(stab_seed)
			stab_batch_size = int(cfg_get(tr, "stability_classifier.batch_size", 256))
			stab_epochs = int(cfg_get(tr, "stability_classifier.epochs", 100))
			stab_lr = float(cfg_get(tr, "stability_classifier.lr", 1e-3))

			stab_train_loader, stab_test_loader = datasets.make_obs_loaders(
				str(obs_train_pkl),
				str(obs_test_pkl),
				batch_size=stab_batch_size,
				num_workers=num_workers,
				device=device,
				mode="stability",
			)
			if len(stab_train_loader.dataset) > 0:
				dim_theta = int(stab_train_loader.dataset.Theta.shape[1])
				dim_u = int(stab_train_loader.dataset.U.shape[1])

				model_stability = nets.StabilityClassifier(
					dim_theta=dim_theta,
					dim_u=dim_u,
					embed_dim=int(cfg_get(tr, "stability_classifier.model.embed_dim", 8)),
					width=cfg_get(tr, "stability_classifier.model.width", [32, 32]),
					depth=cfg_get(tr, "stability_classifier.model.depth", [2, 2]),
				).to(device)

				train_stability_classifier(
					model_stability,
					stab_train_loader,
					val_loader=stab_test_loader,
					epochs=stab_epochs,
					lr=stab_lr,
					device=device,
				)

				label_counts = (
					torch.bincount(stab_train_loader.dataset.Labels.detach().cpu(), minlength=2)
					.to(torch.int64)
					.tolist()
				)

				torch.save(
					{
						"format_version": 1,
						"kind": "stability_classifier",
						"state_dict": model_stability.state_dict(),
						"model": infer_stability_arch(model_stability),
						"data": {
							"obs_train_pkl": str(obs_train_pkl),
							"obs_test_pkl": str(obs_test_pkl),
							"label_counts": label_counts,
						},
						"train": {
							"epochs": int(stab_epochs),
							"lr": float(stab_lr),
							"batch_size": int(stab_batch_size),
							"device": str(device),
							"seed": int(stab_seed),
						},
					},
					out_stability,
				)
				print(f"Saved stability classifier to: {out_stability}")
			else:
				print("Skipping stability classifier: no stability observations found.")
		else:
			print("Skipping stability classifier: observation .pkl not found.")


def run_from_config(config_path: str | Path) -> None:
	config_path = Path(config_path).expanduser().resolve()
	cfg: dict[str, Any] = load_yaml(config_path)
	run_from_loaded_config(cfg, base_dir=config_path.parent)


def main() -> None:
	parser = argparse.ArgumentParser(description="Shared PSNN training runner")
	parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
	args = parser.parse_args()
	run_from_config(args.config)


if __name__ == "__main__":
	main()
