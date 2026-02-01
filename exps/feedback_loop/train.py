import argparse
import os
import sys
import torch
import numpy as np
import random


def _add_repo_root_to_syspath():
	exp_dir = os.path.dirname(os.path.abspath(__file__))          # .../exps/feedback_loop
	repo_root = os.path.abspath(os.path.join(exp_dir, "../.."))   # .../ (repo root)
	if repo_root not in sys.path:
		sys.path.insert(0, repo_root)
	return exp_dir, repo_root


def run_feedback_loop(
	train_npz: str,
	test_npz: str,
	*,
	batch_size: int = 256,
	epochs: int = 100,
	lr: float = 1e-3,
	out_path: str = "psnn_phi.pt",
	num_workers: int = 0,
	dim_theta: int = 4,
	dim_u: int = 2,
	embed_dim: int = 8,
	width: list[int] = (30, 20),
	depth: list[int] = (4, 3),
	eta_scale: float = 1.5,
	eta_cap: float = 0.01,
):
	"""Train a PSNN to learn Phi(theta, u) for the feedback-loop example."""

	import torch
	from psnn import nets, datasets
	from psnn.trainer import train_model

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	train_loader, test_loader = datasets.make_loaders(
		train_npz,
		test_npz,
		batch_size=batch_size,
		num_workers=num_workers,
		device=device,
	)

	# Same heuristic as gray_scott; clamp to a small positive value.
	phi_max = float(train_loader.dataset.Phi.max().item())
	eta = min(float(eta_cap), float(eta_scale) * (phi_max - 1.0))
	print(f"Using eta={eta:.3e} (Phi.max={phi_max:.3e})")

	model = nets.PSNN(
		dim_theta=dim_theta,
		dim_u=dim_u,
		embed_dim=embed_dim,
		width=list(width),
		depth=list(depth),
		eta=eta,
	).to(device)

	train_model(
		model,
		train_loader,
		val_loader=test_loader,
		epochs=epochs,
		lr=lr,
		device=device,
	)

	torch.save(model.state_dict(), out_path)
	return model


def _make_class_weights(labels, num_classes, device):
	counts = torch.bincount(labels.cpu(), minlength=num_classes).float()
	weights = counts.sum() / (num_classes * counts.clamp_min(1.0))
	weights[counts == 0] = 0.0
	return weights.to(device), counts


def train_count_classifier(
	model,
	train_loader,
	val_loader=None,
	*,
	epochs=100,
	lr=1e-3,
	device=None,
):
	import torch
	from psnn import nets, datasets
	from psnn.trainer import train_model

	if device is None:
		device = next(model.parameters()).device

	labels = train_loader.dataset.Labels
	if hasattr(train_loader.dataset, "num_classes"):
		num_classes = int(train_loader.dataset.num_classes)
	else:
		num_classes = int(labels.max().item()) + 1 if labels.numel() > 0 else 0
	class_weights, counts = _make_class_weights(labels, num_classes, device)
	print(f"Count label counts: {counts.tolist()}")

	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

	for epoch in range(1, epochs + 1):
		model.train()
		total_loss = 0.0
		total_acc = 0.0

		for Theta, y in train_loader:
			optimizer.zero_grad()
			logits = model(Theta)
			loss = criterion(logits, y)
			loss.backward()
			optimizer.step()
			total_loss += loss.item() * Theta.size(0)
			total_acc += (logits.argmax(dim=1) == y).float().sum().item()

		train_loss = total_loss / len(train_loader.dataset)
		train_acc = total_acc / len(train_loader.dataset)

		if val_loader is not None:
			val_loss, val_acc = eval_count_classifier(model, val_loader, criterion)
			print(
				f"Epoch {epoch:03d} | "
				f"Train CE {train_loss:.6f} Acc {train_acc:.4f} | "
				f"Val CE {val_loss:.6f} Acc {val_acc:.4f}"
			)
		else:
			print(f"Epoch {epoch:03d} | Train CE {train_loss:.6f} Acc {train_acc:.4f}")

	return model, num_classes


def eval_count_classifier(model, loader, criterion):
	model.eval()
	total_loss = 0.0
	total_acc = 0.0
	with torch.no_grad():
		for Theta, y in loader:
			logits = model(Theta)
			loss = criterion(logits, y)
			total_loss += loss.item() * Theta.size(0)
			total_acc += (logits.argmax(dim=1) == y).float().sum().item()
	return total_loss / len(loader.dataset), total_acc / len(loader.dataset)


def train_stability_classifier(
	model,
	train_loader,
	val_loader=None,
	*,
	epochs=100,
	lr=1e-3,
	device=None,
):
	if device is None:
		device = next(model.parameters()).device

	labels = train_loader.dataset.Labels
	counts = torch.bincount(labels.cpu(), minlength=2).float()
	total = counts.sum().clamp_min(1.0)
	w_pos = (total / (2.0 * counts[1].clamp_min(1.0))).to(device)
	w_neg = (total / (2.0 * counts[0].clamp_min(1.0))).to(device)
	print(f"Stability label counts: {counts.tolist()}")

	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	criterion = torch.nn.BCELoss(reduction="none")

	for epoch in range(1, epochs + 1):
		model.train()
		total_loss = 0.0
		total_acc = 0.0

		for U, Theta, y in train_loader:
			optimizer.zero_grad()
			probs = model(U, Theta).view(-1)
			y_float = y.float()
			weights = torch.where(y_float > 0.5, w_pos, w_neg)
			loss = (criterion(probs, y_float) * weights).mean()
			loss.backward()
			optimizer.step()
			total_loss += loss.item() * U.size(0)
			total_acc += ((probs >= 0.5) == (y_float >= 0.5)).float().sum().item()

		train_loss = total_loss / len(train_loader.dataset)
		train_acc = total_acc / len(train_loader.dataset)

		if val_loader is not None:
			val_loss, val_acc = eval_stability_classifier(model, val_loader, criterion)
			print(
				f"Epoch {epoch:03d} | "
				f"Train CE {train_loss:.6f} Acc {train_acc:.4f} | "
				f"Val CE {val_loss:.6f} Acc {val_acc:.4f}"
			)
		else:
			print(f"Epoch {epoch:03d} | Train CE {train_loss:.6f} Acc {train_acc:.4f}")

	return model


def eval_stability_classifier(model, loader, criterion):
	model.eval()
	total_loss = 0.0
	total_acc = 0.0
	with torch.no_grad():
		for U, Theta, y in loader:
			probs = model(U, Theta).view(-1)
			y_float = y.float()
			loss = criterion(probs, y_float).mean()
			total_loss += loss.item() * U.size(0)
			total_acc += ((probs >= 0.5) == (y_float >= 0.5)).float().sum().item()
	return total_loss / len(loader.dataset), total_acc / len(loader.dataset)


def main():
	import torch
	exp_dir, _repo_root = _add_repo_root_to_syspath()

	from psnn import nets, datasets
	from psnn.trainer import train_model
	from psnn.config import cfg_get, load_yaml, resolve_path

	parser = argparse.ArgumentParser(description="Train feedback-loop models from config.")
	parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
	args = parser.parse_args()

	default_cfg = os.path.join(exp_dir, "config.yaml")
	cfg = {}
	cfg_path = args.config or (default_cfg if os.path.exists(default_cfg) else None)
	if cfg_path:
		cfg = load_yaml(cfg_path)

	tr = cfg_get(cfg, "training", {})
	paths = cfg_get(tr, "paths", {})

	data_dir = resolve_path(exp_dir, cfg_get(paths, "data_dir", "data"))
	out_dir = resolve_path(exp_dir, cfg_get(paths, "out_dir", "."))

	train_npz = resolve_path(exp_dir, cfg_get(paths, "train_npz", data_dir / "feedback_loop_data_train.npz"))
	test_npz = resolve_path(exp_dir, cfg_get(paths, "test_npz", data_dir / "feedback_loop_data_test.npz"))

	obs_train_pkl = resolve_path(exp_dir, cfg_get(paths, "obs_train_pkl", data_dir / "feedback_loop_obs_train.pkl"))
	obs_test_pkl = resolve_path(exp_dir, cfg_get(paths, "obs_test_pkl", data_dir / "feedback_loop_obs_test.pkl"))

	out_phi = resolve_path(out_dir, cfg_get(paths, "phi_ckpt", "psnn_phi.pt"))
	out_count = resolve_path(out_dir, cfg_get(paths, "count_ckpt", "psnn_numsol.pt"))
	out_stability_cls = resolve_path(out_dir, cfg_get(paths, "stability_ckpt", "psnn_stability_cls.pt"))

	if not os.path.exists(train_npz):
		raise FileNotFoundError(f"Missing train file: {train_npz}")
	if not os.path.exists(test_npz):
		raise FileNotFoundError(f"Missing test file: {test_npz}")

	device_cfg = str(cfg_get(tr, "device", "auto")).lower()
	if device_cfg == "cpu":
		device = torch.device("cpu")
	elif device_cfg == "cuda":
		device = torch.device("cuda")
	else:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	num_workers = int(cfg_get(tr, "num_workers", 0))

	if bool(cfg_get(tr, "phi.enabled", False)):
		phi_batch_size = int(cfg_get(tr, "phi.batch_size", 256))
		phi_seed = int(cfg_get(tr, "phi.seed", 123))
		torch.manual_seed(phi_seed)
		np.random.seed(phi_seed)
		random.seed(phi_seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(phi_seed)
		print("Training Phi model...")
		run_feedback_loop(
			train_npz=str(train_npz),
			test_npz=str(test_npz),
			batch_size=phi_batch_size,
			epochs=int(cfg_get(tr, "phi.epochs", 100)),
			lr=float(cfg_get(tr, "phi.lr", 1e-3)),
			out_path=str(out_phi),
			num_workers=num_workers,
			dim_theta=4,
			dim_u=2,
			embed_dim=int(cfg_get(tr, "phi.model.embed_dim", 8)),
			width=cfg_get(tr, "phi.model.width", [30, 20]),
			depth=cfg_get(tr, "phi.model.depth", [4, 3]),
			eta_scale=float(cfg_get(tr, "phi.eta.scale", 1.5)),
			eta_cap=float(cfg_get(tr, "phi.eta.cap", 0.01)),
		)
		print(f"Saved Phi model to: {out_phi}")

	# --- train count classifier: Theta -> number of solutions ---
	if bool(cfg_get(tr, "count.enabled", True)) and os.path.exists(obs_train_pkl) and os.path.exists(obs_test_pkl):
		print("Training count classifier (Theta -> #solutions)...")
		count_batch_size = int(cfg_get(tr, "count.batch_size", 256))
		count_seed = int(cfg_get(tr, "count.seed", 123))
		torch.manual_seed(count_seed)
		np.random.seed(count_seed)
		random.seed(count_seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(count_seed)
		count_train_loader, count_test_loader = datasets.make_obs_loaders(
			str(obs_train_pkl),
			str(obs_test_pkl),
			batch_size=count_batch_size,
			num_workers=num_workers,
			device=device,
			mode="count",
		)
		if len(count_train_loader.dataset) > 0:
			num_classes = int(getattr(count_train_loader.dataset, "num_classes", int(count_train_loader.dataset.Labels.max().item()) + 1))
			model_count = nets.ThetaCountClassifier(
				dim_theta=4,
				num_classes=num_classes,
				width=int(cfg_get(tr, "count.model.width", 64)),
				depth=int(cfg_get(tr, "count.model.depth", 3)),
			).to(device)
			model_count, num_classes = train_count_classifier(
				model_count,
				count_train_loader,
				val_loader=count_test_loader,
				epochs=int(cfg_get(tr, "count.epochs", 100)),
				lr=float(cfg_get(tr, "count.lr", 1e-3)),
				device=device,
			)
			torch.save(
				{
					"state_dict": model_count.state_dict(),
					"num_classes": num_classes,
					"class_values": getattr(count_train_loader.dataset, "ClassValues", None),
				},
				out_count,
			)
			print(f"Saved count classifier to: {out_count}")
		else:
			print("Skipping count classifier: no observation data found.")
	else:
		print("Skipping count classifier: observation .pkl not found.")

	# --- train stability classifier: (Theta, U) -> stable/unstable ---
	if bool(cfg_get(tr, "stability_classifier.enabled", True)) and os.path.exists(obs_train_pkl) and os.path.exists(obs_test_pkl):
		print("Training stability classifier (Theta, U -> stable)...")
		stab_batch_size = int(cfg_get(tr, "stability_classifier.batch_size", 256))
		stab_seed = int(cfg_get(tr, "stability_classifier.seed", 123))
		torch.manual_seed(stab_seed)
		np.random.seed(stab_seed)
		random.seed(stab_seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(stab_seed)
		stab_train_loader, stab_test_loader = datasets.make_obs_loaders(
			str(obs_train_pkl),
			str(obs_test_pkl),
			batch_size=stab_batch_size,
			num_workers=num_workers,
			device=device,
			mode="stability",
		)
		if len(stab_train_loader.dataset) > 0:
			model_stability = nets.StabilityClassifier(
				dim_theta=4,
				dim_u=2,
				embed_dim=int(cfg_get(tr, "stability_classifier.model.embed_dim", 8)),
				width=cfg_get(tr, "stability_classifier.model.width", [32, 32]),
				depth=cfg_get(tr, "stability_classifier.model.depth", [2, 2]),
			).to(device)
			train_stability_classifier(
				model_stability,
				stab_train_loader,
				val_loader=stab_test_loader,
				epochs=int(cfg_get(tr, "stability_classifier.epochs", 100)),
				lr=float(cfg_get(tr, "stability_classifier.lr", 1e-3)),
				device=device,
			)
			torch.save(model_stability.state_dict(), out_stability_cls)
			print(f"Saved stability classifier to: {out_stability_cls}")
		else:
			print("Skipping stability classifier: no stability observations found.")
	else:
		print("Skipping stability classifier: observation .pkl not found.")


if __name__ == "__main__":
	main()
