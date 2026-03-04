from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Tuple

import numpy as np
import torch

from psnn import nets


def _load_torch_checkpoint(path: str | Path, device: torch.device):
	path = str(path)
	try:
		return torch.load(path, map_location=device, weights_only=True)
	except TypeError:  # older torch
		return torch.load(path, map_location=device)


def _ckpt_get_state_dict(ckpt: Any) -> dict:
	if isinstance(ckpt, dict) and "state_dict" in ckpt:
		return ckpt["state_dict"]
	if isinstance(ckpt, dict):
		# Raw state_dict already
		return ckpt
	raise TypeError("Checkpoint is not a dict-like state_dict")


def _normalize_state_dict_keys(state_dict: dict) -> dict:
	"""Normalize common wrappers: torch.compile (_orig_mod.), DDP (module.)."""
	if not state_dict:
		return state_dict
	prefixes = ("_orig_mod.", "module.")
	out: dict[str, Any] = {}
	for k, v in state_dict.items():
		kk = k
		for p in prefixes:
			if kk.startswith(p):
				kk = kk[len(p) :]
		out[kk] = v
	return out


def load_phi_model(
	phi_ckpt: str | Path,
	device: torch.device,
) -> torch.nn.Module:
	ckpt = _load_torch_checkpoint(phi_ckpt, device)
	state_dict = _normalize_state_dict_keys(_ckpt_get_state_dict(ckpt))

	model_meta = ckpt.get("model", None) if isinstance(ckpt, dict) else None
	eta_in_ckpt = model_meta.get("eta", None) if isinstance(model_meta, dict) else None

	if isinstance(model_meta, dict) and eta_in_ckpt is not None:
		eta = float(eta_in_ckpt)
		model = nets.PSNN(
			dim_theta=int(model_meta.get("dim_theta", 2)),
			dim_u=int(model_meta.get("dim_u", 2)),
			embed_dim=int(model_meta.get("embed_dim", 8)),
			width=list(model_meta.get("width", [30, 20])),
			depth=list(model_meta.get("depth", [4, 3])),
			eta=eta,
		).to(device)
	else:
		raise ValueError(
			"Phi checkpoint is missing model/eta metadata. Re-train to produce a metadata-rich checkpoint "
			"(preferred), or load the model with a custom constructor that provides eta explicitly."
		)

	model.load_state_dict(state_dict)
	model.eval()
	return model


def load_count_classifier(
	count_ckpt: str | Path, device: torch.device
) -> Tuple[torch.nn.Module, np.ndarray]:
	ckpt = _load_torch_checkpoint(count_ckpt, device)
	state_dict = _normalize_state_dict_keys(_ckpt_get_state_dict(ckpt))
	model_meta = ckpt.get("model", None) if isinstance(ckpt, dict) else None

	# class_values is optional; for non-contiguous counts we store decode values.
	class_values = None
	if isinstance(ckpt, dict):
		class_values = ckpt.get("class_values", None)
		if class_values is None:
			data_meta = ckpt.get("data", None)
			if isinstance(data_meta, dict):
				class_values = data_meta.get("class_values", None)

	if not isinstance(model_meta, dict):
		raise ValueError(
			"Count-classifier checkpoint is missing 'model' metadata. Re-train to generate a metadata-rich checkpoint."
		)

	in_dim = int(model_meta.get("dim_theta"))
	out_dim = int(model_meta.get("num_classes"))
	width = int(model_meta.get("width"))
	depth = int(model_meta.get("depth"))

	model = nets.ThetaCountClassifier(
		dim_theta=in_dim,
		num_classes=int(out_dim),
		width=int(width),
		depth=int(depth),
	).to(device)
	model.load_state_dict(state_dict)
	model.eval()

	if class_values is None:
		class_values_np = np.arange(int(out_dim), dtype=np.int64)
	else:
		if isinstance(class_values, torch.Tensor):
			class_values_np = class_values.detach().cpu().numpy().astype(np.int64)
		else:
			class_values_np = np.asarray(class_values, dtype=np.int64)
		if class_values_np.shape[0] != int(out_dim):
			class_values_np = np.arange(int(out_dim), dtype=np.int64)

	return model, class_values_np


def load_stability_classifier(stab_ckpt: str | Path, device: torch.device) -> torch.nn.Module:
	ckpt = _load_torch_checkpoint(stab_ckpt, device)
	state_dict = _normalize_state_dict_keys(_ckpt_get_state_dict(ckpt))
	model_meta = ckpt.get("model", None) if isinstance(ckpt, dict) else None

	if not isinstance(model_meta, dict):
		raise ValueError(
			"Stability-classifier checkpoint is missing 'model' metadata. Re-train to generate a metadata-rich checkpoint."
		)

	model = nets.StabilityClassifier(
		dim_theta=int(model_meta.get("dim_theta")),
		dim_u=int(model_meta.get("dim_u")),
		embed_dim=int(model_meta.get("embed_dim")),
		width=list(model_meta.get("width")),
		depth=list(model_meta.get("depth")),
	).to(device)

	model.load_state_dict(state_dict)
	model.eval()
	return model


def make_phi_function(
	phi_model: torch.nn.Module,
	*,
	device: torch.device | None = None,
) -> Callable[[np.ndarray], Callable[[np.ndarray], np.ndarray]]:
	"""Return a callable `phi(theta)` that returns a callable `phi_u(u)`.

	- theta: (dim_theta,) array
	- u: (dim_u,) or (N, dim_u) array
	- returns: scalar array of shape (N,) (or (1,) if single u)
	"""
	if device is None:
		device = next(phi_model.parameters()).device

	phi_model.eval()

	def phi(theta: np.ndarray):
		theta_np = np.asarray(theta, dtype=np.float32).reshape(1, -1)
		theta_t = torch.from_numpy(theta_np).to(device)

		def phi_u(u: np.ndarray) -> np.ndarray:
			u_np = np.asarray(u, dtype=np.float32)
			if u_np.ndim == 1:
				u_np = u_np.reshape(1, -1)
			u_t = torch.from_numpy(u_np).to(device)
			Theta_batch = theta_t.expand(u_t.shape[0], -1)
			with torch.no_grad():
				y = phi_model(u_t, Theta_batch).view(-1).detach().cpu().numpy()
			return y

		return phi_u

	return phi


def make_counting_function(
	count_model: torch.nn.Module,
	class_values: np.ndarray,
	*,
	device: torch.device | None = None,
) -> Callable[[np.ndarray], int]:
	"""Return a callable `count(theta)->int` decoding the predicted class via class_values."""
	if device is None:
		device = next(count_model.parameters()).device

	count_model.eval()
	class_values_np = np.asarray(class_values, dtype=np.int64)

	def predict_count(theta: np.ndarray) -> int:
		theta_np = np.asarray(theta, dtype=np.float32).reshape(1, -1)
		theta_t = torch.from_numpy(theta_np).to(device)
		with torch.no_grad():
			logits = count_model(theta_t).squeeze(0)
			label = int(torch.argmax(torch.softmax(logits, dim=0)).item())
		if 0 <= label < int(class_values_np.shape[0]):
			return int(class_values_np[label])
		return int(label)

	return predict_count


def make_stability_function(
	stability_model: torch.nn.Module,
	*,
	device: torch.device | None = None,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
	"""Return a callable `stability(theta, u)->prob`.

	- theta: (dim_theta,) array
	- u: (dim_u,) or (N, dim_u) array
	- returns: probabilities as shape (N,) array
	"""
	if device is None:
		device = next(stability_model.parameters()).device

	stability_model.eval()

	def stability(theta: np.ndarray, u: np.ndarray) -> np.ndarray:
		theta_np = np.asarray(theta, dtype=np.float32)
		u_np = np.asarray(u, dtype=np.float32)
		if u_np.ndim == 1:
			u_np = u_np.reshape(1, -1)
		if theta_np.ndim == 1:
			theta_np = theta_np.reshape(1, -1)
		# Broadcast theta to match u batch if needed.
		if theta_np.shape[0] == 1 and u_np.shape[0] > 1:
			theta_np = np.repeat(theta_np, u_np.shape[0], axis=0)

		u_t = torch.from_numpy(u_np).to(device)
		theta_t = torch.from_numpy(theta_np).to(device)
		with torch.no_grad():
			p = stability_model(u_t, theta_t).view(-1).detach().cpu().numpy()
		return p

	return stability


def load_inference_functions(
	*,
	phi_ckpt: str | Path | None = None,
	count_ckpt: str | Path | None = None,
	stability_ckpt: str | Path | None = None,
	device: torch.device | None = None,
) -> tuple[
	Callable[[np.ndarray], Callable[[np.ndarray], np.ndarray]] | None,
	Callable[[np.ndarray], int] | None,
	Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
]:
	"""Load checkpoints and return (phi_fn, count_fn, stability_fn)."""
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	phi_fn = None
	if phi_ckpt is not None:
		phi_model = load_phi_model(phi_ckpt, device)
		phi_fn = make_phi_function(phi_model, device=device)

	count_fn = None
	if count_ckpt is not None:
		count_model, class_values = load_count_classifier(count_ckpt, device)
		count_fn = make_counting_function(count_model, class_values, device=device)

	stability_fn = None
	if stability_ckpt is not None:
		stability_model = load_stability_classifier(stability_ckpt, device)
		stability_fn = make_stability_function(stability_model, device=device)

	return phi_fn, count_fn, stability_fn
