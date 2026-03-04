
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn


def unwrap_model(model: nn.Module) -> nn.Module:
	"""Return the underlying nn.Module for common wrappers.

	Handles:
	- torch.compile() / OptimizedModule via `_orig_mod`
	- DistributedDataParallel via `.module`
	"""
	if hasattr(model, "_orig_mod"):
		try:
			return getattr(model, "_orig_mod")
		except Exception:
			pass
	if hasattr(model, "module"):
		try:
			return getattr(model, "module")
		except Exception:
			pass
	return model


def _linear_layers(module: nn.Module) -> List[nn.Linear]:
	return [m for m in module.modules() if isinstance(m, nn.Linear)]


def infer_mlp_arch(mlp_or_seq: nn.Module) -> Dict[str, int]:
	"""Infer (in_dim, width, depth, out_dim) from an MLP-like module.

	Assumes the module contains a stack of Linear layers.
	"""
	module = mlp_or_seq
	if hasattr(module, "net") and isinstance(getattr(module, "net"), nn.Module):
		module = getattr(module, "net")
	linears = _linear_layers(module)
	if not linears:
		raise ValueError("No nn.Linear layers found to infer MLP architecture")
	first = linears[0]
	last = linears[-1]
	in_dim = int(first.in_features)
	width = int(first.out_features) if len(linears) > 1 else int(last.out_features)
	out_dim = int(last.out_features)
	# For our MLP definition, number of Linear layers = depth + 1
	depth = max(0, len(linears) - 1)
	return {"in_dim": in_dim, "width": width, "depth": depth, "out_dim": out_dim}


def infer_psnn_arch(model: nn.Module) -> Dict[str, Any]:
	"""Infer PSNN hyperparams directly from the instantiated model."""
	model_u = unwrap_model(model)
	if not (hasattr(model_u, "pnn") and hasattr(model_u, "snn")):
		raise ValueError("Model does not look like a PSNN (missing pnn/snn)")
	p = infer_mlp_arch(getattr(model_u, "pnn"))
	s = infer_mlp_arch(getattr(model_u, "snn"))
	if p["out_dim"] != s["out_dim"]:
		raise ValueError(f"PSNN embed_dim mismatch: pnn={p['out_dim']} snn={s['out_dim']}")
	eta = getattr(model_u, "eta", None)
	eta_f = float(eta) if eta is not None else None
	return {
		"name": "PSNN",
		"dim_theta": int(p["in_dim"]),
		"dim_u": int(s["in_dim"]),
		"embed_dim": int(p["out_dim"]),
		"width": [int(p["width"]), int(s["width"])],
		"depth": [int(p["depth"]), int(s["depth"])],
		"eta": eta_f,
	}


def infer_theta_count_arch(model: nn.Module) -> Dict[str, Any]:
	model_u = unwrap_model(model)
	if not hasattr(model_u, "net"):
		raise ValueError("Model does not look like ThetaCountClassifier (missing net)")
	arch = infer_mlp_arch(getattr(model_u, "net"))
	return {
		"name": "ThetaCountClassifier",
		"dim_theta": int(arch["in_dim"]),
		"num_classes": int(arch["out_dim"]),
		"width": int(arch["width"]),
		"depth": int(arch["depth"]),
	}


def infer_stability_arch(model: nn.Module) -> Dict[str, Any]:
	model_u = unwrap_model(model)
	if not (hasattr(model_u, "theta_net") and hasattr(model_u, "u_net")):
		raise ValueError("Model does not look like StabilityClassifier (missing theta_net/u_net)")
	t = infer_mlp_arch(getattr(model_u, "theta_net"))
	u = infer_mlp_arch(getattr(model_u, "u_net"))
	if t["out_dim"] != u["out_dim"]:
		raise ValueError(f"Stability embed_dim mismatch: theta_net={t['out_dim']} u_net={u['out_dim']}")
	return {
		"name": "StabilityClassifier",
		"dim_theta": int(t["in_dim"]),
		"dim_u": int(u["in_dim"]),
		"embed_dim": int(t["out_dim"]),
		"width": [int(t["width"]), int(u["width"])],
		"depth": [int(t["depth"]), int(u["depth"])],
	}

