from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two config dicts without mutating inputs."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: str | Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Load a YAML file into a dict.

    Requires PyYAML. We keep this tiny to avoid hard-coding configs in code.
    """
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency check
        raise RuntimeError(
            "PyYAML is required to load config files. Install with `pip install pyyaml`."
        ) from exc

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        return {}

    seen = set() if _seen is None else set(_seen)
    resolved_path = path.resolve()
    if resolved_path in seen:
        raise ValueError(f"Config inheritance cycle detected at: {resolved_path}")
    seen.add(resolved_path)

    parent_ref = data.pop("inherits", None)
    if parent_ref is None:
        return data

    parent_path = resolve_path(path.parent, parent_ref)
    parent_cfg = load_yaml(parent_path, _seen=seen)
    return deep_merge_dicts(parent_cfg, data)


def dump_yaml(path: str | Path, data: dict[str, Any]) -> None:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency check
        raise RuntimeError(
            "PyYAML is required to write config files. Install with `pip install pyyaml`."
        ) from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def cfg_get(cfg: dict[str, Any], path: str | Iterable[str], default: Any = None) -> Any:
    """Safe nested lookup for configs.

    `path` can be "a.b.c" or an iterable of keys.
    """
    if isinstance(path, str):
        keys = path.split(".") if path else []
    else:
        keys = list(path)
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def resolve_path(base_dir: str | Path, path: str | Path) -> Path:
    base = Path(base_dir)
    path = Path(path)
    return path if path.is_absolute() else base / path
