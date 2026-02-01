from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def load_yaml(path: str | Path) -> dict[str, Any]:
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
    return data if isinstance(data, dict) else {}


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
