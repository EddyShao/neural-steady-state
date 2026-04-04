from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import tqdm


def lex_sort_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={arr.shape}")
    if arr.shape[0] == 0:
        return arr.reshape(0, arr.shape[1])
    keys = [arr[:, j] for j in reversed(range(arr.shape[1]))]
    return arr[np.lexsort(keys)]


def load_true_solutions(entry: dict) -> np.ndarray:
    sols = [np.asarray(sol["u"], dtype=np.float32) for sol in entry.get("U", [])]
    if not sols:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(sols, dtype=np.float32).reshape(-1, 2)


def maybe_sample_observations(
    obs: list[dict],
    *,
    limit: int | None = None,
    sample_size: int | None = None,
    sample_seed: int = 0,
) -> list[dict]:
    if limit is not None and sample_size is not None:
        raise ValueError("Use at most one of --limit or --sample-size")
    if limit is not None:
        return obs[: max(0, int(limit))]
    if sample_size is None:
        return obs

    n = len(obs)
    k = max(0, min(int(sample_size), n))
    rng = np.random.default_rng(int(sample_seed))
    indices = np.sort(rng.choice(n, size=k, replace=False))
    return [obs[int(i)] for i in indices.tolist()]


def load_observations(
    obs_path: str,
    *,
    limit: int | None = None,
    sample_size: int | None = None,
    sample_seed: int = 0,
) -> list[dict]:
    obs = joblib.load(obs_path)
    return maybe_sample_observations(
        obs,
        limit=limit,
        sample_size=sample_size,
        sample_seed=sample_seed,
    )


def aggregate_evaluation_results(
    results: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    count_source: str,
) -> tuple[dict, dict]:
    pred_counts: list[int] = []
    true_counts: list[int] = []
    thetas: list[np.ndarray] = []
    pred_solution_lists: list[np.ndarray] = []
    true_solution_lists: list[np.ndarray] = []
    per_theta_l2: list[float] = []
    correctly_counted_mask: list[bool] = []

    for theta, true_solutions, pred_solutions in results:
        true_solutions = lex_sort_rows(np.asarray(true_solutions, dtype=np.float32).reshape(-1, 2))
        pred_solutions = lex_sort_rows(np.asarray(pred_solutions, dtype=np.float32).reshape(-1, 2))
        true_count = int(true_solutions.shape[0])
        pred_count = int(pred_solutions.shape[0])
        is_correct = pred_count == true_count

        pred_counts.append(pred_count)
        true_counts.append(true_count)
        thetas.append(np.asarray(theta, dtype=np.float32).reshape(-1))
        pred_solution_lists.append(pred_solutions)
        true_solution_lists.append(true_solutions)
        correctly_counted_mask.append(is_correct)

        if is_correct and true_count > 0:
            diffs = np.linalg.norm(pred_solutions - true_solutions, axis=1)
            per_theta_l2.append(float(np.mean(diffs)))

    pred_counts_np = np.asarray(pred_counts, dtype=np.int64)
    true_counts_np = np.asarray(true_counts, dtype=np.int64)
    correctly_counted_np = np.asarray(correctly_counted_mask, dtype=bool)
    thetas_np = np.asarray(thetas, dtype=np.float32)

    metrics = {
        "num_samples": int(pred_counts_np.shape[0]),
        "num_correctly_counted": int(np.sum(correctly_counted_np)),
        "count_accuracy": float(np.mean(pred_counts_np == true_counts_np)) if pred_counts_np.size > 0 else float("nan"),
        "mean_l2_correctly_counted_theta": float(np.mean(np.asarray(per_theta_l2, dtype=np.float32))) if per_theta_l2 else float("nan"),
        "count_source": str(count_source),
        "l2_definition": "For each correctly counted theta, lex-sort predicted and true solutions, compute row-wise L2 norms, average within theta, then average across those theta.",
    }

    details = {
        "thetas": thetas_np,
        "pred_counts": pred_counts_np,
        "true_counts": true_counts_np,
        "correctly_counted": correctly_counted_np,
        "per_theta_l2_correct": np.asarray(per_theta_l2, dtype=np.float32),
        "pred_solutions": np.asarray(pred_solution_lists, dtype=object),
        "true_solutions": np.asarray(true_solution_lists, dtype=object),
    }
    return metrics, details


def evaluate_observations(
    obs: list[dict],
    *,
    predict_fn,
    count_source: str,
) -> tuple[dict, dict]:
    results: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for entry in tqdm.tqdm(obs, desc="Evaluating test observations"):
        theta = np.asarray(entry["Theta"], dtype=np.float32).reshape(-1)
        true_solutions = lex_sort_rows(load_true_solutions(entry))
        pred_solutions = np.asarray(predict_fn(theta), dtype=np.float32).reshape(-1, 2)
        results.append((theta, true_solutions, pred_solutions))
    return aggregate_evaluation_results(results, count_source=count_source)


def save_outputs(
    *,
    out_root: str,
    stem: str,
    metrics: dict,
    details: dict,
) -> tuple[Path, Path]:
    out_dir = Path(out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}_metrics.json"
    npz_path = out_dir / f"{stem}_details.npz"

    with json_path.open("w", encoding="ascii") as f:
        json.dump(metrics, f, indent=2)

    np.savez(
        npz_path,
        **details,
        metadata_json=np.asarray(json.dumps(metrics), dtype=np.str_),
    )
    return json_path, npz_path
