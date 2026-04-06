from __future__ import annotations

import itertools
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


def brute_force_hungarian_match(
    true_solutions: np.ndarray,
    pred_solutions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    true_solutions = np.asarray(true_solutions, dtype=np.float32).reshape(-1, 2)
    pred_solutions = np.asarray(pred_solutions, dtype=np.float32).reshape(-1, 2)
    if true_solutions.shape != pred_solutions.shape:
        raise ValueError(
            f"Expected matched shapes for assignment, got true={true_solutions.shape}, pred={pred_solutions.shape}"
        )
    n = true_solutions.shape[0]
    if n <= 1:
        indices = np.arange(n, dtype=np.int64)
        return true_solutions, pred_solutions, indices, indices

    true_order = np.lexsort([true_solutions[:, j] for j in reversed(range(true_solutions.shape[1]))])
    pred_order = np.lexsort([pred_solutions[:, j] for j in reversed(range(pred_solutions.shape[1]))])
    true_sorted = true_solutions[true_order]
    pred_sorted = pred_solutions[pred_order]

    # The number of steady states per parameter is small, so we can just use a
    # brute-force Hungarian match by checking every assignment and taking the minimum-cost one.
    best_perm: tuple[int, ...] | None = None
    best_cost: float | None = None
    for perm in itertools.permutations(range(n)):
        diffs = pred_sorted[list(perm)] - true_sorted
        cost = float(np.sum(np.linalg.norm(diffs, axis=1)))
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_perm = perm

    if best_perm is None:
        best_perm = tuple(range(n))
    pred_matched_order = pred_order[list(best_perm)]
    return true_sorted, pred_solutions[pred_matched_order], true_order.astype(np.int64), np.asarray(pred_matched_order, dtype=np.int64)


def load_true_solutions(entry: dict) -> np.ndarray:
    sols = [np.asarray(sol["u"], dtype=np.float32) for sol in entry.get("U", [])]
    if not sols:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(sols, dtype=np.float32).reshape(-1, 2)


def load_true_stability(entry: dict) -> np.ndarray:
    stable = [bool(sol.get("stable", False)) for sol in entry.get("U", [])]
    if not stable:
        return np.empty((0,), dtype=bool)
    return np.asarray(stable, dtype=bool)


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
    results: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]],
    *,
    count_source: str,
) -> tuple[dict, dict]:
    pred_counts: list[int] = []
    true_counts: list[int] = []
    thetas: list[np.ndarray] = []
    pred_solution_lists: list[np.ndarray] = []
    true_solution_lists: list[np.ndarray] = []
    per_observation_l2: list[float] = []
    per_parameter_l2_values: dict[bytes, list[float]] = {}
    per_parameter_thetas: dict[bytes, np.ndarray] = {}
    per_observation_stability_accuracy: list[float] = []
    per_parameter_stability_accuracy_values: dict[bytes, list[float]] = {}
    true_stability_lists: list[np.ndarray] = []
    pred_stability_lists: list[np.ndarray] = []
    correctly_counted_mask: list[bool] = []

    for theta, true_solutions, pred_solutions, true_stability, pred_stability in results:
        theta_arr = np.asarray(theta, dtype=np.float32).reshape(-1)
        true_solutions = np.asarray(true_solutions, dtype=np.float32).reshape(-1, 2)
        pred_solutions = np.asarray(pred_solutions, dtype=np.float32).reshape(-1, 2)
        true_stability_arr = None if true_stability is None else np.asarray(true_stability, dtype=bool).reshape(-1)
        pred_stability_arr = None if pred_stability is None else np.asarray(pred_stability, dtype=bool).reshape(-1)
        true_count = int(true_solutions.shape[0])
        pred_count = int(pred_solutions.shape[0])
        is_correct = pred_count == true_count

        pred_counts.append(pred_count)
        true_counts.append(true_count)
        thetas.append(theta_arr)
        correctly_counted_mask.append(is_correct)

        if is_correct and true_count > 0:
            true_matched, pred_matched, true_match_indices, pred_match_indices = brute_force_hungarian_match(
                true_solutions,
                pred_solutions,
            )
            diffs = np.linalg.norm(pred_matched - true_matched, axis=1)
            mean_l2 = float(np.mean(diffs))
            per_observation_l2.append(mean_l2)
            theta_key = theta_arr.tobytes()
            per_parameter_thetas.setdefault(theta_key, theta_arr.copy())
            per_parameter_l2_values.setdefault(theta_key, []).append(mean_l2)
            true_solution_lists.append(true_matched)
            pred_solution_lists.append(pred_matched)
            if true_stability_arr is not None and pred_stability_arr is not None:
                true_stability_matched = true_stability_arr[true_match_indices]
                pred_stability_matched = pred_stability_arr[pred_match_indices]
                stability_accuracy = float(np.mean(pred_stability_matched == true_stability_matched))
                per_observation_stability_accuracy.append(stability_accuracy)
                per_parameter_stability_accuracy_values.setdefault(theta_key, []).append(stability_accuracy)
                true_stability_lists.append(true_stability_matched)
                pred_stability_lists.append(pred_stability_matched)
            else:
                true_stability_lists.append(np.empty((0,), dtype=bool))
                pred_stability_lists.append(np.empty((0,), dtype=bool))
        else:
            true_solution_lists.append(lex_sort_rows(true_solutions))
            pred_solution_lists.append(lex_sort_rows(pred_solutions))
            true_stability_lists.append(np.asarray(true_stability_arr if true_stability_arr is not None else [], dtype=bool))
            pred_stability_lists.append(np.asarray(pred_stability_arr if pred_stability_arr is not None else [], dtype=bool))

    pred_counts_np = np.asarray(pred_counts, dtype=np.int64)
    true_counts_np = np.asarray(true_counts, dtype=np.int64)
    correctly_counted_np = np.asarray(correctly_counted_mask, dtype=bool)
    thetas_np = np.asarray(thetas, dtype=np.float32)
    parameter_theta_list = [per_parameter_thetas[k] for k in per_parameter_thetas]
    per_parameter_l2 = [
        float(np.mean(np.asarray(per_parameter_l2_values[k], dtype=np.float32)))
        for k in per_parameter_thetas
    ]
    per_parameter_stability_accuracy = [
        float(np.mean(np.asarray(per_parameter_stability_accuracy_values[k], dtype=np.float32)))
        for k in per_parameter_thetas
        if k in per_parameter_stability_accuracy_values
    ]

    metrics = {
        "num_samples": int(pred_counts_np.shape[0]),
        "num_correctly_counted": int(np.sum(correctly_counted_np)),
        "count_accuracy": float(np.mean(pred_counts_np == true_counts_np)) if pred_counts_np.size > 0 else float("nan"),
        "mean_l2_correctly_counted_theta": float(np.mean(np.asarray(per_parameter_l2, dtype=np.float32))) if per_parameter_l2 else float("nan"),
        "stability_accuracy_correctly_counted_theta": (
            float(np.mean(np.asarray(per_parameter_stability_accuracy, dtype=np.float32)))
            if per_parameter_stability_accuracy
            else float("nan")
        ),
        "count_source": str(count_source),
        "l2_definition": "For each correctly counted parameter with at least one solution, match predicted and true solutions using a brute-force minimum-cost assignment, compute row-wise L2 norms, average within each observation, average those values within the parameter, then average across parameters.",
    }

    details = {
        "thetas": thetas_np,
        "pred_counts": pred_counts_np,
        "true_counts": true_counts_np,
        "correctly_counted": correctly_counted_np,
        "per_theta_l2_correct": np.asarray(per_observation_l2, dtype=np.float32),
        "per_theta_stability_accuracy_correct": np.asarray(per_observation_stability_accuracy, dtype=np.float32),
        "correct_parameter_thetas": np.asarray(parameter_theta_list, dtype=np.float32),
        "per_parameter_l2_correct": np.asarray(per_parameter_l2, dtype=np.float32),
        "per_parameter_stability_accuracy_correct": np.asarray(per_parameter_stability_accuracy, dtype=np.float32),
        "pred_solutions": np.asarray(pred_solution_lists, dtype=object),
        "true_solutions": np.asarray(true_solution_lists, dtype=object),
        "pred_stability": np.asarray(pred_stability_lists, dtype=object),
        "true_stability": np.asarray(true_stability_lists, dtype=object),
    }
    return metrics, details


def evaluate_observations(
    obs: list[dict],
    *,
    predict_fn,
    predict_stability_fn=None,
    count_source: str,
) -> tuple[dict, dict]:
    results: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]] = []
    for entry in tqdm.tqdm(obs, desc="Evaluating test observations"):
        theta = np.asarray(entry["Theta"], dtype=np.float32).reshape(-1)
        true_solutions = load_true_solutions(entry)
        true_stability = load_true_stability(entry)
        pred_solutions = np.asarray(predict_fn(theta), dtype=np.float32).reshape(-1, 2)
        pred_stability = None
        if predict_stability_fn is not None and pred_solutions.shape[0] > 0:
            pred_stability = np.asarray(predict_stability_fn(theta, pred_solutions), dtype=bool).reshape(-1)
        results.append((theta, true_solutions, pred_solutions, true_stability, pred_stability))
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


def attach_missing_metrics(
    metrics: dict,
    details: dict,
    *,
    metrics_missing: dict | None,
) -> tuple[dict, dict]:
    if metrics_missing is None:
        metrics["count_accuracy_missing"] = "N/A"
        metrics["mean_l2_correctly_counted_theta_missing"] = "N/A"
        metrics["stability_accuracy_correctly_counted_theta_missing"] = "N/A"
        details["count_accuracy_missing"] = np.asarray("N/A", dtype=np.str_)
        details["mean_l2_correctly_counted_theta_missing"] = np.asarray("N/A", dtype=np.str_)
        details["stability_accuracy_correctly_counted_theta_missing"] = np.asarray("N/A", dtype=np.str_)
        return metrics, details

    metrics["count_accuracy_missing"] = float(metrics_missing["count_accuracy"])
    metrics["mean_l2_correctly_counted_theta_missing"] = float(metrics_missing["mean_l2_correctly_counted_theta"])
    metrics["stability_accuracy_correctly_counted_theta_missing"] = float(
        metrics_missing["stability_accuracy_correctly_counted_theta"]
    )
    details["count_accuracy_missing"] = np.asarray(metrics_missing["count_accuracy"], dtype=np.float32)
    details["mean_l2_correctly_counted_theta_missing"] = np.asarray(
        metrics_missing["mean_l2_correctly_counted_theta"],
        dtype=np.float32,
    )
    details["stability_accuracy_correctly_counted_theta_missing"] = np.asarray(
        metrics_missing["stability_accuracy_correctly_counted_theta"],
        dtype=np.float32,
    )
    return metrics, details
