from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

exp_dir = Path(__file__).resolve().parent
repo_root = exp_dir.parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from exps.feedback_loop._gen_data import build_run_config
from psnn.config import cfg_get, dump_yaml, load_yaml


def _default_run_dir(repo_root: Path, variant: str, seed: int) -> Path:
    return repo_root / "runs" / "feedback_loop" / variant / f"seed_{seed}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate feedback-loop clustering on the observation test set.")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Variant config under exps/feedback_loop.")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed for this run.")
    parser.add_argument("--output-dir", type=str, default=None, help="Base directory for the repo. Defaults to the parent of the exp dir.")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory. Defaults to runs/feedback_loop/<variant>/seed_<seed>.")
    parser.add_argument("--write-config", type=str, default=None, help="Optionally write the merged run config to a file.")
    parser.add_argument("--mode", type=str, default="strict", choices=["strict", "flexible"], help="Which evaluation runner to use.")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Extra args forwarded to the underlying evaluation script.")
    args = parser.parse_args()

    config_path = (exp_dir / args.config).resolve()
    raw_cfg = load_yaml(config_path)
    seed = int(args.seed if args.seed is not None else cfg_get(raw_cfg, "seed", 42))
    variant = str(cfg_get(raw_cfg, "run.variant", config_path.stem))
    base_root = Path(args.output_dir).resolve() if args.output_dir else repo_root
    run_dir = Path(args.run_dir).resolve() if args.run_dir else _default_run_dir(base_root, variant, seed)
    
    cfg = build_run_config(config_path, seed=seed, run_dir=run_dir)
    if args.write_config:
        dump_yaml(Path(args.write_config).resolve(), cfg)

    script_name = "_eval_strict.py" if args.mode == "strict" else "_eval_flexible.py"
    out_dir_name = "test_observation_eval_strict" if args.mode == "strict" else "test_observation_eval_flexible"
    cmd = [
        sys.executable,
        str(exp_dir / script_name),
        "--config",
        str(config_path),
        "--phi-ckpt",
        str(run_dir / "psnn_phi.pt"),
        "--obs-path",
        str(run_dir / "data" / "feedback_loop_obs_test.pkl"),
        "--out-root",
        str(run_dir / out_dir_name),
        "--random-state",
        str(seed),
    ]
    if args.mode == "strict":
        cmd.extend(["--count-ckpt", str(run_dir / "psnn_numsol.pt")])
    if args.extra_args:
        extras = args.extra_args[1:] if args.extra_args[0] == "--" else args.extra_args
        cmd.extend(extras)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
