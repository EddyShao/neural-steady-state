from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

exp_dir = Path(__file__).resolve().parent
repo_root = exp_dir.parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from exps.gray_scott._gen_data import build_run_config
from psnn.config import cfg_get, dump_yaml, load_yaml


def _default_run_dir(repo_root: Path, variant: str, seed: int) -> Path:
    return repo_root / "runs" / "gray_scott" / variant / f"seed_{seed}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gray-Scott bifurcation plotting for a config/seed pair.")
    parser.add_argument("--config", type=str, default="configs/complete.yaml", help="Variant config under exps/gray_scott.")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed for this run.")
    parser.add_argument("--output-dir", type=str, default=None, help="Base directory for the repo. Defaults to the parent of the exp dir.")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory. Defaults to runs/gray_scott/<variant>/seed_<seed>.")
    parser.add_argument("--write-config", type=str, default=None, help="Optionally write the merged run config to a file.")
    parser.add_argument("--mode", type=str, default="strict", choices=["strict", "flexible"], help="Which bifurcation runner to use.")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Extra args forwarded to the underlying bifurcation script.")
    args = parser.parse_args()

    config_path = (exp_dir / args.config).resolve()
    raw_cfg = load_yaml(config_path)
    seed = int(args.seed if args.seed is not None else cfg_get(raw_cfg, "seed", 123))
    variant = str(cfg_get(raw_cfg, "run.variant", config_path.stem))
    base_root = Path(args.output_dir).resolve() if args.output_dir else repo_root
    run_dir = Path(args.run_dir).resolve() if args.run_dir else _default_run_dir(base_root, variant, seed)

    cfg = build_run_config(config_path, seed=seed, run_dir=run_dir)
    if args.write_config:
        dump_yaml(Path(args.write_config).resolve(), cfg)

    script_name = "_bifur_strict.py" if args.mode == "strict" else "_bifur_flexible.py"
    cmd = [
        sys.executable,
        str(exp_dir / script_name),
        "--config",
        str(config_path),
        "--phi-ckpt",
        str(run_dir / "psnn_phi.pt"),
        "--stability-ckpt",
        str(run_dir / "psnn_stability_cls.pt"),
        "--out-root",
        str(run_dir / ("bifur_strict_runs" if args.mode == "strict" else "bifur_flexible_runs")),
    ]
    if args.mode == "strict":
        cmd.extend(["--count-ckpt", str(run_dir / "psnn_numsol.pt"), "--random-state", str(seed)])
    else:
        cmd.extend(["--random-state", str(seed)])
    if args.extra_args:
        extras = args.extra_args[1:] if args.extra_args[0] == "--" else args.extra_args
        cmd.extend(extras)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
