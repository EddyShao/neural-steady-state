import argparse
import sys
from pathlib import Path

exp_dir = Path(__file__).resolve().parent
repo_root = exp_dir.parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(exp_dir) not in sys.path:
    sys.path.insert(0, str(exp_dir))

from _gen_data import build_run_config, generate_from_loaded_config
from _train_models import run_from_loaded_config
from psnn.config import cfg_get, dump_yaml, load_yaml, resolve_path


def _default_run_dir(repo_root: Path, variant: str, seed: int) -> Path:
    return repo_root / "runs" / "gray-scott-toy" / variant / f"seed_{seed}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate data and train the Gray-Scott toy Phi model.")
    parser.add_argument("--config", type=str, default="configs/complete.yaml", help="Variant config under exps/gray-scott-toy.")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed for this run.")
    parser.add_argument("--output-dir", type=str, default=None, help="Base directory for the repo. Defaults to the parent of the exp dir.")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory. Defaults to runs/gray-scott-toy/<variant>/seed_<seed>.")
    parser.add_argument("--write-config", type=str, default=None, help="Optionally write the merged run config to a file.")
    parser.add_argument("--skip-data", action="store_true", help="Skip data generation and train from existing data in the run dir.")
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

    data_outputs = cfg_get(cfg, "data_generation.outputs", {})
    data_dir = resolve_path(run_dir, cfg_get(cfg, "training.paths.data_dir", cfg_get(data_outputs, "out_dir", "data")))
    required_data_files = [
        data_dir / cfg_get(data_outputs, "data_train_npz", "gray_scott_toy_data_train.npz"),
        data_dir / cfg_get(data_outputs, "data_test_npz", "gray_scott_toy_data_test.npz"),
    ]
    if not args.skip_data:
        if all(path.exists() for path in required_data_files):
            print(f"Reusing shared dataset from: {data_dir}")
        else:
            generate_from_loaded_config(cfg, base_dir=run_dir)
    run_from_loaded_config(cfg, base_dir=run_dir)


if __name__ == "__main__":
    main()
