from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .training import RunConfig


def _deep_update(target: dict[str, Any], update: Mapping[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def run_ablation_matrix(
    base_config: str | Path,
    matrix_path: str | Path,
    *,
    nproc_per_node: int,
    shard_index: int = 0,
    num_shards: int = 1,
    selected_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must lie in [0, num_shards)")
    base_config = Path(base_config).resolve()
    matrix_path = Path(matrix_path).resolve()
    # Resolve all data/output paths against the original base configuration
    # before writing derived configurations into another directory.
    base = RunConfig.from_json(base_config).to_dict()
    with matrix_path.open("r", encoding="utf-8") as handle:
        matrix = json.load(handle)
    experiments = matrix.get("experiments")
    if not isinstance(experiments, list):
        raise ValueError("ablation matrix must contain an experiments list")
    root = Path(base["runtime"]["output_dir"])
    if not root.is_absolute():
        root = (base_config.parent / root).resolve()
    ablation_root = root.parent / f"{root.name}_ablations"
    config_root = ablation_root / "resolved_configs"
    config_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for index, experiment in enumerate(experiments):
        if index % num_shards != shard_index:
            continue
        name = str(experiment["name"])
        if selected_names is not None and name not in selected_names:
            continue
        resolved = copy.deepcopy(base)
        _deep_update(resolved, experiment.get("overrides", {}))
        run_dir = ablation_root / name
        resolved.setdefault("runtime", {})["output_dir"] = str(run_dir)
        config_file = config_root / f"{name}.json"
        with config_file.open("w", encoding="utf-8") as handle:
            json.dump(resolved, handle, indent=2, sort_keys=True)
        if nproc_per_node == 1:
            command = [
                sys.executable,
                "-m",
                "unipansurv.training",
                "--config",
                str(config_file),
            ]
        else:
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={nproc_per_node}",
                "-m",
                "unipansurv.training",
                "--config",
                str(config_file),
            ]
        print(f"[ablation:{name}] {' '.join(command)}", flush=True)
        completed = subprocess.run(command, check=False)
        result: dict[str, Any] = {
            "name": name,
            "return_code": completed.returncode,
            "output_dir": str(run_dir),
        }
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as handle:
                result["summary"] = json.load(handle)
        results.append(result)
        with (ablation_root / f"shard_{shard_index}_manifest.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(results, handle, indent=2, sort_keys=True)
        if completed.returncode != 0:
            raise RuntimeError(f"ablation {name!r} failed with code {completed.returncode}")
    return results


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the prespecified UniPanSurv ablations")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--nproc-per-node", type=int, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args(argv)
    run_ablation_matrix(
        args.base_config,
        args.matrix,
        nproc_per_node=args.nproc_per_node,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        selected_names=None if args.only is None else set(args.only),
    )


if __name__ == "__main__":
    main()
