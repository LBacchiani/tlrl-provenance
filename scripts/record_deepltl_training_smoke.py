#!/usr/bin/env python
"""Validate and hash the completed DeepLTL CUDA/PPO runtime smoke."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "3157200b5910cd7f5c493ffba8913994f2207eaf"
EXPECTED_NAME = "v3_runtime_smoke_utf8"
EXPECTED_SEED = 900003
EXPECTED_STEPS = 2048


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evidence" / "deepltl_point" / "validation" / "deepltl_point_ppo_gpu_smoke.json",
    )
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=upstream, text=True
    ).strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"upstream commit mismatch: {commit}")

    run_root = (
        upstream
        / "experiments"
        / "ppo"
        / "PointLtl2-v0"
        / EXPECTED_NAME
    )
    seed_root = run_root / str(EXPECTED_SEED)
    required = {
        "experiment_config": run_root / "experiment_config.json",
        "log": seed_root / "log.csv",
        "status": seed_root / "status.pth",
        "ltl_net": seed_root / "ltl_net.pth",
        "vocab": seed_root / "vocab.pkl",
        "initial_eval_checkpoint": seed_root / "eval" / "0.pth",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"training smoke artifacts are missing: {missing}")

    config = json.loads(required["experiment_config"].read_text(encoding="utf-8"))
    expected_ppo = {
        "epochs": 1,
        "batch_size": 512,
        "steps_per_process": 1024,
        "discount": 0.998,
        "lr": 0.0003,
        "entropy_coef": 0.003,
    }
    for key, value in expected_ppo.items():
        if config["ppo"].get(key) != value:
            raise RuntimeError(f"unexpected smoke PPO setting {key}: {config['ppo'].get(key)}")
    if config.get("curriculum") != "PointLtl2-v0" or config.get("save") is not True:
        raise RuntimeError("smoke configuration did not use the official curriculum and saving")

    status = torch.load(required["status"], map_location="cpu")
    expected_status = {
        "num_steps": EXPECTED_STEPS,
        "num_updates": 1,
        "curriculum_stage": 0,
        "num_eval_steps": EXPECTED_STEPS,
    }
    for key, value in expected_status.items():
        if status.get(key) != value:
            raise RuntimeError(f"unexpected checkpoint {key}: {status.get(key)}")
    if not status.get("model_state") or not status.get("optimizer_state"):
        raise RuntimeError("final checkpoint lacks model or optimizer state")

    with required["log"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1 or int(rows[0]["num_steps"]) != EXPECTED_STEPS:
        raise RuntimeError("CSV log does not contain exactly the completed smoke update")

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is unavailable while recording the CUDA smoke")
    artifact_records = {
        name: {
            "relative_path": path.relative_to(upstream).as_posix(),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in sorted(required.items())
    }
    source_files = {
        "rabinizer_bridge": upstream / "src" / "ltl" / "automata" / "rabinizer.py",
        "cross_platform_file_logger": upstream
        / "src"
        / "utils"
        / "logging"
        / "file_logger.py",
    }
    result = {
        "schema_version": "tlrl-deepltl-point-ppo-gpu-smoke/1",
        "status": "passed",
        "scope": "development_only_runtime_smoke_no_utility_evaluation",
        "upstream_commit": commit,
        "environment_id": "PointLtl2-v0",
        "algorithm": "formula_conditioned_recurrent_ppo",
        "experiment_name": EXPECTED_NAME,
        "seed": EXPECTED_SEED,
        "environment_seeds": [EXPECTED_SEED * 100, EXPECTED_SEED * 100 + 1],
        "num_processes": 2,
        "num_steps": EXPECTED_STEPS,
        "num_updates": 1,
        "checkpoint_complete": True,
        "csv_log_rows": len(rows),
        "artifacts": artifact_records,
        "patched_runtime_sources": {
            name: sha256(path) for name, path in sorted(source_files.items())
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "python_utf8_mode_required": True,
        },
        "methodology_outcomes_inspected": False,
    }
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write(args.output, payload)
    atomic_write(
        args.output.with_suffix(args.output.suffix + ".sha256"),
        f"{hashlib.sha256(payload).hexdigest()}  {args.output.name}\n".encode("ascii"),
    )
    print(json.dumps({key: value for key, value in result.items() if key != "artifacts"}, sort_keys=True))


if __name__ == "__main__":
    main()
