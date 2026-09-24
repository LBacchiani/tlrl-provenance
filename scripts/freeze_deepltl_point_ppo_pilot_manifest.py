#!/usr/bin/env python
"""Freeze the first DeepLTL PointWorld PPO development-pilot manifest."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
import platform
import subprocess
import tempfile
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / ".runtime" / "deep-ltl"
PYTHON = Path(os.environ.get("TLRL_PYTHON", sys.executable))
RABINIZER_JAR = ROOT / "rabinizer-4" / "lib" / "rabinizer.jar"
EVIDENCE = ROOT / "evidence" / "deepltl_point"
OUTPUT = EVIDENCE / "development" / "training" / "deepltl_point_ppo_5m_development_pilot_manifest.json"

UPSTREAM_COMMIT = "3157200b5910cd7f5c493ffba8913994f2207eaf"
EXPERIMENT_NAME = "v3_dev_ppo_5m_seed_910001"
SEED = 910001
NUM_PROCS = 16
STEPS_PER_PROCESS = 4096
TARGET_STEPS = 5_000_000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
    temporary.replace(path)


def check_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")


def evidence_sha(name: str) -> str:
    path = EVIDENCE / "validation" / name
    check_file(path, name)
    check_file(path.with_suffix(path.suffix + ".sha256"), f"{name}.sha256")
    expected = path.with_suffix(path.suffix + ".sha256").read_text(encoding="ascii").split()[0]
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"hash mismatch for {name}: {actual} != {expected}")
    return actual


def main() -> None:
    check_file(PYTHON, "DeepLTL Python runtime")
    check_file(RABINIZER_JAR, "Rabinizer jar")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True
    ).strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"upstream commit mismatch: {commit}")

    batch_steps = NUM_PROCS * STEPS_PER_PROCESS
    expected_updates = -(-TARGET_STEPS // batch_steps)
    expected_final_steps = expected_updates * batch_steps
    run_root = (
        UPSTREAM
        / "experiments"
        / "ppo"
        / "PointLtl2-v0"
        / EXPERIMENT_NAME
        / str(SEED)
    )

    command = [
        str(PYTHON),
        "src/train/train_ppo.py",
        "--env",
        "PointLtl2-v0",
        "--steps_per_process",
        str(STEPS_PER_PROCESS),
        "--batch_size",
        "2048",
        "--lr",
        "0.0003",
        "--discount",
        "0.998",
        "--entropy_coef",
        "0.003",
        "--log_interval",
        "1",
        "--save_interval",
        "1",
        "--epochs",
        "10",
        "--num_steps",
        str(TARGET_STEPS),
        "--model_config",
        "PointLtl2-v0",
        "--curriculum",
        "PointLtl2-v0",
        "--name",
        EXPERIMENT_NAME,
        "--seed",
        str(SEED),
        "--device",
        "gpu",
        "--num_procs",
        str(NUM_PROCS),
        "--log_csv",
        "--save",
        "--no-log_wandb",
    ]
    env = {
        "PYTHONPATH": "src/",
        "PYTHONUTF8": "1",
        "RABINIZER_JAR": str(RABINIZER_JAR.resolve()),
    }

    manifest = {
        "schema_version": "tlrl-deepltl-point-ppo-development-pilot/1",
        "status": "frozen_not_launched",
        "scope": "development_training_only_no_confirmatory_usefulness_evaluation",
        "created_local_time": datetime.now().astimezone().isoformat(),
        "benchmark_id": "deepltl-pointworld-official-eval-50",
        "upstream": {
            "repository": "https://github.com/mathiasj33/deep-ltl",
            "commit": commit,
            "environment_id": "PointLtl2-v0",
        },
        "prelaunch_evidence": {
            "benchmark_manifest": {
                "path": "src/tlrl_benchmarks/deepltl_point/manifest.json",
                "sha256": sha256(ROOT / "src" / "tlrl_benchmarks" / "deepltl_point" / "manifest.json"),
            },
            "rabinizer_verification": {
                "path": "evidence/deepltl_point/validation/deepltl_point_rabinizer_verification.json",
                "sha256": evidence_sha("deepltl_point_rabinizer_verification.json"),
            },
            "official_ldba_environment_smoke": {
                "path": "evidence/deepltl_point/validation/deepltl_point_official_ldba_env_smoke.json",
                "sha256": evidence_sha("deepltl_point_official_ldba_env_smoke.json"),
            },
            "ppo_gpu_smoke": {
                "path": "evidence/deepltl_point/validation/deepltl_point_ppo_gpu_smoke.json",
                "sha256": evidence_sha("deepltl_point_ppo_gpu_smoke.json"),
            },
        },
        "training": {
            "algorithm": "formula_conditioned_recurrent_ppo",
            "experiment_name": EXPERIMENT_NAME,
            "seed": SEED,
            "environment_seeds": [SEED * 100 + index for index in range(NUM_PROCS)],
            "num_processes": NUM_PROCS,
            "target_steps": TARGET_STEPS,
            "steps_per_process": STEPS_PER_PROCESS,
            "batch_steps": batch_steps,
            "expected_updates": expected_updates,
            "expected_final_steps_due_to_batch_boundary": expected_final_steps,
            "epochs": 10,
            "batch_size": 2048,
            "discount": 0.998,
            "learning_rate": 0.0003,
            "entropy_coefficient": 0.003,
            "save_interval_updates": 1,
            "log_interval_updates": 1,
            "device": "gpu",
            "log_wandb": False,
            "log_csv": True,
            "save": True,
        },
        "runtime": {
            "python": platform.python_version(),
            "python_executable": str(PYTHON.resolve()),
            "rabinizer_jar": str(RABINIZER_JAR.resolve()),
            "rabinizer_jar_sha256": sha256(RABINIZER_JAR),
            "required_environment": env,
            "working_directory": str(UPSTREAM.resolve()),
            "stdout_path": "v3.0/.runtime/deep-ltl/experiments/ppo/PointLtl2-v0/"
            + f"{EXPERIMENT_NAME}/{SEED}/launch.stdout.log",
            "stderr_path": "v3.0/.runtime/deep-ltl/experiments/ppo/PointLtl2-v0/"
            + f"{EXPERIMENT_NAME}/{SEED}/launch.stderr.log",
        },
        "command": command,
        "contamination_controls": {
            "v2_dependency": False,
            "uses_reserved_development_seed": True,
            "utility_outcomes_inspected_before_launch": False,
            "confirmatory_training_authorized": False,
            "confirmatory_usefulness_evaluation_authorized": False,
            "run_directory_must_not_exist_before_launch": str(run_root.resolve()),
        },
    }
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write(OUTPUT, payload)
    atomic_write(
        OUTPUT.with_suffix(OUTPUT.suffix + ".sha256"),
        f"{hashlib.sha256(payload).hexdigest()}  {OUTPUT.name}\n".encode("ascii"),
    )
    print(json.dumps({"status": "frozen", "path": str(OUTPUT), "sha256": hashlib.sha256(payload).hexdigest()}))


if __name__ == "__main__":
    main()
