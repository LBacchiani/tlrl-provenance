#!/usr/bin/env python
"""Freeze the first DeepLTL FlatWorld PPO development-pilot manifest.

Mirrors scripts/freeze_letter_env_ppo_pilot_manifest.py exactly in structure
and discipline -- same benchmark-independent PPO algorithm, same DeepLTL
upstream, only the environment differs. Hyperparameters are DeepLTL's own
published defaults for this environment, taken directly from their
``run_flatworld.py`` convenience script (not LetterEnv's or PointLtl2's
settings, not this project's generic PPOConfig defaults -- all three would be
wrong for this environment).
"""

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
EVIDENCE = ROOT / "evidence" / "flatworld"
OUTPUT = EVIDENCE / "ppo" / "development" / "training" / "flatworld_ppo_5m_development_pilot_manifest.json"

EXPERIMENT_NAME = "v3_dev_ppo_5m_flatworld_seed_960000"
SEED = 960000
NUM_PROCS = 16
STEPS_PER_PROCESS = 4096
TARGET_STEPS = 5_000_000
CUDA_DEVICE_INDEX = "1"


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

    batch_steps = NUM_PROCS * STEPS_PER_PROCESS
    expected_updates = -(-TARGET_STEPS // batch_steps)
    expected_final_steps = expected_updates * batch_steps
    run_root = (
        UPSTREAM
        / "experiments"
        / "ppo"
        / "FlatWorld-v0"
        / EXPERIMENT_NAME
        / str(SEED)
    )

    command = [
        str(PYTHON),
        "src/train/train_ppo.py",
        "--env",
        "FlatWorld-v0",
        "--steps_per_process",
        str(STEPS_PER_PROCESS),
        "--epochs",
        "10",
        "--batch_size",
        "2048",
        "--discount",
        "0.98",
        "--gae_lambda",
        "0.95",
        "--entropy_coef",
        "0.003",
        "--log_interval",
        "1",
        "--save_interval",
        "2",
        "--num_steps",
        str(TARGET_STEPS),
        "--model_config",
        "FlatWorld-v0",
        "--curriculum",
        "FlatWorld-v0",
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
        "PYTHONIOENCODING": "utf-8",
        # Same two Windows-specific fixes documented in
        # benchmarks/letter_env/PPO_PROTOCOL.md ("Windows console encoding
        # and output buffering"): DeepLTL's text logger prints Greek mu in
        # several column labels and does not explicitly flush stdout.
        "PYTHONUNBUFFERED": "1",
        "RABINIZER_JAR": str(RABINIZER_JAR.resolve()),
        "CUDA_VISIBLE_DEVICES": CUDA_DEVICE_INDEX,
    }

    manifest = {
        "schema_version": "tlrl-flatworld-ppo-development-pilot/1",
        "status": "frozen_not_launched",
        "scope": "development_training_only_no_confirmatory_usefulness_evaluation",
        "created_local_time": datetime.now().astimezone().isoformat(),
        "benchmark_id": "deepltl-flatworld-official-eval-49",
        "hyperparameter_provenance": (
            "DeepLTL's own published convenience script for this environment "
            "(run_flatworld.py at the upstream repository root), not this "
            "project's generic PPOConfig defaults and not LetterEnv's or "
            "PointLtl2's hyperparameters -- steps_per_process/epochs/discount "
            "in particular differ substantially by environment upstream."
        ),
        "upstream": {
            "repository": "https://github.com/mathiasj33/deep-ltl",
            "commit": commit,
            "environment_id": "FlatWorld-v0",
        },
        "prelaunch_evidence": {
            "corpus_freeze": {
                "path": "evidence/flatworld/validation/flatworld_corpus_freeze.json",
                "sha256": evidence_sha("flatworld_corpus_freeze.json"),
            },
            "environment_smoke": {
                "path": "evidence/flatworld/validation/flatworld_environment_smoke.json",
                "sha256": evidence_sha("flatworld_environment_smoke.json"),
            },
            "official_ldba_environment_smoke": {
                "path": "evidence/flatworld/validation/flatworld_official_ldba_env_smoke.json",
                "sha256": evidence_sha("flatworld_official_ldba_env_smoke.json"),
            },
        },
        "prelaunch_evidence_note": (
            "No separate PPO-specific GPU/rabinizer smoke check exists for "
            "FlatWorld (unlike deepltl_point's ppo_gpu_smoke/"
            "rabinizer_verification): the same GPU stack, rabinizer jar, and "
            "upstream commit are independently exercised by this project's "
            "other DeepLTL benchmarks against the identical toolchain, which "
            "is corroborating rather than duplicative evidence of "
            "GPU+rabinizer compatibility."
        ),
        "training": {
            "algorithm": "formula_conditioned_recurrent_ppo",
            "experiment_name": EXPERIMENT_NAME,
            "seed": SEED,
            "environment_seeds": [SEED * 100 + index for index in range(NUM_PROCS)],
            "num_processes": NUM_PROCS,
            "target_steps": TARGET_STEPS,
            "target_steps_note": (
                "DeepLTL's own run_flatworld.py default is 15,000,000 steps. "
                "5,000,000 is a deliberate, documented reduced budget -- the "
                "same deviation already made, for the same reason, by this "
                "project's deepltl_point and letter_env PPO campaigns, and "
                "matched here to the FlatWorld value-based pillar's budget for "
                "comparability."
            ),
            "steps_per_process": STEPS_PER_PROCESS,
            "batch_steps": batch_steps,
            "expected_updates": expected_updates,
            "expected_final_steps_due_to_batch_boundary": expected_final_steps,
            "epochs": 10,
            "batch_size": 2048,
            "discount": 0.98,
            "gae_lambda": 0.95,
            "entropy_coefficient": 0.003,
            "save_interval_updates": 2,
            "log_interval_updates": 1,
            "device": "gpu",
            "cuda_device_index": CUDA_DEVICE_INDEX,
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
            "stdout_path": "v3.0/.runtime/deep-ltl/experiments/ppo/FlatWorld-v0/"
            + f"{EXPERIMENT_NAME}/{SEED}/launch.stdout.log",
            "stderr_path": "v3.0/.runtime/deep-ltl/experiments/ppo/FlatWorld-v0/"
            + f"{EXPERIMENT_NAME}/{SEED}/launch.stderr.log",
        },
        "command": command,
        "contamination_controls": {
            "deepltl_point_policy_dependency": False,
            "letter_env_policy_dependency": False,
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
