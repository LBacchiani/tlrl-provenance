#!/usr/bin/env python
"""Launch the frozen DeepLTL FlatWorld PPO development pilot.

Mirrors scripts/launch_letter_env_ppo_pilot.py exactly.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "flatworld" / "ppo" / "development" / "training"
MANIFEST = EVIDENCE / "flatworld_ppo_5m_development_pilot_manifest.json"
LAUNCH_RECORD = EVIDENCE / "flatworld_ppo_5m_development_pilot_launch.json"


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
    manifest_hash = sha256(MANIFEST)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["status"] != "frozen_not_launched":
        raise RuntimeError(f"unexpected manifest status: {manifest['status']}")
    if manifest["contamination_controls"]["confirmatory_usefulness_evaluation_authorized"]:
        raise RuntimeError("manifest unexpectedly authorizes confirmatory usefulness evaluation")

    run_root = Path(manifest["contamination_controls"]["run_directory_must_not_exist_before_launch"])
    if run_root.exists():
        raise RuntimeError(f"run directory already exists; refusing to resume or overwrite: {run_root}")
    run_root.mkdir(parents=True)

    stdout_path = run_root / "launch.stdout.log"
    stderr_path = run_root / "launch.stderr.log"
    env = os.environ.copy()
    env.update(manifest["runtime"]["required_environment"])
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            manifest["command"],
            cwd=manifest["runtime"]["working_directory"],
            env=env,
            stdout=stdout,
            stderr=stderr,
            creationflags=creationflags,
        )

    record = {
        "schema_version": "tlrl-flatworld-ppo-development-pilot-launch/1",
        "status": "launched",
        "scope": manifest["scope"],
        "manifest_path": "evidence/flatworld/ppo/development/training/flatworld_ppo_5m_development_pilot_manifest.json",
        "manifest_sha256": manifest_hash,
        "pid": process.pid,
        "launched_epoch_seconds": time.time(),
        "experiment_name": manifest["training"]["experiment_name"],
        "seed": manifest["training"]["seed"],
        "target_steps": manifest["training"]["target_steps"],
        "expected_final_steps_due_to_batch_boundary": manifest["training"][
            "expected_final_steps_due_to_batch_boundary"
        ],
        "run_directory": str(run_root),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "utility_outcomes_inspected_before_launch": False,
        "confirmatory_usefulness_evaluation_authorized": False,
    }
    payload = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write(LAUNCH_RECORD, payload)
    atomic_write(
        LAUNCH_RECORD.with_suffix(LAUNCH_RECORD.suffix + ".sha256"),
        f"{hashlib.sha256(payload).hexdigest()}  {LAUNCH_RECORD.name}\n".encode("ascii"),
    )
    print(json.dumps({"status": "launched", "pid": process.pid, "run_directory": str(run_root)}))


if __name__ == "__main__":
    main()
