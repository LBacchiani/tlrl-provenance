#!/usr/bin/env python
"""Launch the frozen LetterEnv PPO confirmatory campaign.

Mirrors scripts/launch_deepltl_point_ppo_confirmatory_campaign.py exactly in
structure and discipline (frozen protocol read first, one campaign manifest
plus one immutable per-seed manifest written before any process starts, the
launcher itself supervises and records completion). Two deliberate
differences from that precedent, both documented in the frozen protocol
itself, not just here:

- Single GPU (GPU1) for every seed, not a two-GPU cycle -- this project
  dedicates GPU0 to the letter_env DQN pillar throughout.
- max_concurrent defaults to 1 (fully sequential), not 2. Running multiple
  PPO seeds concurrently on one GPU is exactly the failure mode that caused
  the DQN pilot's multi-hour throughput collapse earlier in this project
  (see DQN_PROTOCOL.md) -- avoided here deliberately, not by accident.

Like the deepltl_point precedent, this script does not itself check whether
the development pilot's outcomes were inspected -- that is a judgment call by the
experimenter, made before running this script, recorded in
PPO_PROTOCOL.md, not something the script can verify on its own.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import platform
import subprocess
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / ".runtime" / "deep-ltl"
PYTHON = Path(os.environ.get("TLRL_PYTHON", sys.executable))
RABINIZER_JAR = ROOT / "rabinizer-4" / "lib" / "rabinizer.jar"
PROTOCOL = ROOT / "evidence" / "letter_env" / "ppo" / "confirmatory" / "protocol" / "letter_env_ppo_paper_benchmark_protocol.json"
PROTOCOL_HASH = PROTOCOL.with_suffix(PROTOCOL.suffix + ".sha256")
EVIDENCE = ROOT / "evidence" / "letter_env" / "ppo" / "confirmatory" / "training"
CAMPAIGN_ID = "letter_env_ppo_5m_confirmatory_5seed"
ENVIRONMENT_ID = "LetterEnv-v0"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
    temporary.replace(path)


def write_json_with_hash(path: Path, value: dict[str, Any]) -> str:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write(path, payload)
    digest = hashlib.sha256(payload).hexdigest()
    atomic_write(
        path.with_suffix(path.suffix + ".sha256"),
        f"{digest}  {path.name}\n".encode("ascii"),
    )
    return digest


def verify_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file():
        raise RuntimeError(f"missing file: {path}")
    if not sidecar.is_file():
        raise RuntimeError(f"missing hash sidecar: {sidecar}")
    expected = sidecar.read_text(encoding="ascii").split()[0].lower()
    actual = sha256(path).lower()
    if actual != expected:
        raise RuntimeError(f"hash mismatch for {path}: {actual} != {expected}")
    return actual


def parse_gpu_cycle(raw: str) -> list[str]:
    gpus = [item.strip() for item in raw.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpu-cycle must contain at least one GPU id")
    return gpus


def build_command(seed: int, training: dict[str, Any], experiment_name: str) -> list[str]:
    return [
        str(PYTHON.resolve()),
        "src/train/train_ppo.py",
        "--env",
        ENVIRONMENT_ID,
        "--steps_per_process",
        str(training["steps_per_process"]),
        "--batch_size",
        str(training["batch_size"]),
        "--lr",
        str(training["learning_rate"]),
        "--discount",
        str(training["discount"]),
        "--gae_lambda",
        str(training["gae_lambda"]),
        "--entropy_coef",
        str(training["entropy_coefficient"]),
        "--log_interval",
        "5",
        "--save_interval",
        "20",
        "--epochs",
        str(training["epochs"]),
        "--num_steps",
        str(training["target_steps"]),
        "--model_config",
        ENVIRONMENT_ID,
        "--curriculum",
        ENVIRONMENT_ID,
        "--name",
        experiment_name,
        "--seed",
        str(seed),
        "--device",
        "gpu",
        "--num_procs",
        str(training["num_processes"]),
        "--log_csv",
        "--save",
        "--no-log_wandb",
    ]


def run_root_for(experiment_name: str, seed: int) -> Path:
    return UPSTREAM / "experiments" / "ppo" / ENVIRONMENT_ID / experiment_name / str(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--gpu-cycle", default="1")
    args = parser.parse_args()
    if args.max_concurrent < 1:
        raise ValueError("--max-concurrent must be positive")

    protocol_digest = verify_sidecar(PROTOCOL)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["benchmark_id"] != "deepltl-letterenv-official-eval-50":
        raise RuntimeError("unexpected benchmark id in protocol")
    training = protocol["training"]
    seeds = list(training["confirmatory_training_seeds"])
    if len(seeds) != 5:
        raise RuntimeError(f"expected exactly five confirmatory seeds, got {seeds}")
    if training.get("development_seed_excluded_from_primary") in seeds:
        raise RuntimeError("development seed appears in confirmatory seeds")

    # Launch-time namespace check: confirmatory seeds must be disjoint from
    # this pillar's own pilot seed AND from the letter_env DQN pillar's
    # entire seed range (930000-930005), since both pillars share one
    # evidence tree and formula corpus.
    dqn_pillar_seeds = {930000, 930001, 930002, 930003, 930004, 930005}
    forbidden_seed_values = {
        int(training["development_seed_excluded_from_primary"]),
        *map(int, seeds),
        *dqn_pillar_seeds,
    }
    if len(forbidden_seed_values) != 1 + len(seeds) + len(dqn_pillar_seeds):
        raise RuntimeError("seed namespace collision between PPO confirmatory seeds, pilot seed, or DQN pillar seeds")

    if not PYTHON.is_file():
        raise RuntimeError(f"missing Python runtime: {PYTHON}")
    if not RABINIZER_JAR.is_file():
        raise RuntimeError(f"missing Rabinizer jar: {RABINIZER_JAR}")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True
    ).strip()

    gpus = parse_gpu_cycle(args.gpu_cycle)
    experiment_names = {
        seed: f"v3_confirm_ppo_5m_letter_env_seed_{seed}" for seed in seeds
    }
    run_roots = {
        seed: run_root_for(experiment_names[seed], seed) for seed in seeds
    }
    for seed, run_root in run_roots.items():
        if run_root.exists():
            raise RuntimeError(
                f"run directory already exists for seed {seed}; refusing to resume or overwrite: {run_root}"
            )

    campaign_manifest = {
        "schema_version": "tlrl-letter-env-ppo-confirmatory-campaign/1",
        "status": "frozen_before_launch",
        "scope": "confirmatory_training_only_no_result_inspection",
        "campaign_id": CAMPAIGN_ID,
        "created_epoch_seconds": time.time(),
        "benchmark_id": protocol["benchmark_id"],
        "protocol_path": "evidence/letter_env/ppo/confirmatory/protocol/letter_env_ppo_paper_benchmark_protocol.json",
        "protocol_sha256": protocol_digest,
        "upstream": {
            "path": str(UPSTREAM.resolve()),
            "commit": commit,
        },
        "runtime": {
            "python": platform.python_version(),
            "python_executable": str(PYTHON.resolve()),
            "rabinizer_jar": str(RABINIZER_JAR.resolve()),
            "rabinizer_jar_sha256": sha256(RABINIZER_JAR),
            "max_concurrent": args.max_concurrent,
            "gpu_cycle": gpus,
        },
        "training": {
            "algorithm": "formula_conditioned_recurrent_ppo",
            "seeds": seeds,
            "target_steps": training["target_steps"],
            "expected_final_steps_due_to_batch_boundary": training[
                "batch_rounded_expected_steps"
            ],
            "num_processes": training["num_processes"],
            "steps_per_process": training["steps_per_process"],
            "batch_size": training["batch_size"],
            "learning_rate": training["learning_rate"],
            "discount": training["discount"],
            "gae_lambda": training["gae_lambda"],
            "entropy_coefficient": training["entropy_coefficient"],
            "epochs": training["epochs"],
        },
        "launch_time_seed_namespace_check": {
            "confirmatory_training_seeds": seeds,
            "development_training_seed": training["development_seed_excluded_from_primary"],
            "letter_env_dqn_pillar_seeds": sorted(dqn_pillar_seeds),
            "status": "passed",
        },
        "contamination_controls": {
            "dqn_pillar_dependency": False,
            "development_seed_excluded": True,
            "outcomes_inspected_before_launch": False,
            "primary_policy_selection_rule": training["primary_policy_selection_rule"],
        },
    }
    manifest_path = EVIDENCE / f"{CAMPAIGN_ID}_manifest.json"
    manifest_digest = write_json_with_hash(manifest_path, campaign_manifest)

    seed_manifests: dict[int, Path] = {}
    for index, seed in enumerate(seeds):
        experiment_name = experiment_names[seed]
        run_root = run_roots[seed]
        assigned_gpu = gpus[index % len(gpus)]
        command = build_command(seed, training, experiment_name)
        seed_manifest = {
            "schema_version": "tlrl-letter-env-ppo-confirmatory-seed/1",
            "status": "frozen_before_launch",
            "scope": "confirmatory_training_only_no_result_inspection",
            "campaign_id": CAMPAIGN_ID,
            "campaign_manifest_path": f"evidence/letter_env/ppo/confirmatory/training/{manifest_path.name}",
            "campaign_manifest_sha256": manifest_digest,
            "seed": seed,
            "assigned_cuda_visible_devices": assigned_gpu,
            "experiment_name": experiment_name,
            "run_directory": str(run_root.resolve()),
            "stdout_path": str((run_root / "launch.stdout.log").resolve()),
            "stderr_path": str((run_root / "launch.stderr.log").resolve()),
            "command": command,
            "environment": {
                "PYTHONPATH": "src/",
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1",
                "RABINIZER_JAR": str(RABINIZER_JAR.resolve()),
                "CUDA_VISIBLE_DEVICES": assigned_gpu,
            },
        }
        seed_path = EVIDENCE / f"{CAMPAIGN_ID}_seed_{seed}_manifest.json"
        write_json_with_hash(seed_path, seed_manifest)
        seed_manifests[seed] = seed_path

    launch_record_path = EVIDENCE / f"{CAMPAIGN_ID}_launch.json"
    pending = list(seeds)
    running: dict[int, dict[str, Any]] = {}
    completed: dict[int, dict[str, Any]] = {}
    launch_events: list[dict[str, Any]] = []
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    def persist(status: str) -> None:
        record = {
            "schema_version": "tlrl-letter-env-ppo-confirmatory-launch/1",
            "status": status,
            "campaign_id": CAMPAIGN_ID,
            "campaign_manifest_path": f"evidence/letter_env/ppo/confirmatory/training/{manifest_path.name}",
            "campaign_manifest_sha256": manifest_digest,
            "seed_manifests": {
                str(seed): f"evidence/letter_env/ppo/confirmatory/training/{path.name}"
                for seed, path in seed_manifests.items()
            },
            "pending": pending,
            "running": {
                str(seed): {
                    "pid": item["process"].pid,
                    "gpu": item["gpu"],
                    "run_directory": str(item["run_root"]),
                    "launched_epoch_seconds": item["launched_epoch_seconds"],
                }
                for seed, item in running.items()
            },
            "completed": completed,
            "events": launch_events,
            "updated_epoch_seconds": time.time(),
        }
        write_json_with_hash(launch_record_path, record)

    while pending or running:
        while pending and len(running) < args.max_concurrent:
            seed = pending.pop(0)
            seed_manifest = json.loads(seed_manifests[seed].read_text(encoding="utf-8"))
            run_root = Path(seed_manifest["run_directory"])
            run_root.mkdir(parents=True)
            env = os.environ.copy()
            env.update(seed_manifest["environment"])
            stdout_path = Path(seed_manifest["stdout_path"])
            stderr_path = Path(seed_manifest["stderr_path"])
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                process = subprocess.Popen(
                    seed_manifest["command"],
                    cwd=UPSTREAM,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    creationflags=creationflags,
                )
            event = {
                "event": "launched",
                "seed": seed,
                "pid": process.pid,
                "gpu": seed_manifest["assigned_cuda_visible_devices"],
                "epoch_seconds": time.time(),
                "run_directory": str(run_root),
            }
            launch_events.append(event)
            running[seed] = {
                "process": process,
                "gpu": seed_manifest["assigned_cuda_visible_devices"],
                "run_root": run_root,
                "launched_epoch_seconds": event["epoch_seconds"],
            }
            print(json.dumps(event), flush=True)
            persist("running")

        time.sleep(30)
        finished = []
        for seed, item in running.items():
            returncode = item["process"].poll()
            if returncode is not None:
                finished.append((seed, returncode))
        for seed, returncode in finished:
            item = running.pop(seed)
            event = {
                "event": "completed" if returncode == 0 else "failed",
                "seed": seed,
                "pid": item["process"].pid,
                "gpu": item["gpu"],
                "returncode": returncode,
                "epoch_seconds": time.time(),
                "run_directory": str(item["run_root"]),
            }
            completed[str(seed)] = event
            launch_events.append(event)
            print(json.dumps(event), flush=True)
            persist("running" if pending or running else "completed")
            if returncode != 0:
                raise RuntimeError(f"training failed for seed {seed}; see {item['run_root']}")
        if running:
            print(
                json.dumps(
                    {
                        "event": "heartbeat",
                        "pending": pending,
                        "running": {
                            str(seed): item["process"].pid for seed, item in running.items()
                        },
                        "completed": sorted(completed),
                        "epoch_seconds": time.time(),
                    }
                ),
                flush=True,
            )
            persist("running")

    print(
        json.dumps(
            {
                "status": "completed",
                "campaign_id": CAMPAIGN_ID,
                "completed_seeds": sorted(completed),
                "launch_record": str(launch_record_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
