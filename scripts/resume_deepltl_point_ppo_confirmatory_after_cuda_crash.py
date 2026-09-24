#!/usr/bin/env python
"""Resume the frozen PointWorld campaign after seed 920003's CUDA-driver crash.

This is an operational recovery controller, not a new experimental design.  It
keeps the original manifests and launch record immutable, supervises the still
running seed 920002, completes unaffected seeds 920004/920005, and retries
920003 only after all four unaffected seeds have completed.  The failed,
zero-step attempt is archived rather than overwritten.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any

import psutil


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / ".runtime" / "deep-ltl"
EVIDENCE = ROOT / "evidence" / "deepltl_point" / "confirmatory" / "training"
CAMPAIGN_ID = "deepltl_point_ppo_5m_confirmatory_5seed"
ORIGINAL_LAUNCH = EVIDENCE / f"{CAMPAIGN_ID}_launch.json"
RECOVERY_RECORD = EVIDENCE / f"{CAMPAIGN_ID}_cuda_recovery.json"
EXPECTED_FINAL_STEPS = 5_046_272
UNAFFECTED_SEEDS = {920001, 920002, 920004, 920005}
RETRY_SEED = 920003
GPU_COOLDOWN_SECONDS = 120


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    expected = sidecar.read_text(encoding="ascii").split()[0].lower()
    actual = sha256(path).lower()
    if actual != expected:
        raise RuntimeError(f"hash mismatch for {path}: {actual} != {expected}")
    return actual


def atomic_json_with_hash(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
    temporary.replace(path)
    digest = hashlib.sha256(payload).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )


def manifest_path(seed: int) -> Path:
    return EVIDENCE / f"{CAMPAIGN_ID}_seed_{seed}_manifest.json"


def load_manifest(seed: int) -> dict[str, Any]:
    path = manifest_path(seed)
    verify_sidecar(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["seed"] != seed or value["campaign_id"] != CAMPAIGN_ID:
        raise RuntimeError(f"unexpected seed manifest contents: {path}")
    return value


def final_step(run_root: Path) -> int | None:
    log = run_root / "log.csv"
    if not log.is_file():
        return None
    with log.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    return int(rows[-1]["num_steps"])


def validate_completed(run_root: Path) -> int:
    step = final_step(run_root)
    if step is None or step < EXPECTED_FINAL_STEPS:
        raise RuntimeError(f"incomplete training output at {run_root}: step={step}")
    for required in ("status.pth", "ltl_net.pth"):
        if not (run_root / required).is_file():
            raise RuntimeError(f"missing {required} at {run_root}")
    return step


def process_is_original(pid: int, launched_epoch_seconds: float) -> bool:
    try:
        process = psutil.Process(pid)
        return (
            process.is_running()
            and process.name().lower().startswith("python")
            and abs(process.create_time() - launched_epoch_seconds) < 10
        )
    except (psutil.Error, OSError):
        return False


def zero_step_inventory(run_root: Path) -> list[dict[str, Any]]:
    forbidden = ("log.csv", "status.pth", "ltl_net.pth")
    present_forbidden = [name for name in forbidden if (run_root / name).exists()]
    if present_forbidden:
        raise RuntimeError(
            f"refusing automatic retry: failed attempt has training artifacts {present_forbidden}"
        )
    inventory = []
    for path in sorted(item for item in run_root.rglob("*") if item.is_file()):
        inventory.append(
            {
                "relative_path": path.relative_to(run_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return inventory


def launch(seed: int, manifest: dict[str, Any]) -> dict[str, Any]:
    run_root = Path(manifest["run_directory"])
    if run_root.exists():
        raise RuntimeError(f"refusing to overwrite existing run directory: {run_root}")
    run_root.mkdir(parents=True)
    environment = os.environ.copy()
    environment.update(manifest["environment"])
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    stdout_path = Path(manifest["stdout_path"])
    stderr_path = Path(manifest["stderr_path"])
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            manifest["command"],
            cwd=UPSTREAM,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            creationflags=creationflags,
        )
    return {
        "process": process,
        "pid": process.pid,
        "seed": seed,
        "gpu": manifest["assigned_cuda_visible_devices"],
        "run_root": run_root,
        "launched_epoch_seconds": time.time(),
    }


def public_running(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "pid": item["pid"],
        "gpu": item["gpu"],
        "run_directory": str(item["run_root"]),
        "launched_epoch_seconds": item["launched_epoch_seconds"],
        "attached_from_original_launcher": item.get("attached", False),
    }


def main() -> None:
    original_digest = verify_sidecar(ORIGINAL_LAUNCH)
    original = json.loads(ORIGINAL_LAUNCH.read_text(encoding="utf-8"))
    if original["campaign_id"] != CAMPAIGN_ID:
        raise RuntimeError("unexpected original launch record")
    failed = original["completed"].get(str(RETRY_SEED), {})
    if failed.get("event") != "failed" or failed.get("returncode") != 3221226505:
        raise RuntimeError("the expected pre-training CUDA failure is not recorded")

    manifests = {seed: load_manifest(seed) for seed in range(920001, 920006)}
    completed: dict[int, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    failures: dict[int, dict[str, Any]] = {}
    gpu_available_after = {"0": 0.0, "1": 0.0}

    seed1_root = Path(manifests[920001]["run_directory"])
    completed[920001] = {
        "event": "validated_original_completion",
        "steps": validate_completed(seed1_root),
        "run_directory": str(seed1_root),
        "epoch_seconds": time.time(),
    }

    original_seed2 = original["running"].get("920002")
    if original_seed2 is None:
        raise RuntimeError("seed 920002 is not recorded as running")
    seed2_root = Path(manifests[920002]["run_directory"])
    attached: dict[int, dict[str, Any]] = {}
    if process_is_original(
        int(original_seed2["pid"]), float(original_seed2["launched_epoch_seconds"])
    ):
        attached[920002] = {
            "pid": int(original_seed2["pid"]),
            "gpu": str(original_seed2["gpu"]),
            "run_root": seed2_root,
            "launched_epoch_seconds": float(original_seed2["launched_epoch_seconds"]),
            "attached": True,
        }
    else:
        completed[920002] = {
            "event": "validated_original_completion",
            "steps": validate_completed(seed2_root),
            "run_directory": str(seed2_root),
            "epoch_seconds": time.time(),
        }

    pending = [920005, 920004]
    running: dict[int, dict[str, Any]] = {}
    archived_failed_attempt: dict[str, Any] | None = None
    retry_launched = False

    def persist(status: str) -> None:
        atomic_json_with_hash(
            RECOVERY_RECORD,
            {
                "schema_version": "tlrl-deepltl-point-ppo-cuda-recovery/1",
                "status": status,
                "campaign_id": CAMPAIGN_ID,
                "scope": "operational_recovery_only_frozen_training_design_unchanged",
                "original_launch_record": f"evidence/{ORIGINAL_LAUNCH.name}",
                "original_launch_sha256": original_digest,
                "reason": {
                    "seed": RETRY_SEED,
                    "stage": "before_first_training_step",
                    "windows_event": "BEX64",
                    "faulting_module": "nvcuda64.dll",
                    "exception_code": "0xc0000409",
                    "classification": "infrastructure_failure_not_policy_outcome",
                },
                "user_authorized_order": (
                    "complete unaffected seeds 920001/920002/920004/920005; "
                    "then retry 920003"
                ),
                "gpu_context_cooldown_seconds": GPU_COOLDOWN_SECONDS,
                "pending_unaffected": pending,
                "attached": {str(k): public_running(v) for k, v in attached.items()},
                "running": {str(k): public_running(v) for k, v in running.items()},
                "completed": {str(k): v for k, v in completed.items()},
                "failures": {str(k): v for k, v in failures.items()},
                "failed_attempt_archive": archived_failed_attempt,
                "retry_launched": retry_launched,
                "events": events,
                "updated_epoch_seconds": time.time(),
            },
        )

    persist("running")
    while True:
        now = time.time()

        for seed, item in list(attached.items()):
            if not process_is_original(item["pid"], item["launched_epoch_seconds"]):
                attached.pop(seed)
                try:
                    step = validate_completed(item["run_root"])
                    event = {
                        "event": "attached_seed_completed",
                        "seed": seed,
                        "steps": step,
                        "epoch_seconds": now,
                        "run_directory": str(item["run_root"]),
                    }
                    completed[seed] = event
                except RuntimeError as error:
                    event = {
                        "event": "attached_seed_failed",
                        "seed": seed,
                        "error": str(error),
                        "epoch_seconds": now,
                        "run_directory": str(item["run_root"]),
                    }
                    failures[seed] = event
                events.append(event)
                gpu_available_after[item["gpu"]] = now + GPU_COOLDOWN_SECONDS
                print(json.dumps(event), flush=True)

        for seed, item in list(running.items()):
            returncode = item["process"].poll()
            if returncode is None:
                continue
            running.pop(seed)
            if returncode == 0:
                step = validate_completed(item["run_root"])
                event = {
                    "event": "completed",
                    "seed": seed,
                    "steps": step,
                    "returncode": returncode,
                    "epoch_seconds": now,
                    "run_directory": str(item["run_root"]),
                }
                completed[seed] = event
            else:
                event = {
                    "event": "failed",
                    "seed": seed,
                    "returncode": returncode,
                    "epoch_seconds": now,
                    "run_directory": str(item["run_root"]),
                }
                failures[seed] = event
            events.append(event)
            gpu_available_after[item["gpu"]] = now + GPU_COOLDOWN_SECONDS
            print(json.dumps(event), flush=True)

        occupied_gpus = {
            item["gpu"] for item in [*attached.values(), *running.values()]
        }
        for seed in list(pending):
            gpu = str(manifests[seed]["assigned_cuda_visible_devices"])
            if gpu in occupied_gpus or now < gpu_available_after[gpu]:
                continue
            item = launch(seed, manifests[seed])
            running[seed] = item
            pending.remove(seed)
            occupied_gpus.add(gpu)
            event = {
                "event": "launched_unaffected_seed",
                "seed": seed,
                "pid": item["pid"],
                "gpu": gpu,
                "epoch_seconds": item["launched_epoch_seconds"],
                "run_directory": str(item["run_root"]),
            }
            events.append(event)
            print(json.dumps(event), flush=True)

        if (
            UNAFFECTED_SEEDS.issubset(completed)
            and not failures
            and not retry_launched
            and RETRY_SEED not in running
        ):
            failed_root = Path(manifests[RETRY_SEED]["run_directory"])
            inventory = zero_step_inventory(failed_root)
            archive_root = failed_root.with_name(
                f"{failed_root.name}.failed-pretraining-nvcuda-{int(failed['epoch_seconds'])}"
            )
            if archive_root.exists():
                raise RuntimeError(f"failed-attempt archive already exists: {archive_root}")
            failed_root.rename(archive_root)
            archived_failed_attempt = {
                "original_run_directory": str(failed_root),
                "archive_directory": str(archive_root),
                "files": inventory,
                "archived_epoch_seconds": time.time(),
            }
            gpu = str(manifests[RETRY_SEED]["assigned_cuda_visible_devices"])
            wait_seconds = max(0.0, gpu_available_after[gpu] - time.time())
            if wait_seconds:
                persist("waiting_for_retry_gpu_cooldown")
                time.sleep(wait_seconds)
            item = launch(RETRY_SEED, manifests[RETRY_SEED])
            running[RETRY_SEED] = item
            retry_launched = True
            event = {
                "event": "relaunched_pretraining_cuda_failure",
                "seed": RETRY_SEED,
                "pid": item["pid"],
                "gpu": gpu,
                "epoch_seconds": item["launched_epoch_seconds"],
                "run_directory": str(item["run_root"]),
                "archived_attempt": str(archive_root),
            }
            events.append(event)
            print(json.dumps(event), flush=True)

        if retry_launched and RETRY_SEED in completed:
            persist("completed")
            print(json.dumps({"status": "completed", "completed": sorted(completed)}))
            return
        if failures and not running and not attached:
            persist("incomplete_due_to_failure")
            raise RuntimeError(f"recovery campaign has failures: {sorted(failures)}")

        persist("running")
        print(
            json.dumps(
                {
                    "event": "heartbeat",
                    "pending_unaffected": pending,
                    "attached": sorted(attached),
                    "running": sorted(running),
                    "completed": sorted(completed),
                    "failures": sorted(failures),
                    "epoch_seconds": time.time(),
                }
            ),
            flush=True,
        )
        time.sleep(30)


if __name__ == "__main__":
    main()
