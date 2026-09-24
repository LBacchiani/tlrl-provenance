#!/usr/bin/env python
"""Generic, resumable, fail-closed confirmatory semantic-audit runner.

Benchmark logic is loaded through the adapter declared in a JSON campaign
configuration.  The controller freezes a complete plan before any rollout,
then launches process-isolated subject/task shards.  A summary is emitted only
after exact completeness and semantic-agreement checks pass.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tlrl_provenance.campaign import (  # noqa: E402
    RECORD_SCHEMA,
    atomic_write,
    build_plan,
    build_summary,
    canonical_json,
    expected_keys,
    freeze_plan,
    json_ready,
    load_adapter,
    load_campaign_config,
    read_jsonl,
    record_key,
    render_markdown,
    sha256_bytes,
    task_shards,
    validate_record,
)


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def output_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    return {key: resolve(ROOT, value) for key, value in config["outputs"].items()}


def shard_path(paths: Mapping[str, Path], shard_id: str) -> Path:
    return paths["shard_directory"] / f"{shard_id}.jsonl"


def shard_log_paths(paths: Mapping[str, Path], shard_id: str) -> tuple[Path, Path]:
    return (
        paths["shard_directory"] / f"{shard_id}.stdout.log",
        paths["shard_directory"] / f"{shard_id}.stderr.log",
    )


def _inventory(plan: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {item["subject_id"]: item for item in plan["subjects"]},
        {item["task_id"]: item for item in plan["tasks"]},
    )


def shard_expected_keys(plan: Mapping[str, Any], shard: Mapping[str, Any]) -> set[tuple[str, str, int, int]]:
    task_ids = set(shard["task_ids"])
    return {
        record_key(cell)
        for cell in plan["cells"]
        if cell["subject_id"] == shard["subject_id"] and cell["task_id"] in task_ids
    }


def validate_shard(path: Path, plan: Mapping[str, Any], shard: Mapping[str, Any]) -> tuple[bool, int]:
    expected = shard_expected_keys(plan, shard)
    seen = set()
    for record in read_jsonl(path):
        validate_record(record, plan)
        key = record_key(record)
        if key not in expected or key in seen:
            raise RuntimeError(f"invalid or duplicate record in shard {path}: {key}")
        seen.add(key)
    return seen == expected, len(seen)


def run_worker(config_path: Path, plan_path: Path, shard_id: str) -> None:
    config = load_campaign_config(config_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if sha256_bytes(canonical_json({key: value for key, value in plan.items() if key != "plan_sha256"})) != plan["plan_sha256"]:
        raise RuntimeError("frozen plan self-digest mismatch")
    shards = {item["shard_id"]: item for item in task_shards(plan, int(config["design"]["task_shard_size"]))}
    if shard_id not in shards:
        raise ValueError(f"unknown shard: {shard_id}")
    shard = shards[shard_id]
    paths = output_paths(config)
    records_path = shard_path(paths, shard_id)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    expected = shard_expected_keys(plan, shard)
    existing: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for record in read_jsonl(records_path):
        validate_record(record, plan)
        key = record_key(record)
        if key not in expected or key in existing:
            raise RuntimeError(f"unexpected or duplicate resumed shard record: {key}")
        existing[key] = record

    adapter = load_adapter(config["adapter"], config.get("adapter_config", {}), ROOT)
    subjects = {item.subject_id: item for item in adapter.subjects()}
    tasks = {item.task_id: item for item in adapter.tasks()}
    if set(subjects) != {item["subject_id"] for item in plan["subjects"]}:
        raise RuntimeError("adapter subject inventory changed after plan freeze")
    if set(tasks) != {item["task_id"] for item in plan["tasks"]}:
        raise RuntimeError("adapter task inventory changed after plan freeze")

    subject = subjects[shard["subject_id"]]
    cell_lookup = {
        (cell["task_id"], int(cell["episode_index"])): cell
        for cell in plan["cells"]
        if cell["subject_id"] == subject.subject_id and cell["task_id"] in set(shard["task_ids"])
    }
    for task_id in shard["task_ids"]:
        task = tasks[task_id]
        pending = [
            cell_lookup[(task_id, episode)]
            for episode in range(int(plan["design"]["episodes_per_task"]))
            if record_key(cell_lookup[(task_id, episode)]) not in existing
        ]
        if not pending:
            continue
        session = adapter.open_session(subject, task)
        try:
            for cell in pending:
                # scenario_seed is only present when the campaign config
                # declares a scenario_namespace; omitting it here for
                # adapters that predate this (e.g. the frozen PointLtl2
                # session) keeps their call signature unchanged.
                scenario_kwargs = (
                    {"scenario_seed": int(cell["scenario_seed"])} if "scenario_seed" in cell else {}
                )
                outcome = dict(
                    session.run_episode(
                        evaluation_seed=int(cell["evaluation_seed"]),
                        deterministic=bool(plan["design"]["deterministic"]),
                        verification_mode=str(plan["design"]["verification_mode"]),
                        **scenario_kwargs,
                    )
                )
                record = {
                    "schema_version": RECORD_SCHEMA,
                    "campaign_id": plan["campaign_id"],
                    "benchmark_id": plan["benchmark_id"],
                    "plan_sha256": plan["plan_sha256"],
                    "engine": plan["engine"],
                    "subject_id": subject.subject_id,
                    "subject_metadata": json_ready(subject.metadata),
                    "task_id": task.task_id,
                    "formula": task.formula,
                    "formula_family": task.family,
                    "task_metadata": json_ready(task.metadata or {}),
                    "episode_index": int(cell["episode_index"]),
                    "evaluation_seed": int(cell["evaluation_seed"]),
                    "deterministic": bool(plan["design"]["deterministic"]),
                    "verification_mode": plan["design"]["verification_mode"],
                    **json_ready(outcome),
                }
                validate_record(record, plan)
                with records_path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                existing[record_key(record)] = record
                print(
                    json.dumps(
                        {"status": "episode_completed", "shard_id": shard_id,
                         "task_id": task_id, "episode_index": cell["episode_index"],
                         "completed": len(existing), "expected": len(expected)},
                        sort_keys=True,
                    ), flush=True,
                )
        finally:
            session.close()
    complete, count = validate_shard(records_path, plan, shard)
    if not complete:
        raise RuntimeError(f"worker ended with incomplete shard: {count}/{len(expected)}")


def merge_and_report(config: Mapping[str, Any], plan: Mapping[str, Any], paths: Mapping[str, Path]) -> dict[str, Any]:
    shards = task_shards(plan, int(config["design"]["task_shard_size"]))
    records = []
    for shard in shards:
        path = shard_path(paths, shard["shard_id"])
        complete, _ = validate_shard(path, plan, shard)
        if not complete:
            raise RuntimeError(f"cannot summarize incomplete shard: {path}")
        records.extend(read_jsonl(path))
    # build_summary performs the final exact missing/duplicate/unexpected audit.
    summary = build_summary(records, plan, confidence=float(config["design"].get("confidence", 0.95)))
    records.sort(key=record_key)
    records_payload = b"".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for record in records
    )
    atomic_write(paths["records"], records_payload)
    summary["artifacts"] = {
        "records_path": str(paths["records"]),
        "records_sha256": sha256_bytes(records_payload),
        "plan_path": str(paths["plan"]),
        "plan_sha256": plan["plan_sha256"],
    }
    summary_payload = canonical_json(summary)
    atomic_write(paths["summary"], summary_payload)
    atomic_write(paths["summary"].with_suffix(paths["summary"].suffix + ".sha256"),
                 f"{sha256_bytes(summary_payload)}  {paths['summary'].name}\n".encode("ascii"))
    atomic_write(paths["report"], render_markdown(summary).encode("utf-8"))
    return summary


def execute_controller(config_path: Path, config: Mapping[str, Any], plan: Mapping[str, Any], paths: Mapping[str, Path]) -> dict[str, Any]:
    shards = task_shards(plan, int(config["design"]["task_shard_size"]))
    pending = []
    for shard in shards:
        complete, _ = validate_shard(shard_path(paths, shard["shard_id"]), plan, shard)
        if not complete:
            pending.append(shard)
    workers = int(config["design"]["workers"])
    active: list[tuple[subprocess.Popen[bytes], dict[str, Any], Any, Any]] = []
    completed = len(shards) - len(pending)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    for key, value in config.get("environment", {}).items():
        env[str(key)] = str(resolve(ROOT, value) if str(key).endswith("_PATH") or str(key).endswith("_JAR") else value)
    try:
        while pending or active:
            while pending and len(active) < workers:
                shard = pending.pop(0)
                stdout_path, stderr_path = shard_log_paths(paths, shard["shard_id"])
                stdout_path.parent.mkdir(parents=True, exist_ok=True)
                stdout, stderr = stdout_path.open("ab"), stderr_path.open("ab")
                command = [sys.executable, str(Path(__file__).resolve()), "--config", str(config_path),
                           "--worker", "--shard-id", shard["shard_id"]]
                process = subprocess.Popen(
                    command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                    creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                )
                active.append((process, shard, stdout, stderr))
                print(json.dumps({"status": "shard_launched", "shard_id": shard["shard_id"],
                                  "active": len(active), "completed": completed, "total": len(shards)}, sort_keys=True), flush=True)
            time.sleep(0.5)
            remaining = []
            for process, shard, stdout, stderr in active:
                code = process.poll()
                if code is None:
                    remaining.append((process, shard, stdout, stderr))
                    continue
                stdout.close(); stderr.close()
                if code != 0:
                    _, stderr_path = shard_log_paths(paths, shard["shard_id"])
                    raise RuntimeError(f"shard failed ({code}): {shard['shard_id']}; see {stderr_path}")
                complete, _ = validate_shard(shard_path(paths, shard["shard_id"]), plan, shard)
                if not complete:
                    raise RuntimeError(f"successful worker produced incomplete shard: {shard['shard_id']}")
                completed += 1
                print(json.dumps({"status": "shard_completed", "shard_id": shard["shard_id"],
                                  "completed": completed, "total": len(shards)}, sort_keys=True), flush=True)
            active = remaining
    except BaseException:
        # A failed or interrupted controller must not orphan rollout workers.
        for process, _, stdout, stderr in active:
            if process.poll() is None:
                process.terminate()
        for process, _, stdout, stderr in active:
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            if not stdout.closed:
                stdout.close()
            if not stderr.closed:
                stderr.close()
        raise
    return merge_and_report(config, plan, paths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="run the frozen campaign")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--shard-id", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_campaign_config(config_path)
    paths = output_paths(config)
    adapter = load_adapter(config["adapter"], config.get("adapter_config", {}), ROOT)
    plan = build_plan(config, adapter, runner_path=Path(__file__).resolve())
    freeze_plan(paths["plan"], plan)
    if args.worker:
        if not args.shard_id:
            raise ValueError("--worker requires --shard-id")
        run_worker(config_path, paths["plan"], args.shard_id)
        return
    if args.summarize_only:
        summary = merge_and_report(config, plan, paths)
        print(json.dumps({"status": "summarized", "records": summary["completeness"]["observed_records"]}, sort_keys=True))
        return
    if args.execute:
        summary = execute_controller(config_path, config, plan, paths)
        print(json.dumps({"status": "completed", "records": summary["completeness"]["observed_records"],
                          "summary": str(paths["summary"]), "report": str(paths["report"])}, sort_keys=True))
        return
    print(json.dumps({"status": "plan_frozen", "plan": str(paths["plan"]),
                      "plan_sha256": plan["plan_sha256"], "expected_records": len(expected_keys(plan)),
                      "message": "Re-run with --execute to collect outcomes."}, sort_keys=True))


if __name__ == "__main__":
    main()
