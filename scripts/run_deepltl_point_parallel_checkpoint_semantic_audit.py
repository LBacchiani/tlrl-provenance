#!/usr/bin/env python
"""Parallel sharded DeepLTL PointWorld semantic checkpoint sweep.

Workers never write to the same JSONL file.  Each subprocess owns one
checkpoint/formula-slice shard; this avoids Windows append races and keeps
MuJoCo resource lifetime bounded to small fresh processes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PYTHON = Path(os.environ.get("TLRL_PYTHON", sys.executable))
RUNNER = SCRIPTS / "run_deepltl_point_checkpoint_semantic_audit.py"
DEVELOPMENT_EVIDENCE = ROOT / "evidence" / "deepltl_point" / "development" / "semantic"
DEFAULT_SHARD_DIR = DEVELOPMENT_EVIDENCE / "checkpoint_semantic_parallel_shards"
DEFAULT_RECORDS_OUTPUT = DEVELOPMENT_EVIDENCE / "deepltl_point_checkpoint_semantic_records_parallel.jsonl"
DEFAULT_SUMMARY_OUTPUT = DEVELOPMENT_EVIDENCE / "deepltl_point_checkpoint_semantic_summary_parallel.json"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_deepltl_point_checkpoint_semantic_audit import (  # noqa: E402
    DEFAULT_EXPERIMENT,
    DEFAULT_SEED,
    available_eval_checkpoints,
    expected_seed,
    json_ready,
    parse_checkpoint_steps,
    record_key,
    summarize_records,
)
from run_deepltl_point_preliminary_semantic_audit import atomic_write, sha256  # noqa: E402
from tlrl_benchmarks.deepltl_point.adapter import load_tasks  # noqa: E402


def resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def formula_slices(total: int, width: int) -> list[tuple[int, int]]:
    if width < 1:
        raise ValueError("formula-shard-size must be positive")
    return [(start, min(start + width, total)) for start in range(0, total, width)]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def shard_paths(shard_dir: Path, checkpoint: int, start: int, end: int) -> tuple[Path, Path, Path, Path]:
    stem = f"ckpt_{checkpoint:07d}_f{start:03d}_{end:03d}"
    return (
        shard_dir / f"{stem}.jsonl",
        shard_dir / f"{stem}.summary.json",
        shard_dir / f"{stem}.stdout.log",
        shard_dir / f"{stem}.stderr.log",
    )


def expected_keys(
    *,
    checkpoints: list[int],
    task_count: int,
    episodes_per_formula: int,
    eval_seed_base: int,
    deterministic: bool,
    experiment: str,
    seed: int,
) -> set[tuple[object, ...]]:
    ranks = {
        step: index
        for index, step in enumerate(available_eval_checkpoints(experiment, seed), start=1)
    }
    tasks = list(load_tasks())[:task_count]
    return {
        (
            checkpoint,
            task.task_id,
            episode_index,
            expected_seed(
                eval_seed_base=eval_seed_base,
                checkpoint_rank=ranks[checkpoint],
                task_index=task_index,
                episode_index=episode_index,
            ),
            deterministic,
        )
        for checkpoint in checkpoints
        for task_index, task in enumerate(tasks, start=1)
        for episode_index in range(episodes_per_formula)
    }


def shard_complete(
    path: Path,
    *,
    checkpoint: int,
    start: int,
    end: int,
    episodes_per_formula: int,
) -> bool:
    records = read_jsonl(path)
    if len(records) != (end - start) * episodes_per_formula:
        return False
    return all(int(record["checkpoint_steps"]) == checkpoint for record in records)


def launch_shard(
    *,
    checkpoint: int,
    start: int,
    end: int,
    args: argparse.Namespace,
    env: dict[str, str],
) -> subprocess.Popen[bytes]:
    records_path, summary_path, stdout_path, stderr_path = shard_paths(
        args.shard_dir,
        checkpoint,
        start,
        end,
    )
    command = [
        str(PYTHON),
        str(RUNNER),
        "--experiment",
        args.experiment,
        "--training-seed",
        str(args.training_seed),
        "--checkpoint-steps",
        str(checkpoint),
        "--episodes-per-formula",
        str(args.episodes_per_formula),
        "--formula-start",
        str(start),
        "--formula-end",
        str(end),
        "--eval-seed-base",
        str(args.eval_seed_base),
        "--verification-mode",
        args.verification_mode,
        "--records-output",
        str(records_path),
        "--summary-output",
        str(summary_path),
    ]
    command.append("--deterministic" if args.deterministic else "--no-deterministic")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout = stdout_path.open("ab")
    stderr = stderr_path.open("ab")
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    process = subprocess.Popen(
        command,
        cwd=ROOT.parent,
        env=env,
        stdout=stdout,
        stderr=stderr,
        creationflags=creationflags,
    )
    # Keep handles alive on the process object so Windows does not close them.
    process._tlrl_stdout = stdout  # type: ignore[attr-defined]
    process._tlrl_stderr = stderr  # type: ignore[attr-defined]
    process._tlrl_shard = (checkpoint, start, end, records_path, summary_path)  # type: ignore[attr-defined]
    return process


def close_process_logs(process: subprocess.Popen[bytes]) -> None:
    for name in ("_tlrl_stdout", "_tlrl_stderr"):
        handle = getattr(process, name, None)
        if handle is not None:
            handle.close()


def write_merged_outputs(
    *,
    args: argparse.Namespace,
    checkpoints: list[int],
    task_count: int,
    expected: set[tuple[object, ...]],
    started: float,
) -> dict[str, Any]:
    records_by_key: dict[tuple[object, ...], dict[str, Any]] = {}
    duplicate_counts: Counter[tuple[object, ...]] = Counter()
    for checkpoint in checkpoints:
        for start, end in formula_slices(task_count, args.formula_shard_size):
            records_path, _, _, _ = shard_paths(args.shard_dir, checkpoint, start, end)
            for record in read_jsonl(records_path):
                key = record_key(record)
                if key not in expected:
                    continue
                if key in records_by_key:
                    duplicate_counts[key] += 1
                records_by_key[key] = record
    missing = sorted(expected - set(records_by_key), key=str)
    records = sorted(
        records_by_key.values(),
        key=lambda item: (
            int(item["checkpoint_steps"]),
            item["task_id"],
            int(item["episode_index"]),
            int(item["seed"]),
        ),
    )
    payload = "".join(json.dumps(json_ready(record), sort_keys=True) + "\n" for record in records)
    atomic_write(args.records_output, payload.encode("utf-8"))
    summary = summarize_records(
        records=records,
        checkpoints=checkpoints,
        task_count=task_count,
        episodes_per_formula=args.episodes_per_formula,
        started=started,
    )
    summary.update(
        {
            "schema_version": "tlrl-deepltl-point-parallel-checkpoint-semantic-summary/1",
            "parallel": {
                "workers": args.workers,
                "formula_shard_size": args.formula_shard_size,
                "shard_dir": str(args.shard_dir),
                "missing_record_count": len(missing),
                "duplicate_key_count": len(duplicate_counts),
            },
            "records_path": str(args.records_output),
            "records_sha256": sha256(args.records_output),
        }
    )
    summary_payload = (
        json.dumps(json_ready(summary), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    atomic_write(args.summary_output, summary_payload)
    atomic_write(
        args.summary_output.with_suffix(args.summary_output.suffix + ".sha256"),
        f"{hashlib.sha256(summary_payload).hexdigest()}  {args.summary_output.name}\n".encode(
            "ascii"
        ),
    )
    return {
        "record_count": len(records),
        "missing_record_count": len(missing),
        "duplicate_key_count": len(duplicate_counts),
        "records": str(args.records_output),
        "summary": str(args.summary_output),
        "satisfied_rate": summary["aggregate"]["satisfied_rate"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--training-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--checkpoint-steps", default="auto")
    parser.add_argument("--episodes-per-formula", type=int, default=1)
    parser.add_argument("--max-formulas", type=int, default=None)
    parser.add_argument("--eval-seed-base", type=int, default=830_000)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verification-mode", choices=("fast", "full"), default="fast")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--formula-shard-size", type=int, default=5)
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR)
    parser.add_argument("--records-output", type=Path, default=DEFAULT_RECORDS_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.formula_shard_size < 1:
        raise ValueError("formula-shard-size must be positive")
    if args.episodes_per_formula < 1:
        raise ValueError("episodes-per-formula must be positive")
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    args.shard_dir = resolve(args.shard_dir)
    args.records_output = resolve(args.records_output)
    args.summary_output = resolve(args.summary_output)

    checkpoints = parse_checkpoint_steps(
        args.checkpoint_steps,
        experiment=args.experiment,
        seed=args.training_seed,
    )
    full_task_count = len(load_tasks())
    task_count = full_task_count if args.max_formulas is None else args.max_formulas
    if not 1 <= task_count <= full_task_count:
        raise ValueError(f"max-formulas must be in 1..{full_task_count}")
    expected = expected_keys(
        checkpoints=checkpoints,
        task_count=task_count,
        episodes_per_formula=args.episodes_per_formula,
        eval_seed_base=args.eval_seed_base,
        deterministic=args.deterministic,
        experiment=args.experiment,
        seed=args.training_seed,
    )

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["RABINIZER_JAR"] = str((ROOT / "rabinizer-4" / "lib" / "rabinizer.jar").resolve())
    args.shard_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    all_jobs = [
        (checkpoint, start, end)
        for checkpoint in checkpoints
        for start, end in formula_slices(task_count, args.formula_shard_size)
    ]
    pending = list(all_jobs)
    active: list[subprocess.Popen[bytes]] = []
    completed = 0
    while pending or active:
        while pending and len(active) < args.workers:
            checkpoint, start, end = pending.pop(0)
            records_path, _, _, _ = shard_paths(args.shard_dir, checkpoint, start, end)
            if shard_complete(
                records_path,
                checkpoint=checkpoint,
                start=start,
                end=end,
                episodes_per_formula=args.episodes_per_formula,
            ):
                completed += 1
                continue
            print(
                json.dumps(
                    {
                        "status": "launching",
                        "checkpoint": checkpoint,
                        "formula_start": start,
                        "formula_end": end,
                        "active": len(active) + 1,
                        "completed_shards": completed,
                        "total_shards": len(all_jobs),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            active.append(launch_shard(checkpoint=checkpoint, start=start, end=end, args=args, env=env))
        time.sleep(1.0)
        still_active = []
        for process in active:
            code = process.poll()
            if code is None:
                still_active.append(process)
                continue
            close_process_logs(process)
            checkpoint, start, end, records_path, _ = process._tlrl_shard  # type: ignore[attr-defined]
            if code != 0:
                raise RuntimeError(
                    f"shard failed: checkpoint={checkpoint} formulas={start}:{end} "
                    f"exit={code}; see {records_path.with_suffix('.stderr.log')}"
                )
            if not shard_complete(
                records_path,
                checkpoint=checkpoint,
                start=start,
                end=end,
                episodes_per_formula=args.episodes_per_formula,
            ):
                raise RuntimeError(
                    f"shard finished but is incomplete: checkpoint={checkpoint} formulas={start}:{end}"
                )
            completed += 1
            print(
                json.dumps(
                    {
                        "status": "shard_completed",
                        "checkpoint": checkpoint,
                        "formula_start": start,
                        "formula_end": end,
                        "completed_shards": completed,
                        "total_shards": len(all_jobs),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        active = still_active

    result = write_merged_outputs(
        args=args,
        checkpoints=checkpoints,
        task_count=task_count,
        expected=expected,
        started=started,
    )
    result["status"] = "completed"
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
