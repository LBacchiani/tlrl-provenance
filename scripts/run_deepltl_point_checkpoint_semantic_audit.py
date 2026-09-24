#!/usr/bin/env python
"""Run formula-conditioned semantic audits across DeepLTL PPO checkpoints.

This is a development-analysis driver: it does not train, tune, or select
benchmarks.  It reuses frozen DeepLTL PointWorld checkpoints and records one
auditable JSONL row per checkpoint/formula/episode so interrupted runs can be
resumed without losing completed evaluations.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
UPSTREAM = ROOT / ".runtime" / "deep-ltl"
DEVELOPMENT_EVIDENCE = ROOT / "evidence" / "deepltl_point" / "development" / "semantic"
DEFAULT_RECORDS_OUTPUT = DEVELOPMENT_EVIDENCE / "deepltl_point_checkpoint_semantic_records.jsonl"
DEFAULT_SUMMARY_OUTPUT = DEVELOPMENT_EVIDENCE / "deepltl_point_checkpoint_semantic_summary.json"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_deepltl_point_preliminary_semantic_audit import (  # noqa: E402
    DEFAULT_EXPERIMENT,
    DEFAULT_SEED,
    EvaluationLimits,
    atomic_write,
    checkpoint_path,
    json_ready,
    load_agent,
    run_episode,
    sha256,
)
from tlrl_benchmarks.deepltl_point.adapter import load_tasks  # noqa: E402


def formula_family(formula: str) -> str:
    source = formula.strip()
    if source.startswith("F "):
        return "eventually_chain"
    if " U " in source:
        return "until_chain"
    return "other"


def resolve_output(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def available_eval_checkpoints(experiment: str, seed: int) -> list[int]:
    eval_dir = (
        UPSTREAM
        / "experiments"
        / "ppo"
        / "PointLtl2-v0"
        / experiment
        / str(seed)
        / "eval"
    )
    if not eval_dir.is_dir():
        raise FileNotFoundError(eval_dir)
    steps: list[int] = []
    for path in eval_dir.glob("*.pth"):
        try:
            steps.append(int(path.stem))
        except ValueError:
            continue
    if not steps:
        raise RuntimeError(f"no numeric eval checkpoints found in {eval_dir}")
    return sorted(set(steps))


def parse_checkpoint_steps(raw: str, *, experiment: str, seed: int) -> list[int]:
    value = raw.strip().lower()
    available = available_eval_checkpoints(experiment, seed)
    if value == "auto":
        return available
    if value == "latest":
        return [available[-1]]
    requested = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    missing = [step for step in requested if step not in available]
    if missing:
        raise ValueError(f"requested checkpoint(s) not found: {missing}; available={available}")
    return requested


def record_key(record: dict[str, Any]) -> tuple[object, ...]:
    return (
        record["checkpoint_steps"],
        record["task_id"],
        record["episode_index"],
        record["seed"],
        record["deterministic"],
    )


def expected_seed(
    *,
    eval_seed_base: int,
    checkpoint_rank: int,
    task_index: int,
    episode_index: int,
) -> int:
    """Return a batching-stable evaluation seed.

    ``checkpoint_rank`` is the checkpoint's rank in the complete eval-checkpoint
    directory, not its rank in the user-selected subset.  This makes a full
    one-process sweep and a fresh-process-per-checkpoint sweep produce identical
    keys and therefore resume each other safely.
    """

    return eval_seed_base + checkpoint_rank * 100_000 + task_index * 100 + episode_index


def load_existing_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
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


def append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_ready(record), sort_keys=True) + "\n")
        handle.flush()


def numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values]
    if not clean:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(clean),
        "min": min(clean),
        "median": statistics.median(clean),
        "mean": statistics.fmean(clean),
        "max": max(clean),
    }


def summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    dispositions = Counter(record["disposition"] for record in records)
    verdicts = Counter(record["independent_verdict"] for record in records)
    satisfied = [record for record in records if record["independent_verdict"] == "satisfied"]
    mismatches = [
        record
        for record in records
        if bool(record["official_success"]) != (record["independent_verdict"] == "satisfied")
    ]
    failure_classes = Counter()
    for record in records:
        if record["independent_verdict"] == "satisfied":
            continue
        if record["disposition"] == "decided_violation":
            failure_classes["forbidden_or_temporal_decided_violation"] += 1
        elif record["disposition"] == "environment_termination" and not record["atom_counts"]:
            failure_classes["termination_without_relevant_progress"] += 1
        elif record["disposition"] == "environment_termination":
            failure_classes["termination_after_relevant_progress"] += 1
        elif record["disposition"] == "censored":
            failure_classes["censored"] += 1
        else:
            failure_classes["other_unresolved"] += 1

    necessary_ratios = [
        record["mean_necessary_evidence"] / record["steps"]
        for record in satisfied
        if record["steps"] > 0
    ]
    possible_ratios = [
        record["mean_possible_evidence"] / record["steps"]
        for record in satisfied
        if record["steps"] > 0
    ]
    zero_necessary_satisfied = sum(
        1 for record in satisfied if record["mean_necessary_evidence"] == 0
    )
    return {
        "episodes": n,
        "dispositions": dict(sorted(dispositions.items())),
        "independent_verdicts": dict(sorted(verdicts.items())),
        "satisfied_rate": verdicts["satisfied"] / n if n else None,
        "censored_rate": dispositions["censored"] / n if n else None,
        "official_independent_mismatch_count": len(mismatches),
        "failure_classes": dict(sorted(failure_classes.items())),
        "mean_steps": statistics.fmean(record["steps"] for record in records) if records else None,
        "mean_possible_evidence": (
            statistics.fmean(record["mean_possible_evidence"] for record in records)
            if records
            else None
        ),
        "mean_necessary_evidence": (
            statistics.fmean(record["mean_necessary_evidence"] for record in records)
            if records
            else None
        ),
        "satisfied_necessary_per_step": numeric_summary(necessary_ratios),
        "satisfied_possible_per_step": numeric_summary(possible_ratios),
        "satisfied_zero_necessary_count": zero_necessary_satisfied,
        "satisfied_zero_necessary_rate": (
            zero_necessary_satisfied / len(satisfied) if satisfied else None
        ),
    }


def summarize_records(
    *,
    records: list[dict[str, Any]],
    checkpoints: list[int],
    task_count: int,
    episodes_per_formula: int,
    started: float,
) -> dict[str, Any]:
    by_checkpoint: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_family_checkpoint: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    by_formula: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_formula_checkpoint: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        checkpoint = int(record["checkpoint_steps"])
        family = record["formula_family"]
        by_checkpoint[checkpoint].append(record)
        by_family_checkpoint[(family, checkpoint)].append(record)
        by_formula[record["task_id"]].append(record)
        by_formula_checkpoint[(record["task_id"], checkpoint)].append(record)

    formula_summaries = {}
    for task_id, task_records in sorted(by_formula.items()):
        formula_summaries[task_id] = {
            "formula": task_records[0]["formula"],
            "formula_family": task_records[0]["formula_family"],
            "overall": summarize_group(task_records),
            "by_checkpoint": {
                str(checkpoint): summarize_group(by_formula_checkpoint[(task_id, checkpoint)])
                for checkpoint in checkpoints
                if (task_id, checkpoint) in by_formula_checkpoint
            },
        }

    family_by_checkpoint: dict[str, dict[str, Any]] = defaultdict(dict)
    for (family, checkpoint), group in sorted(by_family_checkpoint.items()):
        family_by_checkpoint[family][str(checkpoint)] = summarize_group(group)

    failures = [
        {
            "checkpoint_steps": record["checkpoint_steps"],
            "task_id": record["task_id"],
            "formula_family": record["formula_family"],
            "formula": record["formula"],
            "seed": record["seed"],
            "steps": record["steps"],
            "disposition": record["disposition"],
            "atom_counts": record["atom_counts"],
            "mean_possible_evidence": record["mean_possible_evidence"],
            "mean_necessary_evidence": record["mean_necessary_evidence"],
        }
        for record in records
        if record["independent_verdict"] != "satisfied"
    ]
    return {
        "schema_version": "tlrl-deepltl-point-checkpoint-semantic-summary/1",
        "status": "completed",
        "scope": "development_only_formula_conditioned_checkpoint_analysis",
        "benchmark_id": "deepltl-pointworld-official-eval-50",
        "checkpoint_steps": checkpoints,
        "task_count": task_count,
        "episodes_per_formula": episodes_per_formula,
        "record_count": len(records),
        "elapsed_seconds_this_invocation": time.time() - started,
        "aggregate": summarize_group(records),
        "by_checkpoint": {
            str(checkpoint): summarize_group(by_checkpoint[checkpoint])
            for checkpoint in checkpoints
            if checkpoint in by_checkpoint
        },
        "by_formula_family_checkpoint": dict(sorted(family_by_checkpoint.items())),
        "by_formula": formula_summaries,
        "failures": failures,
        "interpretation_guardrails": {
            "cross_formula_necessary_evidence_is_descriptive_not_policy_quality": True,
            "formula_family_effects_must_not_be_read_as_controller_robustness": True,
            "policy_learning_signal_requires_holding_formula_fixed_across_checkpoints_or_seeds": True,
            "eventually_chain_zero_necessity_can_be_formula_predictable_under_repeated_witnesses": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--training-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--checkpoint-steps",
        default="auto",
        help="auto, latest, or a comma-separated list such as 0,1048576,4980736",
    )
    parser.add_argument("--episodes-per-formula", type=int, default=1)
    parser.add_argument("--max-formulas", type=int, default=None)
    parser.add_argument(
        "--formula-start",
        type=int,
        default=0,
        help="zero-based inclusive formula index for sharded runs",
    )
    parser.add_argument(
        "--formula-end",
        type=int,
        default=None,
        help="zero-based exclusive formula index for sharded runs",
    )
    parser.add_argument("--eval-seed-base", type=int, default=830_000)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--verification-mode",
        choices=("fast", "full"),
        default="fast",
        help=(
            "fast skips expensive support enumeration and semantic-event decision "
            "collection; full matches the small-audit structural verifier"
        ),
    )
    parser.add_argument("--records-output", type=Path, default=DEFAULT_RECORDS_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.episodes_per_formula < 1:
        raise ValueError("episodes-per-formula must be positive")
    if args.max_formulas is not None and args.max_formulas < 1:
        raise ValueError("max-formulas must be positive when provided")
    if args.formula_start < 0:
        raise ValueError("formula-start must be non-negative")
    if args.formula_end is not None and args.formula_end <= args.formula_start:
        raise ValueError("formula-end must be greater than formula-start")

    args.records_output = resolve_output(args.records_output)
    args.summary_output = resolve_output(args.summary_output)
    fast_mode = args.verification_mode == "fast"
    evaluation_limits = EvaluationLimits(collect_decisions=not fast_mode)
    verification_kwargs: dict[str, Any] | None = None
    if fast_mode:
        verification_kwargs = None
    checkpoints = parse_checkpoint_steps(
        args.checkpoint_steps,
        experiment=args.experiment,
        seed=args.training_seed,
    )
    checkpoint_ranks = {
        step: index
        for index, step in enumerate(
            available_eval_checkpoints(args.experiment, args.training_seed),
            start=1,
        )
    }
    all_tasks = list(load_tasks())
    tasks = all_tasks[args.formula_start : args.formula_end]
    if args.max_formulas is not None:
        tasks = tasks[: args.max_formulas]
    if not tasks:
        raise ValueError("formula slice selected no tasks")
    task_by_id = {task.task_id: task for task in tasks}
    task_global_indices = {
        task.task_id: args.formula_start + offset + 1 for offset, task in enumerate(tasks)
    }
    expected_keys = {
        (
            checkpoint,
            task.task_id,
            episode_index,
            expected_seed(
                eval_seed_base=args.eval_seed_base,
                checkpoint_rank=checkpoint_ranks[checkpoint],
                task_index=task_global_indices[task.task_id],
                episode_index=episode_index,
            ),
            args.deterministic,
        )
        for checkpoint in checkpoints
        for task in tasks
        for episode_index in range(args.episodes_per_formula)
    }

    existing_records = load_existing_records(args.records_output) if args.resume else []
    usable_existing_records = [
        record for record in existing_records if record_key(record) in expected_keys
    ]
    completed = {record_key(record) for record in usable_existing_records}
    all_records = list(usable_existing_records)
    started = time.time()

    os.chdir(UPSTREAM)
    total = len(checkpoints) * len(tasks) * args.episodes_per_formula
    completed_this_run = 0
    for checkpoint_index, checkpoint_steps in enumerate(checkpoints, start=1):
        checkpoint = checkpoint_path(args.experiment, args.training_seed, checkpoint_steps)
        checkpoint_digest = sha256(checkpoint)
        for local_task_index, task in enumerate(tasks, start=1):
            env, agent = load_agent(
                args.experiment,
                args.training_seed,
                checkpoint,
                task.source,
            )
            try:
                for episode_index in range(args.episodes_per_formula):
                    seed = (
                        expected_seed(
                            eval_seed_base=args.eval_seed_base,
                            checkpoint_rank=checkpoint_ranks[checkpoint_steps],
                            task_index=task_global_indices[task.task_id],
                            episode_index=episode_index,
                        )
                    )
                    key = (
                        checkpoint_steps,
                        task.task_id,
                        episode_index,
                        seed,
                        args.deterministic,
                    )
                    if key in completed:
                        continue
                    print(
                        "["
                        f"ckpt {checkpoint_index}/{len(checkpoints)} "
                        f"formula {local_task_index}/{len(tasks)} "
                        f"episode {episode_index + 1}/{args.episodes_per_formula}"
                        f"] {checkpoint_steps} {task.task_id}: {task.source}",
                        flush=True,
                    )
                    record = run_episode(
                        env=env,
                        agent=agent,
                        task_id=task.task_id,
                        formula=task.source,
                        task_by_id=task_by_id,
                        policy_id=f"{args.experiment}:{args.training_seed}:{checkpoint_steps}",
                        seed=seed,
                        deterministic=args.deterministic,
                        evaluation_limits=evaluation_limits,
                        verify_structural_certificate=not fast_mode,
                        verification_kwargs=verification_kwargs,
                        collect_semantic_events=not fast_mode,
                        compute_prefix=not fast_mode,
                    )
                    record.update(
                        {
                            "schema_version": "tlrl-deepltl-point-checkpoint-semantic-record/1",
                            "experiment": args.experiment,
                            "training_seed": args.training_seed,
                            "checkpoint_steps": checkpoint_steps,
                            "checkpoint_sha256": checkpoint_digest,
                            "checkpoint_path": str(checkpoint.relative_to(ROOT)),
                            "episode_index": episode_index,
                            "deterministic": args.deterministic,
                            "formula_family": formula_family(task.source),
                        }
                    )
                    append_record(args.records_output, record)
                    all_records.append(record)
                    completed.add(key)
                    completed_this_run += 1
                    print(
                        json.dumps(
                            {
                                "status": "recorded",
                                "completed_this_run": completed_this_run,
                                "planned_total": total,
                                "checkpoint_steps": checkpoint_steps,
                                "task_id": task.task_id,
                                "verdict": record["independent_verdict"],
                                "disposition": record["disposition"],
                                "steps": record["steps"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            finally:
                env.close()

    relevant_records = [
        record
        for record in all_records
        if record_key(record) in expected_keys
    ]
    summary = summarize_records(
        records=relevant_records,
        checkpoints=checkpoints,
        task_count=len(tasks),
        episodes_per_formula=args.episodes_per_formula,
        started=started,
    )
    summary["records_path"] = str(args.records_output)
    summary["records_sha256"] = sha256(args.records_output)
    payload = (json.dumps(json_ready(summary), indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write(args.summary_output, payload)
    atomic_write(
        args.summary_output.with_suffix(args.summary_output.suffix + ".sha256"),
        f"{hashlib.sha256(payload).hexdigest()}  {args.summary_output.name}\n".encode("ascii"),
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "records": str(args.records_output),
                "summary": str(args.summary_output),
                "record_count": len(relevant_records),
                "completed_this_run": completed_this_run,
                "satisfied_rate": summary["aggregate"]["satisfied_rate"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
