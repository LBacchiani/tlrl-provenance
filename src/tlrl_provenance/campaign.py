"""Benchmark-independent orchestration for confirmatory semantic audits.

The campaign layer deliberately knows nothing about environments, policies, or
temporal-logic syntax.  A benchmark adapter supplies frozen subjects and tasks
and evaluates one episode.  This module freezes the Cartesian evaluation plan,
validates resumable JSONL records, and produces outcome summaries only after
the plan is complete.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import statistics
import tempfile
from typing import Any, Mapping, Protocol, Sequence

from .provenance import ENGINE_VERSION, SEMANTICS_VERSION, engine_source_digest
from .statistics import exact_proportion


CAMPAIGN_SCHEMA = "tlrl-confirmatory-semantic-campaign/1"
PLAN_SCHEMA = "tlrl-confirmatory-semantic-plan/1"
RECORD_SCHEMA = "tlrl-confirmatory-semantic-record/1"
SUMMARY_SCHEMA = "tlrl-confirmatory-semantic-summary/1"


@dataclass(frozen=True, slots=True)
class AuditSubject:
    subject_id: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class AuditTask:
    task_id: str
    formula: str
    family: str | None = None
    metadata: Mapping[str, Any] | None = None


class AuditSession(Protocol):
    def run_episode(
        self,
        *,
        evaluation_seed: int,
        deterministic: bool,
        verification_mode: str,
        scenario_seed: int | None = None,
    ) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


class CampaignAdapter(Protocol):
    def subjects(self) -> Sequence[AuditSubject]: ...
    def tasks(self) -> Sequence[AuditTask]: ...
    def provenance(self) -> Mapping[str, Any]: ...
    def open_session(self, subject: AuditSubject, task: AuditTask) -> AuditSession: ...


def canonical_json(value: Any) -> bytes:
    return (json.dumps(json_ready(value), sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(json_ready(item) for item in value)
    return str(value)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
    temporary.replace(path)


def load_adapter(specification: str, config: Mapping[str, Any], root: Path) -> CampaignAdapter:
    """Load ``module:factory``.  The factory receives adapter config and v3 root."""

    if specification.count(":") != 1:
        raise ValueError("adapter must use 'module:factory' syntax")
    module_name, factory_name = specification.split(":")
    factory = getattr(importlib.import_module(module_name), factory_name)
    return factory(config=dict(config), root=root)


def load_campaign_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != CAMPAIGN_SCHEMA:
        raise ValueError(f"unsupported campaign schema: {config.get('schema_version')!r}")
    required = {"campaign_id", "benchmark_id", "adapter", "design", "seed_namespace", "outputs"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"campaign config is missing: {missing}")
    design = config["design"]
    if int(design.get("episodes_per_task", 0)) < 1:
        raise ValueError("design.episodes_per_task must be positive")
    if design.get("verification_mode") not in {"fast", "full"}:
        raise ValueError("design.verification_mode must be 'fast' or 'full'")
    if int(design.get("task_shard_size", 0)) < 1 or int(design.get("workers", 0)) < 1:
        raise ValueError("design.task_shard_size and design.workers must be positive")
    return config


def _unique_identifier(items: Sequence[Any], attribute: str, kind: str) -> None:
    identifiers = [getattr(item, attribute) for item in items]
    if any(not isinstance(item, str) or not item for item in identifiers):
        raise ValueError(f"every {kind} must have a nonempty string {attribute}")
    duplicates = [item for item, count in Counter(identifiers).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate {kind} identifiers: {sorted(duplicates)}")


def build_plan(
    config: Mapping[str, Any],
    adapter: CampaignAdapter,
    *,
    runner_path: Path | None = None,
) -> dict[str, Any]:
    subjects = tuple(adapter.subjects())
    tasks = tuple(adapter.tasks())
    if not subjects or not tasks:
        raise ValueError("a campaign requires at least one subject and one task")
    _unique_identifier(subjects, "subject_id", "subject")
    _unique_identifier(tasks, "task_id", "task")

    seed_config = config["seed_namespace"]
    base = int(seed_config["base"])
    subject_stride = int(seed_config["subject_stride"])
    task_stride = int(seed_config["task_stride"])
    episode_stride = int(seed_config.get("episode_stride", 1))
    episodes = int(config["design"]["episodes_per_task"])
    if min(base, subject_stride, task_stride, episode_stride) < 0 or episode_stride == 0:
        raise ValueError("seed namespace values must be nonnegative and episode_stride positive")

    # Optional, opt-in, backward-compatible: a scenario_namespace has no
    # subject term, so every subject sees the same scenario_seed for a given
    # (task, episode_index) cell -- required whenever the environment itself
    # (not just the policy) is randomized per reset, so that cross-policy
    # comparisons for "the same formula, the same episode index" are not
    # confounded by different subjects drawing different random scenarios.
    # Absent entirely when a campaign config has no scenario_namespace: no
    # cell gets a scenario_seed key and existing adapters are unaffected.
    scenario_config = config.get("scenario_namespace")
    scenario_base = scenario_stride_task = scenario_stride_episode = None
    if scenario_config is not None:
        scenario_base = int(scenario_config["base"])
        scenario_stride_task = int(scenario_config["task_stride"])
        scenario_stride_episode = int(scenario_config.get("episode_stride", 1))
        if min(scenario_base, scenario_stride_task, scenario_stride_episode) < 0 or scenario_stride_episode == 0:
            raise ValueError("scenario namespace values must be nonnegative and episode_stride positive")

    cells: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    seen_scenarios: dict[int, tuple[str, int]] = {}
    for subject_index, subject in enumerate(subjects):
        for task_index, task in enumerate(tasks):
            for episode_index in range(episodes):
                seed = (
                    base
                    + subject_index * subject_stride
                    + task_index * task_stride
                    + episode_index * episode_stride
                )
                if seed in seen_seeds:
                    raise ValueError("seed namespace aliases two evaluation cells")
                seen_seeds.add(seed)
                cell = {
                    "subject_id": subject.subject_id,
                    "task_id": task.task_id,
                    "episode_index": episode_index,
                    "evaluation_seed": seed,
                }
                if scenario_config is not None:
                    scenario_seed = (
                        scenario_base
                        + task_index * scenario_stride_task
                        + episode_index * scenario_stride_episode
                    )
                    scenario_key = (task.task_id, episode_index)
                    previous = seen_scenarios.get(scenario_seed)
                    if previous is None:
                        seen_scenarios[scenario_seed] = scenario_key
                    elif previous != scenario_key:
                        raise ValueError(
                            "scenario namespace aliases two task/episode cells: "
                            f"{previous} and {scenario_key}"
                        )
                    cell["scenario_seed"] = scenario_seed
                cells.append(cell)
    forbidden_values = {int(item) for item in seed_config.get("forbidden_values", [])}
    forbidden_ranges = [tuple(map(int, item)) for item in seed_config.get("forbidden_ranges", [])]
    for lower, upper in forbidden_ranges:
        if lower > upper:
            raise ValueError("forbidden seed range lower bound exceeds upper bound")
    collisions = sorted(
        seed
        for seed in seen_seeds
        if seed in forbidden_values or any(lower <= seed <= upper for lower, upper in forbidden_ranges)
    )
    if collisions:
        raise ValueError(f"evaluation seed namespace collides with forbidden seeds: {collisions[:8]}")

    adapter_path_source = inspect.getsourcefile(type(adapter))
    adapter_path = Path(adapter_path_source).resolve() if adapter_path_source else None
    plan = {
        "schema_version": PLAN_SCHEMA,
        "campaign_id": config["campaign_id"],
        "benchmark_id": config["benchmark_id"],
        "config_sha256": sha256_bytes(canonical_json(config)),
        "engine": {
            "version": ENGINE_VERSION,
            "semantics_version": SEMANTICS_VERSION,
            "source_sha256": engine_source_digest(),
        },
        "adapter": {
            "specification": config["adapter"],
            "source_path": str(adapter_path) if adapter_path else None,
            "source_sha256": sha256_file(adapter_path) if adapter_path and adapter_path.is_file() else None,
            "provenance": json_ready(adapter.provenance()),
        },
        "runner": {
            "source_path": str(runner_path.resolve()) if runner_path else None,
            "source_sha256": sha256_file(runner_path.resolve()) if runner_path and runner_path.is_file() else None,
        },
        "design": json_ready(config["design"]),
        "seed_namespace": json_ready(seed_config),
        "scenario_namespace": json_ready(scenario_config) if scenario_config is not None else None,
        "subjects": [asdict(item) for item in subjects],
        "tasks": [asdict(item) for item in tasks],
        "cells": cells,
        "expected": {
            "subject_count": len(subjects),
            "task_count": len(tasks),
            "episodes_per_task": episodes,
            "record_count": len(cells),
        },
    }
    plan["plan_sha256"] = sha256_bytes(canonical_json(plan))
    return plan


def freeze_plan(path: Path, plan: Mapping[str, Any]) -> None:
    payload = canonical_json(plan)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"frozen plan differs from current inventory/config: {path}")
        return
    atomic_write(path, payload)
    atomic_write(path.with_suffix(path.suffix + ".sha256"), f"{sha256_bytes(payload)}  {path.name}\n".encode("ascii"))


def record_key(record: Mapping[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(record["subject_id"]),
        str(record["task_id"]),
        int(record["episode_index"]),
        int(record["evaluation_seed"]),
    )


def expected_keys(plan: Mapping[str, Any]) -> set[tuple[str, str, int, int]]:
    return {record_key(cell) for cell in plan["cells"]}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def validate_record(record: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "campaign_id", "benchmark_id", "plan_sha256", "subject_id",
        "task_id", "episode_index", "evaluation_seed", "steps", "terminated", "truncated",
        "official_success", "official_violation", "independent_verdict", "disposition",
        "return_sum", "possible_evidence_count", "necessary_evidence_count", "atom_counts",
        "failure_class", "official_independent_mismatch",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"record is missing required fields: {missing}")
    if record["schema_version"] != RECORD_SCHEMA:
        raise ValueError("wrong record schema")
    if record["campaign_id"] != plan["campaign_id"] or record["benchmark_id"] != plan["benchmark_id"]:
        raise ValueError("record campaign identity mismatch")
    if record["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("record plan digest mismatch")
    if int(record["steps"]) < 1:
        raise ValueError("record steps must be positive")
    if int(record["possible_evidence_count"]) < int(record["necessary_evidence_count"]):
        raise ValueError("necessary evidence cannot exceed possible evidence")
    if bool(record["official_independent_mismatch"]):
        raise ValueError("official/independent semantic mismatch (fail closed)")


def task_shards(plan: Mapping[str, Any], width: int) -> list[dict[str, Any]]:
    task_ids = [item["task_id"] for item in plan["tasks"]]
    shards = []
    for subject in plan["subjects"]:
        for start in range(0, len(task_ids), width):
            selected = task_ids[start : start + width]
            shards.append(
                {
                    "shard_id": f"{subject['subject_id']}__t{start:04d}_{start + len(selected):04d}",
                    "subject_id": subject["subject_id"],
                    "task_ids": selected,
                }
            )
    return shards


def summarize_group(records: Sequence[Mapping[str, Any]], confidence: float) -> dict[str, Any]:
    trials = len(records)
    if not trials:
        raise ValueError("cannot summarize an empty record group")
    satisfied = sum(item["independent_verdict"] == "satisfied" for item in records)
    estimate = exact_proportion(satisfied, trials, confidence=confidence)
    failure_counts = Counter(str(item["failure_class"]) for item in records)
    disposition_counts = Counter(str(item["disposition"]) for item in records)
    possible = [int(item["possible_evidence_count"]) for item in records]
    necessary = [int(item["necessary_evidence_count"]) for item in records]
    steps = [int(item["steps"]) for item in records]
    satisfied_records = [item for item in records if item["independent_verdict"] == "satisfied"]
    zero_necessary = sum(int(item["necessary_evidence_count"]) == 0 for item in satisfied_records)
    satisfied_possible = [int(item["possible_evidence_count"]) for item in satisfied_records]
    satisfied_necessary = [int(item["necessary_evidence_count"]) for item in satisfied_records]
    satisfied_steps = [int(item["steps"]) for item in satisfied_records]
    return {
        "episodes": trials,
        "satisfaction": asdict(estimate),
        "dispositions": dict(sorted(disposition_counts.items())),
        "failure_classes": dict(sorted(failure_counts.items())),
        "mean_steps": math.fsum(steps) / trials,
        "mean_possible_evidence": math.fsum(possible) / trials,
        "mean_necessary_evidence": math.fsum(necessary) / trials,
        "mean_possible_evidence_per_step": math.fsum(p / s for p, s in zip(possible, steps)) / trials,
        "mean_necessary_evidence_per_step": math.fsum(n / s for n, s in zip(necessary, steps)) / trials,
        "satisfied_semantic_evidence": {
            "possible_count": numeric_summary(satisfied_possible),
            "necessary_count": numeric_summary(satisfied_necessary),
            "possible_per_step": numeric_summary(
                [value / step for value, step in zip(satisfied_possible, satisfied_steps)]
            ),
            "necessary_per_step": numeric_summary(
                [value / step for value, step in zip(satisfied_necessary, satisfied_steps)]
            ),
        },
        "zero_necessary_satisfied": {
            "count": zero_necessary,
            "satisfied_episodes": len(satisfied_records),
            "rate": zero_necessary / len(satisfied_records) if satisfied_records else None,
        },
    }


def numeric_summary(values: Sequence[float | int]) -> dict[str, Any]:
    clean = sorted(float(value) for value in values)
    if not clean:
        return {"count": 0, "min": None, "q25": None, "median": None, "mean": None, "q75": None, "max": None}

    def percentile(quantile: float) -> float:
        position = quantile * (len(clean) - 1)
        lower = int(position)
        upper = min(lower + 1, len(clean) - 1)
        weight = position - lower
        return clean[lower] * (1.0 - weight) + clean[upper] * weight

    return {
        "count": len(clean), "min": clean[0], "q25": percentile(0.25),
        "median": statistics.median(clean), "mean": math.fsum(clean) / len(clean),
        "q75": percentile(0.75), "max": clean[-1],
    }


def build_summary(
    records: Sequence[Mapping[str, Any]], plan: Mapping[str, Any], *, confidence: float = 0.95
) -> dict[str, Any]:
    expected = expected_keys(plan)
    actual: dict[tuple[str, str, int, int], Mapping[str, Any]] = {}
    duplicates: list[tuple[str, str, int, int]] = []
    for record in records:
        validate_record(record, plan)
        key = record_key(record)
        if key not in expected:
            raise ValueError(f"unexpected record key: {key}")
        if key in actual:
            duplicates.append(key)
        actual[key] = record
    missing = expected - set(actual)
    if duplicates or missing or len(records) != len(expected):
        raise RuntimeError(
            f"campaign incomplete: missing={len(missing)} duplicate={len(duplicates)} "
            f"records={len(records)} expected={len(expected)}"
        )

    ordered = list(actual.values())
    by_subject: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_subject_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    task_family = {item["task_id"]: item.get("family") or "unclassified" for item in plan["tasks"]}
    for record in ordered:
        by_subject[str(record["subject_id"])].append(record)
        by_task[str(record["task_id"])].append(record)
        by_subject_task[f"{record['subject_id']}::{record['task_id']}"] .append(record)
        by_family[task_family[str(record["task_id"])]].append(record)

    fixed_formula_policy_contrasts: dict[str, Any] = {}
    for task_id in sorted(by_task):
        subject_groups = {
            subject_id: by_subject_task[f"{subject_id}::{task_id}"]
            for subject_id in sorted(by_subject)
        }
        subject_summaries = {
            subject_id: summarize_group(group, confidence)
            for subject_id, group in subject_groups.items()
        }
        sat_rates = [item["satisfaction"]["value"] for item in subject_summaries.values()]
        evidence_means = [
            item["satisfied_semantic_evidence"]["necessary_per_step"]["mean"]
            for item in subject_summaries.values()
            if item["satisfied_semantic_evidence"]["necessary_per_step"]["mean"] is not None
        ]
        fixed_formula_policy_contrasts[task_id] = {
            "subjects": subject_summaries,
            "satisfaction_rate_range": max(sat_rates) - min(sat_rates),
            "satisfied_necessary_per_step_mean_range": (
                max(evidence_means) - min(evidence_means) if len(evidence_means) >= 2 else None
            ),
            "interpretation": "descriptive_between_policy_contrast_with_formula_held_fixed",
        }

    return {
        "schema_version": SUMMARY_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "benchmark_id": plan["benchmark_id"],
        "plan_sha256": plan["plan_sha256"],
        "completeness": {
            "expected_records": len(expected), "observed_records": len(records),
            "missing_records": 0, "duplicate_records": 0,
            "official_independent_mismatches": 0, "status": "passed",
        },
        "aggregate": summarize_group(ordered, confidence),
        "by_subject": {key: summarize_group(value, confidence) for key, value in sorted(by_subject.items())},
        "by_task": {key: summarize_group(value, confidence) for key, value in sorted(by_task.items())},
        "by_subject_task": {
            key: summarize_group(value, confidence) for key, value in sorted(by_subject_task.items())
        },
        "by_formula_family_descriptive_only": {
            key: summarize_group(value, confidence) for key, value in sorted(by_family.items())
        },
        "fixed_formula_policy_contrasts": fixed_formula_policy_contrasts,
        "interpretation_guardrails": plan.get("design", {}).get("interpretation_guardrails", {}),
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    aggregate = summary["aggregate"]
    satisfaction = aggregate["satisfaction"]
    lines = [
        f"# Confirmatory semantic audit: {summary['benchmark_id']}", "",
        "## Integrity", "",
        f"- Complete: **{summary['completeness']['status']}** "
        f"({summary['completeness']['observed_records']}/{summary['completeness']['expected_records']} records).",
        "- Official/independent mismatches: **0**.", "",
        "## What the ordinary success rate hides", "",
        f"Overall `P_sat` is **{satisfaction['value']:.3f}** "
        f"({satisfaction['successes']}/{satisfaction['trials']}; "
        f"{int(satisfaction['confidence'] * 100)}% exact CI "
        f"[{satisfaction['lower']:.3f}, {satisfaction['upper']:.3f}]).",
        "", "Failure/disposition counts:", "",
    ]
    for key, value in aggregate["failure_classes"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Per-policy view", "", "| Subject | P_sat | Satisfied / episodes | Mean necessary evidence/step |", "|---|---:|---:|---:|"])
    for subject, group in summary["by_subject"].items():
        sat = group["satisfaction"]
        lines.append(
            f"| {subject} | {sat['value']:.3f} | {sat['successes']} / {sat['trials']} | "
            f"{group['mean_necessary_evidence_per_step']:.3f} |"
        )
    ranked = sorted(
        summary["fixed_formula_policy_contrasts"].items(),
        key=lambda item: item[1]["satisfaction_rate_range"],
        reverse=True,
    )[:10]
    lines.extend(["", "## Formulas with the largest between-policy P_sat spread", "",
                  "These are descriptive screening results with the formula held fixed.", "",
                  "| Formula ID | P_sat range across policies | Necessary-evidence/step mean range |",
                  "|---|---:|---:|"])
    for task_id, contrast in ranked:
        evidence_range = contrast["satisfied_necessary_per_step_mean_range"]
        evidence_text = "n/a" if evidence_range is None else f"{evidence_range:.3f}"
        lines.append(f"| {task_id} | {contrast['satisfaction_rate_range']:.3f} | {evidence_text} |")
    lines.extend([
        "", "## Interpretation boundary", "",
        "The audit enriches `P_sat` with trace-level semantic evidence and failure diagnosis. "
        "Formula-family aggregates are descriptive: policy comparisons must hold the formula fixed, "
        "and structural evidence burden must not be relabeled as policy quality.", "",
    ])
    return "\n".join(lines)
