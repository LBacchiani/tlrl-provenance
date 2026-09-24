"""Validate and summarize the full-prefix PointWorld semantic campaign.

This analysis is intentionally read-only with respect to the frozen plan and
records.  It validates every retained trace, checks scenario pairing, and
performs the pre-existing formula-fixed permutation analyses before writing a
separately hash-anchored JSON result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tlrl_provenance import (  # noqa: E402
    bonferroni_alpha,
    permutation_test_group_range,
    permutation_test_median_range,
)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def quantiles(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    quartiles = statistics.quantiles(ordered, n=4, method="inclusive")
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "q25": quartiles[0],
        "q75": quartiles[2],
        "min": ordered[0],
        "max": ordered[-1],
    }


def expected_labels(trace: dict[str, Any], step: int) -> list[str]:
    x, y = trace["agent_xy"][step]
    labels = []
    for zone in trace["zone_layout"]:
        radius_sq = float(zone["radius"]) ** 2
        if any((x - zx) ** 2 + (y - zy) ** 2 <= radius_sq for zx, zy in zone["centers_xy"]):
            labels.append(str(zone["color"]))
    return sorted(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    records_path = args.records.resolve()
    output_path = args.output.resolve()
    failure_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    outcomes: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    subject_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cell_layouts: dict[tuple[str, int], set[str]] = defaultdict(set)
    prefix_counts: Counter[str] = Counter()
    disposition_counts: Counter[str] = Counter()
    mismatch_count = 0
    record_count = 0
    plan_hashes: set[str] = set()

    with records_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            record_count += 1
            plan_hashes.add(record["plan_sha256"])
            mismatch_count += int(bool(record["official_independent_mismatch"]))
            prefix_counts[record["prefix_status"]] += 1
            disposition_counts[record["disposition"]] += 1

            trace = record["retained_trace"]
            if canonical_sha256(trace) != record["retained_trace_sha256"]:
                raise ValueError(f"retained trace hash mismatch on line {line_number}")
            steps = int(record["steps"])
            for field in ("agent_xy", "reported_labels", "actions", "rewards"):
                if len(trace[field]) != steps:
                    raise ValueError(f"{field} length mismatch on line {line_number}")
            for step, reported in enumerate(trace["reported_labels"]):
                if sorted(reported) != expected_labels(trace, step):
                    raise ValueError(f"geometry/label mismatch on line {line_number}, step {step}")

            expected_prefix = {
                "decided_satisfaction": "definitely_satisfied",
                "decided_violation": "definitely_violated",
                "censored": "open",
                "environment_termination": "open",
            }[record["disposition"]]
            if record["prefix_status"] != expected_prefix:
                raise ValueError(f"prefix/disposition mismatch on line {line_number}")

            layout_hash = canonical_sha256(trace["zone_layout"])
            cell_layouts[(record["task_id"], int(record["episode_index"]))].add(layout_hash)
            failure_rows[record["failure_class"]].append(record)
            subject_rows[record["subject_id"]].append(record)
            outcomes[record["task_id"]][record["subject_id"]].append(
                record["disposition"] == "decided_satisfaction"
            )

    if record_count != 5000:
        raise ValueError(f"expected 5000 records, found {record_count}")
    if mismatch_count:
        raise ValueError(f"official/independent mismatches: {mismatch_count}")
    if len(plan_hashes) != 1:
        raise ValueError(f"records contain multiple plan hashes: {sorted(plan_hashes)}")
    unpaired = [key for key, values in cell_layouts.items() if len(values) != 1]
    if unpaired:
        raise ValueError(f"zone layout is not paired in {len(unpaired)} cells; first={unpaired[0]}")
    if len(cell_layouts) != 1000:
        raise ValueError(f"expected 1000 formula/episode scenario cells, found {len(cell_layouts)}")

    by_failure_class = {}
    for failure_class, rows in sorted(failure_rows.items()):
        steps = [float(row["steps"]) for row in rows]
        necessary = [float(row["necessary_evidence_count"]) for row in rows]
        possible = [float(row["possible_evidence_count"]) for row in rows]
        by_failure_class[failure_class] = {
            "episodes": len(rows),
            "steps": quantiles(steps),
            "necessary_count": quantiles(necessary),
            "necessary_per_step": quantiles([n / s for n, s in zip(necessary, steps)]),
            "possible_count": quantiles(possible),
            "possible_per_step": quantiles([p / s for p, s in zip(possible, steps)]),
            "zero_necessary_count": sum(value == 0 for value in necessary),
            "zero_necessary_rate": sum(value == 0 for value in necessary) / len(necessary),
        }

    by_subject = {}
    for subject_id, rows in sorted(subject_rows.items()):
        successes = sum(row["disposition"] == "decided_satisfaction" for row in rows)
        by_subject[subject_id] = {
            "episodes": len(rows),
            "successes": successes,
            "p_sat": successes / len(rows),
            "failure_classes": dict(sorted(Counter(row["failure_class"] for row in rows).items())),
        }

    alpha = 0.05
    task_ids = sorted(outcomes)
    threshold = bonferroni_alpha(alpha, len(task_ids))
    per_formula = []
    formula_groups: list[list[list[bool]]] = []
    for task_id in task_ids:
        subjects = outcomes[task_id]
        groups = [subjects[subject] for subject in sorted(subjects)]
        if len(groups) != 5 or {len(group) for group in groups} != {20}:
            raise ValueError(f"incomplete formula cell: {task_id}")
        formula_groups.append(groups)
        test = permutation_test_group_range(groups, iterations=args.iterations, seed=args.seed)
        rates = {subject: sum(values) / len(values) for subject, values in sorted(subjects.items())}
        per_formula.append(
            {
                "task_id": task_id,
                "rates": rates,
                "observed_range": test.observed,
                "null_mean": test.null_mean,
                "permutation_p_value": test.p_value,
                "significant_uncorrected_p05": test.p_value < alpha,
                "significant_bonferroni": test.p_value < threshold,
            }
        )
    aggregate_test = permutation_test_median_range(
        formula_groups, iterations=args.iterations, seed=args.seed
    )

    result = {
        "schema_version": "deepltl-point-full-semantic-analysis/1",
        "source_records": records_path.name,
        "source_records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "integrity": {
            "records": record_count,
            "plan_sha256": next(iter(plan_hashes)),
            "official_independent_mismatches": mismatch_count,
            "retained_trace_hashes_verified": record_count,
            "geometry_label_agreements_verified": record_count,
            "prefix_disposition_agreements_verified": record_count,
            "scenario_paired_cells_verified": len(cell_layouts),
        },
        "prefix_statuses": dict(sorted(prefix_counts.items())),
        "dispositions": dict(sorted(disposition_counts.items())),
        "by_failure_class": by_failure_class,
        "by_subject": by_subject,
        "formula_fixed_policy_tests": {
            "iterations": args.iterations,
            "seed": args.seed,
            "alpha": alpha,
            "bonferroni_alpha": threshold,
            "aggregate_median_range": {
                "observed": aggregate_test.observed,
                "null_mean": aggregate_test.null_mean,
                "null_median": aggregate_test.null_median,
                "null_p05": aggregate_test.null_p05,
                "null_p95": aggregate_test.null_p95,
                "permutation_p_value": aggregate_test.p_value,
            },
            "significant_uncorrected_count": sum(
                row["significant_uncorrected_p05"] for row in per_formula
            ),
            "significant_bonferroni_count": sum(
                row["significant_bonferroni"] for row in per_formula
            ),
            "per_formula": per_formula,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    output_path.with_name(output_path.name + ".sha256").write_text(
        f"{digest}  {output_path.name}\n", encoding="utf-8"
    )
    print(json.dumps({"status": "validated", "records": record_count, "output": str(output_path)}))


if __name__ == "__main__":
    main()
