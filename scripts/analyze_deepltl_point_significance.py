"""Significance testing for the scenario-paired PointWorld confirmatory campaign.

Answers, with real hypothesis tests instead of raw ranges, the "N=20
statistical trap" critique: at n=20 episodes per policy, how much of the
observed cross-policy P_sat spread is distinguishable from binomial noise
alone?

Reads the already-frozen, already-verified
``deepltl_point_full_confirmatory_semantic_records.jsonl`` (5,000 records,
zero official/independent mismatches -- see that campaign's own artifacts).
This script is read-only with respect to the frozen campaign: it does not
touch the plan, records, summary, or auto-generated report, and writes a
new, separately hash-anchored analysis artifact alongside them.

Fixes two issues present in an earlier version of this script (and shared,
via the same statistics functions, with the first version of the LetterEnv
significance analysis):

1. The earlier per-formula test pooled every episode across all five
   subjects and reshuffled the pool freely, discarding exactly the pairing
   the campaign was designed to provide (a shared ``scenario_seed`` across
   subjects for a given (task_id, episode_index) cell). This version uses
   ``permutation_test_paired_group_range``/``..._median_range``, which only
   permutes subject labels *within* each matched scenario, never moving an
   outcome to a different scenario -- and the pairing is verified directly
   from each record's retained zone layout, not assumed from the seed
   namespace.
2. The earlier p-value was a raw ``count / iterations`` ratio, which can
   report an exact ``p = 0``. The shared statistics functions now apply the
   standard add-one Monte Carlo correction.

The weakest-vs-strongest Fisher exact screen is retained but reported under
a name and description that make its status explicit: it compares the two
most different-looking subjects for a formula selected after seeing the
data, which is a form of selection on top of (not replacing) the Bonferroni
correction across formulas. It is not confirmatory evidence.

Usage:
    python scripts/analyze_deepltl_point_significance.py
"""

from __future__ import annotations

import os
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tlrl_provenance import (  # noqa: E402
    bonferroni_alpha,
    fisher_exact_two_sided,
    permutation_test_paired_group_range,
    permutation_test_paired_median_range,
)

RECORDS_PATH = (
    REPO_ROOT
    / "evidence"
    / "deepltl_point"
    / "confirmatory"
    / "semantic"
    / "deepltl_point_full_confirmatory_semantic_records.jsonl"
)
OUTPUT_PATH = (
    REPO_ROOT
    / "evidence"
    / "deepltl_point"
    / "confirmatory"
    / "semantic"
    / "deepltl_point_significance_analysis.json"
)
if os.environ.get("TLRL_OUTPUT_DIR"):  # redirect generated evidence; frozen files stay untouched
    OUTPUT_PATH = Path(os.environ["TLRL_OUTPUT_DIR"]) / OUTPUT_PATH.name
ITERATIONS = 100_000
SEED = 0
ALPHA = 0.05
EXPECTED_FORMULAS = 50
EXPECTED_SEEDS = 5
EXPECTED_EPISODES_PER_CELL = 20


def canonical_zone_layout_digest(zone_layout: list[dict]) -> str:
    """Hash the retained zone layout, order-independent of color listing."""

    normalized = sorted(
        (
            str(zone["color"]),
            round(float(zone["radius"]), 12),
            tuple(sorted(tuple(round(float(c), 12) for c in center) for center in zone["centers_xy"])),
        )
        for zone in zone_layout
    )
    payload = json.dumps(normalized, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_cells(records_path: Path) -> dict[str, dict[int, dict[str, tuple[bool, str]]]]:
    """task_id -> episode_index -> subject_id -> (official_success, zone_layout_digest)."""

    cells: dict[str, dict[int, dict[str, tuple[bool, str]]]] = defaultdict(lambda: defaultdict(dict))
    with records_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            zone_layout = record.get("retained_trace", {}).get("zone_layout")
            if zone_layout is None:
                raise ValueError(
                    f"record for {record['task_id']}/{record['subject_id']}/episode "
                    f"{record['episode_index']} has no retained zone_layout; cannot verify pairing"
                )
            digest = canonical_zone_layout_digest(zone_layout)
            cells[record["task_id"]][int(record["episode_index"])][record["subject_id"]] = (
                bool(record["official_success"]),
                digest,
            )
    return {task_id: dict(episodes) for task_id, episodes in cells.items()}


def validate_pairing(
    cells: dict[str, dict[int, dict[str, tuple[bool, str]]]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    task_ids = tuple(sorted(cells))
    if len(task_ids) != EXPECTED_FORMULAS:
        raise ValueError(f"expected {EXPECTED_FORMULAS} formulas, found {len(task_ids)}")

    subject_ids: set[str] = set()
    for task_id in task_ids:
        episodes = cells[task_id]
        if len(episodes) != EXPECTED_EPISODES_PER_CELL:
            raise ValueError(
                f"expected {EXPECTED_EPISODES_PER_CELL} episode indices for {task_id}, found {len(episodes)}"
            )
        for episode_index, by_subject in episodes.items():
            if len(by_subject) != EXPECTED_SEEDS:
                raise ValueError(
                    f"expected {EXPECTED_SEEDS} subjects for {task_id}/episode {episode_index}, "
                    f"found {sorted(by_subject)}"
                )
            subject_ids.update(by_subject)
            digests = {digest for _success, digest in by_subject.values()}
            if len(digests) != 1:
                raise ValueError(
                    f"scenario pairing violated at {task_id}/episode {episode_index}: "
                    f"subjects disagree on the generated zone layout ({digests}); the paired "
                    "permutation test below is invalid until this is fixed"
                )
    sorted_subjects = tuple(sorted(subject_ids))
    if len(sorted_subjects) != EXPECTED_SEEDS:
        raise ValueError(f"expected {EXPECTED_SEEDS} distinct subjects, found {sorted_subjects}")
    return task_ids, sorted_subjects


def blocks_for_task(
    cells: dict[str, dict[int, dict[str, tuple[bool, str]]]],
    task_id: str,
    subject_ids: tuple[str, ...],
) -> list[list[bool]]:
    episodes = cells[task_id]
    return [
        [episodes[episode_index][subject_id][0] for subject_id in subject_ids]
        for episode_index in sorted(episodes)
    ]


def main() -> None:
    if not RECORDS_PATH.exists():
        raise FileNotFoundError(f"frozen records not found: {RECORDS_PATH}")

    cells = load_cells(RECORDS_PATH)
    task_ids, subject_ids = validate_pairing(cells)
    bonferroni_threshold = bonferroni_alpha(ALPHA, len(task_ids))

    per_formula = []
    blocks_by_task: dict[str, list[list[bool]]] = {}
    for task_id in task_ids:
        blocks = blocks_for_task(cells, task_id, subject_ids)
        blocks_by_task[task_id] = blocks
        result = permutation_test_paired_group_range(blocks, iterations=ITERATIONS, seed=SEED)
        per_formula.append(
            {
                "task_id": task_id,
                "observed_range": result.observed,
                "null_mean": result.null_mean,
                "null_p05": result.null_p05,
                "null_median": result.null_median,
                "null_p95": result.null_p95,
                "permutation_p_value": result.p_value,
                "significant_uncorrected_p05": result.p_value < ALPHA,
                "significant_bonferroni": result.p_value < bonferroni_threshold,
            }
        )

    significant_uncorrected = sum(1 for row in per_formula if row["significant_uncorrected_p05"])
    significant_bonferroni = sum(1 for row in per_formula if row["significant_bonferroni"])

    aggregate_blocks = [blocks_by_task[task_id] for task_id in task_ids]
    aggregate = permutation_test_paired_median_range(aggregate_blocks, iterations=ITERATIONS, seed=SEED)

    descriptive_screen = []
    screen_ids = tuple(
        row["task_id"]
        for row in sorted(per_formula, key=lambda item: item["observed_range"], reverse=True)[:4]
    )
    for task_id in screen_ids:
        by_subject: dict[str, list[bool]] = defaultdict(list)
        for episode_index, subjects in cells[task_id].items():
            for subject_id, (success, _digest) in subjects.items():
                by_subject[subject_id].append(success)
        rates = sorted(
            ((subject_id, sum(v), len(v)) for subject_id, v in by_subject.items()),
            key=lambda item: (item[1], item[0]),
        )
        weakest_id, weakest_successes, weakest_trials = rates[0]
        strongest_id, strongest_successes, strongest_trials = rates[-1]
        p_value = fisher_exact_two_sided(weakest_successes, weakest_trials, strongest_successes, strongest_trials)
        descriptive_screen.append(
            {
                "task_id": task_id,
                "weakest_subject_id": weakest_id,
                "weakest_successes": weakest_successes,
                "weakest_trials": weakest_trials,
                "strongest_subject_id": strongest_id,
                "strongest_successes": strongest_successes,
                "strongest_trials": strongest_trials,
                "fisher_exact_p_value": p_value,
            }
        )

    analysis = {
        "schema_version": "deepltl-point-paired-significance-analysis/2",
        "source_records": RECORDS_PATH.name,
        "source_records_sha256": hashlib.sha256(RECORDS_PATH.read_bytes()).hexdigest(),
        "pairing_validation": {
            "status": "passed",
            "note": (
                "Every (task_id, episode_index) cell was confirmed -- from each record's "
                "retained zone_layout, not assumed from the seed namespace -- to have an "
                "identical generated zone layout across all subjects before any test ran."
            ),
        },
        "method": {
            "iterations": ITERATIONS,
            "seed": SEED,
            "alpha": ALPHA,
            "num_comparisons": len(task_ids),
            "bonferroni_alpha": bonferroni_threshold,
            "per_formula_test": "permutation_test_paired_group_range (tlrl_provenance.statistics)",
            "aggregate_test": "permutation_test_paired_median_range (tlrl_provenance.statistics)",
            "p_value_correction": "add-one Monte Carlo correction (k+1)/(B+1); p=0 is never reported",
            "pairing": "subject labels permuted within each matched scenario only, never across scenarios",
        },
        "per_formula": per_formula,
        "significant_uncorrected_p05_count": significant_uncorrected,
        "significant_bonferroni_count": significant_bonferroni,
        "num_formulas": len(task_ids),
        "aggregate_median_range": {
            "observed": aggregate.observed,
            "null_mean": aggregate.null_mean,
            "null_p05": aggregate.null_p05,
            "null_median": aggregate.null_median,
            "null_p95": aggregate.null_p95,
            "permutation_p_value": aggregate.p_value,
        },
        "descriptive_weakest_vs_strongest_screen": {
            "status": "NOT_CONFIRMATORY",
            "warning": (
                "These rows compare the two most different-looking subjects for a formula "
                "selected after seeing the data (by observed range). This is a form of "
                "selection on top of, not replacing, the Bonferroni correction across formulas "
                "above. Do not cite these p-values as confirmatory evidence; use them only to "
                "illustrate what a specific requirement-level gap looks like."
            ),
            "selection_rule": "top_4_by_observed_paired_permutation_range_post_hoc",
            "rows": descriptive_screen,
        },
    }

    # Write canonical LF bytes even on Windows. This evidence directory is
    # intentionally marked ``-text`` so Git never changes frozen artifacts.
    OUTPUT_PATH.write_bytes(
        (json.dumps(analysis, indent=2, sort_keys=False) + "\n").encode("utf-8")
    )
    digest = hashlib.sha256(OUTPUT_PATH.read_bytes()).hexdigest()
    (OUTPUT_PATH.parent / f"{OUTPUT_PATH.name}.sha256").write_bytes(
        f"{digest}  {OUTPUT_PATH.name}\n".encode("utf-8")
    )

    print(f"wrote {OUTPUT_PATH} ({digest})")
    print(f"pairing validation: passed ({len(task_ids)} formulas x {len(subject_ids)} subjects x {EXPECTED_EPISODES_PER_CELL} episodes)")
    print(f"significant at uncorrected p<0.05: {significant_uncorrected}/{len(task_ids)}")
    print(f"significant at Bonferroni alpha={bonferroni_threshold:.4f}: {significant_bonferroni}/{len(task_ids)}")
    print(f"aggregate median-range paired permutation p-value: {aggregate.p_value:.4f}")
    print("descriptive (NOT confirmatory) weakest-vs-strongest screen:")
    for row in descriptive_screen:
        print(f"  {row['task_id']}: fisher p={row['fisher_exact_p_value']:.5f}")


if __name__ == "__main__":
    main()
