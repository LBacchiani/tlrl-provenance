"""Significance testing for the scenario-paired FlatWorld confirmatory campaigns.

Mirrors ``analyze_letter_env_significance.py`` exactly in method (paired
permutation tests that only shuffle within a matched scenario, add-one Monte
Carlo p-value correction, Bonferroni correction across the formula corpus,
and a direct data-level pairing check rather than trusting the seed
namespace). The one structural difference: FlatWorld's nine-circle geometry
is a frozen class constant (see ``tlrl_benchmarks.flatworld.adapter``), not
a per-episode random draw, so there is no map digest to compare across
subjects the way LetterEnv has one. Pairing is instead verified directly
from each record's own ``start_position`` (the one thing that does vary with
the reset seed): every subject sharing a ``scenario_seed`` must land on the
identical starting position.

Reads the already-frozen, already-verified
``flatworld_{algorithm}_full_confirmatory_semantic_records.jsonl``. Read-only
with respect to the frozen campaign: does not touch the plan, records,
summary, or auto-generated report, and writes a new, separately
hash-anchored analysis artifact alongside them.

Usage:
    python scripts/analyze_flatworld_significance.py --algorithm ppo
"""

from __future__ import annotations

import os
import argparse
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

ITERATIONS = 100_000
SEED = 0
ALPHA = 0.05
EXPECTED_FORMULAS = 49
EXPECTED_SEEDS = 5
EXPECTED_EPISODES_PER_CELL = 20
POSITION_TOLERANCE = 1e-9


def paths_for(algorithm: str) -> tuple[Path, Path]:
    base = REPO_ROOT / "evidence" / "flatworld" / algorithm / "confirmatory" / "semantic"
    records = base / f"flatworld_{algorithm}_full_confirmatory_semantic_records.jsonl"
    output = base / f"flatworld_{algorithm}_significance_analysis.json"
    return records, output


def load_cells(records_path: Path) -> dict[str, dict[int, dict[str, tuple[bool, tuple[float, float]]]]]:
    """task_id -> episode_index -> subject_id -> (official_success, start_position)."""

    cells: dict[str, dict[int, dict[str, tuple[bool, tuple[float, float]]]]] = defaultdict(lambda: defaultdict(dict))
    with records_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            position = record.get("start_position")
            if position is None or len(position) != 2:
                raise ValueError(
                    f"record for {record['task_id']}/{record['subject_id']}/episode "
                    f"{record['episode_index']} is missing a valid start_position; "
                    "cannot verify scenario pairing"
                )
            cells[record["task_id"]][int(record["episode_index"])][record["subject_id"]] = (
                bool(record["official_success"]),
                (float(position[0]), float(position[1])),
            )
    return {task_id: dict(episodes) for task_id, episodes in cells.items()}


def validate_pairing(
    cells: dict[str, dict[int, dict[str, tuple[bool, tuple[float, float]]]]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Confirm every matched cell actually has identical start positions across subjects."""

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
            positions = [position for _success, position in by_subject.values()]
            reference = positions[0]
            for position in positions[1:]:
                if abs(position[0] - reference[0]) > POSITION_TOLERANCE or abs(position[1] - reference[1]) > POSITION_TOLERANCE:
                    raise ValueError(
                        f"scenario pairing violated at {task_id}/episode {episode_index}: "
                        f"subjects disagree on start position ({positions}); the paired "
                        "permutation test below is invalid until this is fixed"
                    )
    sorted_subjects = tuple(sorted(subject_ids))
    if len(sorted_subjects) != EXPECTED_SEEDS:
        raise ValueError(f"expected {EXPECTED_SEEDS} distinct subjects, found {sorted_subjects}")
    return task_ids, sorted_subjects


def blocks_for_task(
    cells: dict[str, dict[int, dict[str, tuple[bool, tuple[float, float]]]]],
    task_id: str,
    subject_ids: tuple[str, ...],
) -> list[list[bool]]:
    episodes = cells[task_id]
    return [
        [episodes[episode_index][subject_id][0] for subject_id in subject_ids]
        for episode_index in sorted(episodes)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["ppo"], required=True)
    parser.add_argument(
        "--screen-formulas", nargs="*", default=(),
        help="Optional task_ids to additionally run the descriptive weakest-vs-strongest Fisher screen on.",
    )
    args = parser.parse_args()

    records_path, output_path = paths_for(args.algorithm)
    if os.environ.get("TLRL_OUTPUT_DIR"):  # redirect generated evidence; frozen files stay untouched
        output_path = Path(os.environ["TLRL_OUTPUT_DIR"]) / output_path.name
    if not records_path.exists():
        raise FileNotFoundError(f"frozen records not found: {records_path}")

    cells = load_cells(records_path)
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
    screen_ids = tuple(args.screen_formulas) or tuple(
        row["task_id"]
        for row in sorted(per_formula, key=lambda item: item["observed_range"], reverse=True)[:4]
    )
    for task_id in screen_ids:
        by_subject: dict[str, list[bool]] = defaultdict(list)
        for episode_index, subjects in cells[task_id].items():
            for subject_id, (success, _position) in subjects.items():
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
        "schema_version": "flatworld-paired-significance-analysis/1",
        "algorithm": args.algorithm,
        "source_records": records_path.name,
        "source_records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "pairing_validation": {
            "status": "passed",
            "note": (
                "Every (task_id, episode_index) cell was confirmed -- from the recorded "
                "start_position, not assumed from the seed namespace -- to have an "
                "identical starting position across all subjects before any test ran. "
                "FlatWorld's nine-circle geometry is a frozen constant, so start_position "
                "is the only thing a shared scenario_seed can vary."
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
                "selected after seeing the data (by observed range, or explicitly requested). "
                "This is a form of selection on top of, not replacing, the Bonferroni correction "
                "across formulas above. Do not cite these p-values as confirmatory evidence; use "
                "them only to illustrate what a specific requirement-level gap looks like."
            ),
            "selection_rule": (
                "user_specified" if args.screen_formulas
                else "top_4_by_observed_paired_permutation_range_post_hoc"
            ),
            "rows": descriptive_screen,
        },
    }

    output_path.write_text(json.dumps(analysis, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    (output_path.parent / f"{output_path.name}.sha256").write_text(
        f"{digest}  {output_path.name}\n", encoding="utf-8"
    )

    print(f"wrote {output_path} ({digest})")
    print(f"pairing validation: passed ({len(task_ids)} formulas x {len(subject_ids)} subjects x {EXPECTED_EPISODES_PER_CELL} episodes)")
    print(f"significant at uncorrected p<0.05: {significant_uncorrected}/{len(task_ids)}")
    print(f"significant at Bonferroni alpha={bonferroni_threshold:.4f}: {significant_bonferroni}/{len(task_ids)}")
    print(f"aggregate median-range paired permutation p-value: {aggregate.p_value:.4f}")
    print("descriptive (NOT confirmatory) weakest-vs-strongest screen:")
    for row in descriptive_screen:
        print(f"  {row['task_id']}: fisher p={row['fisher_exact_p_value']:.5f}")


if __name__ == "__main__":
    main()
