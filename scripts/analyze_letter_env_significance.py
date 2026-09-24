"""Significance testing for the scenario-paired LetterEnv confirmatory campaigns.

Mirrors ``analyze_deepltl_point_significance.py`` in spirit, generalized to
take an algorithm (dqn/ppo) instead of one hardcoded benchmark, and fixes two
issues the earlier version shared with it:

1. The earlier per-formula test pooled every episode across all five
   subjects and reshuffled the pool freely, discarding exactly the pairing
   the campaign was designed to provide (a shared ``scenario_seed`` --
   confirmed here by checking recorded map digests agree, not assumed --
   across subjects for a given (task_id, episode_index) cell). This version
   uses ``permutation_test_paired_group_range``/``..._median_range``, which
   only permutes subject labels *within* each matched scenario, never moving
   an outcome to a different scenario.
2. The earlier p-value was a raw ``count / iterations`` ratio, which can
   report an exact ``p = 0``. The shared statistics functions now apply the
   standard add-one Monte Carlo correction; nothing in this script needs to
   do that separately.

The weakest-vs-strongest Fisher exact screen is retained but reported under
a name and description that make its status explicit: it is a post-hoc
comparison of the two most different-looking subjects for a formula chosen
*after* seeing the data, which is a form of selection on top of (not
replacing) the Bonferroni correction across formulas. It is not confirmatory
evidence and must not be cited as such in the paper without saying so.

Reads the already-frozen, already-verified
``letter_env_{algorithm}_full_confirmatory_semantic_records.jsonl`` (5,000
records per algorithm). Read-only with respect to the frozen campaign: does
not touch the plan, records, summary, or auto-generated report, and writes a
new, separately hash-anchored analysis artifact alongside them.

Usage:
    python scripts/analyze_letter_env_significance.py --algorithm dqn
    python scripts/analyze_letter_env_significance.py --algorithm ppo
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
EXPECTED_FORMULAS = 50
EXPECTED_SEEDS = 5
EXPECTED_EPISODES_PER_CELL = 20


def paths_for(algorithm: str) -> tuple[Path, Path]:
    base = REPO_ROOT / "evidence" / "letter_env" / algorithm / "confirmatory" / "semantic"
    records = base / f"letter_env_{algorithm}_full_confirmatory_semantic_records.jsonl"
    output = base / f"letter_env_{algorithm}_significance_analysis.json"
    return records, output


def load_cells(records_path: Path) -> dict[str, dict[int, dict[str, tuple[bool, str]]]]:
    """task_id -> episode_index -> subject_id -> (official_success, map_digest).

    Retaining episode_index (rather than pooling per-subject like the
    unpaired version did) is what lets the paired permutation test and the
    map-digest cross-check below actually see the matched-scenario structure.
    """

    cells: dict[str, dict[int, dict[str, tuple[bool, str]]]] = defaultdict(lambda: defaultdict(dict))
    with records_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            digest = record.get("generated_map_canonical_sha256")
            if digest is None:
                raise ValueError(
                    f"record for {record['task_id']}/{record['subject_id']}/episode "
                    f"{record['episode_index']} is missing generated_map_canonical_sha256; "
                    "cannot verify scenario pairing"
                )
            cells[record["task_id"]][int(record["episode_index"])][record["subject_id"]] = (
                bool(record["official_success"]),
                str(digest),
            )
    return {task_id: dict(episodes) for task_id, episodes in cells.items()}


def validate_pairing(
    cells: dict[str, dict[int, dict[str, tuple[bool, str]]]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Confirm every matched cell actually has identical map digests across subjects.

    This is a direct check, not an assumption from the campaign config: the
    scenario_namespace makes subjects *share a seed*, but only the recorded
    map digest confirms they actually generated the same map from it. Returns
    the sorted task_ids and subject_ids after confirming a fully rectangular,
    correctly paired 50 x 5 x 20 design.
    """

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
                    f"subjects disagree on the generated map ({digests}); the paired "
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
    """One row per episode_index, one column per subject, in a fixed subject order.

    This is exactly the matched-block shape ``permutation_test_paired_group_range``
    requires: row i's five outcomes all came from the same generated map.
    """

    episodes = cells[task_id]
    return [
        [episodes[episode_index][subject_id][0] for subject_id in subject_ids]
        for episode_index in sorted(episodes)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["dqn", "ppo"], required=True)
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
        "schema_version": "letter-env-paired-significance-analysis/2",
        "algorithm": args.algorithm,
        "source_records": records_path.name,
        "source_records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "pairing_validation": {
            "status": "passed",
            "note": (
                "Every (task_id, episode_index) cell was confirmed -- from the recorded "
                "generated_map_canonical_sha256, not assumed from the seed namespace -- to have "
                "an identical generated map across all subjects before any test ran."
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
