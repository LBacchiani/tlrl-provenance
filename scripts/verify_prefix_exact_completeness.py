"""Retained verification for the prefix-status completeness fix.

Confirms two claims made in the paper about the exact continuation-aware
status procedure in ``tlrl_provenance.prefix``:

1. It agrees with a separately implemented reachability checker across every
   formula in all three confirmatory corpora, under many random synthetic
   traces.  The checker deliberately shares the already-tested
   ``progress_open`` and ``evaluate_reference_dp`` semantic primitives with
   the production procedure: this is a regression check of the new search
   control logic, not an independent validation of those primitives.
2. For PointLTL specifically -- the only benchmark whose frozen records
   retain the full per-step proposition trace (``retained_trace.reported
   _labels``) -- recomputing ``prefix_status`` from the real, frozen,
   already-audited 5,000-episode confirmatory dataset reproduces the stored
   value for every record.

Both were previously run once, interactively, during development. This
script makes them a retained, rerunnable, hash-anchored artifact rather
than an unrepeatable development event. It does not touch the frozen
confirmatory campaigns themselves.

Usage:
    python scripts/verify_prefix_exact_completeness.py
"""

from __future__ import annotations

import os
import hashlib
import itertools
import json
import pathlib
import random
import re
import sys
from collections import Counter
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tlrl_provenance import parse_formula  # noqa: E402
from tlrl_provenance.prefix import assess_prefix, progress_open  # noqa: E402
from tlrl_provenance.reference import evaluate_reference_dp  # noqa: E402
from tlrl_provenance.trace import LabeledTrace  # noqa: E402

OUTPUT_PATH = ROOT / "evidence" / "prefix_exact_completeness_verification.json"
if os.environ.get("TLRL_OUTPUT_DIR"):  # redirect generated evidence; frozen files stay untouched
    OUTPUT_PATH = pathlib.Path(os.environ["TLRL_OUTPUT_DIR"]) / OUTPUT_PATH.name

_OPERATOR_TOKENS = {"U", "F", "G", "X", "Xw", "R", "W", "and", "or", "not", "top", "true", "bottom", "false"}

CORPORA = {
    "deepltl_point": (ROOT / "src/tlrl_benchmarks/deepltl_point/tasks.txt", False),
    "letter_env": (ROOT / "src/tlrl_benchmarks/letter_env/tasks.txt", False),
    "flatworld": (ROOT / "src/tlrl_benchmarks/flatworld/tasks.txt", True),
}

POINTLTL_RECORDS = (
    ROOT
    / "evidence/deepltl_point/confirmatory/semantic/deepltl_point_full_confirmatory_semantic_records.jsonl"
)

TRACE_LENGTHS = (1, 2, 3, 6, 10)
TRIALS_PER_LENGTH = 4
SEED = 7


def _formula_atoms(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text)
    return sorted(set(tokens) - _OPERATOR_TOKENS)


def _powerset_alphabet(atoms: list[str]) -> list[frozenset[str]]:
    return [frozenset(c) for r in range(len(atoms) + 1) for c in itertools.combinations(atoms, r)]


def _reachability_checker_status(
    formula: Any,
    prefix_labels: list[frozenset[str]],
    atoms: list[str],
    max_states: int = 4000,
) -> str:
    """Status from a separate implementation of the reachability traversal.

    This intentionally shares ``progress_open`` and ``evaluate_reference_dp``
    with the production procedure.  Its purpose is to catch regressions in
    the new traversal/control logic; it is not an independent implementation
    of formula progression or finite-trace evaluation.
    """

    finite = evaluate_reference_dp(formula, prefix_labels)
    residual = formula
    for lab in prefix_labels:
        residual = progress_open(residual, frozenset(lab))
    alphabet = _powerset_alphabet(atoms)
    seen_true, seen_false = finite, not finite
    visited = {repr(residual)}
    frontier = [residual]
    explored = 0
    while frontier:
        if seen_true and seen_false:
            return "open"
        psi = frontier.pop()
        explored += 1
        if explored > max_states:
            return "BOUND_HIT"
        for label in alphabet:
            if evaluate_reference_dp(psi, [label]):
                seen_true = True
            else:
                seen_false = True
            if seen_true and seen_false:
                return "open"
            successor = progress_open(psi, label)
            key = repr(successor)
            if key not in visited:
                visited.add(key)
                frontier.append(successor)
    return "definitely_satisfied" if seen_true else "definitely_violated"


def _random_labels(n: int, atoms: list[str], rng: random.Random, allow_multi: bool) -> list[frozenset[str]]:
    alphabet = _powerset_alphabet(atoms) if allow_multi else [frozenset(), *[frozenset({a}) for a in atoms]]
    return [rng.choice(alphabet) for _ in range(n)]


def verify_corpus_sweep() -> dict[str, Any]:
    rng = random.Random(SEED)
    total_trials = 0
    mismatches: list[dict[str, Any]] = []
    checker_bound_hits: list[dict[str, Any]] = []
    incomplete_decisions: list[dict[str, Any]] = []
    decision_origins: Counter[str] = Counter()
    checker_statuses: Counter[str] = Counter()
    per_corpus: dict[str, dict[str, Any]] = {}
    for name, (path, allow_multi) in CORPORA.items():
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        corpus_trials = 0
        for formula_text in lines:
            formula = parse_formula(formula_text)
            atoms = _formula_atoms(formula_text)
            for trace_len in TRACE_LENGTHS:
                for _ in range(TRIALS_PER_LENGTH):
                    labels = _random_labels(trace_len, atoms, rng, allow_multi)
                    trace = LabeledTrace(labels, terminated=False, truncated=True)
                    assessment = assess_prefix(formula, trace)
                    impl = assessment.status.value
                    checker = _reachability_checker_status(formula, labels, atoms)
                    total_trials += 1
                    corpus_trials += 1
                    decision_origins[assessment.decision_origin.value] += 1
                    checker_statuses[checker] += 1
                    case = {
                        "corpus": name,
                        "formula": formula_text,
                        "labels": [sorted(lab) for lab in labels],
                    }
                    if not assessment.decision_complete:
                        incomplete_decisions.append(
                            {
                                **case,
                                "implementation_status": impl,
                                "decision_origin": assessment.decision_origin.value,
                            }
                        )
                    if checker == "BOUND_HIT":
                        checker_bound_hits.append(case)
                    elif impl != checker:
                        mismatches.append(
                            {
                                **case,
                                "implementation_status": impl,
                                "checker_status": checker,
                            }
                        )
        per_corpus[name] = {"num_formulas": len(lines), "trials": corpus_trials}
    return {
        "total_trials": total_trials,
        "total_mismatches": len(mismatches),
        "mismatches": mismatches,
        "checker_bound_hit_count": len(checker_bound_hits),
        "checker_bound_hits": checker_bound_hits,
        "incomplete_decision_count": len(incomplete_decisions),
        "incomplete_decisions": incomplete_decisions,
        "decision_origin_counts": dict(sorted(decision_origins.items())),
        "checker_status_counts": dict(sorted(checker_statuses.items())),
        "per_corpus": per_corpus,
        "trace_lengths": list(TRACE_LENGTHS),
        "trials_per_length": TRIALS_PER_LENGTH,
        "seed": SEED,
    }


def verify_pointltl_record_replay() -> dict[str, Any]:
    if not POINTLTL_RECORDS.exists():
        return {"skipped": True, "reason": "frozen PointLTL records not found"}
    formula_cache: dict[str, Any] = {}
    total = 0
    mismatches: list[dict[str, Any]] = []
    incomplete_decisions: list[dict[str, Any]] = []
    decision_origins: Counter[str] = Counter()
    with POINTLTL_RECORDS.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            total += 1
            text = record["formula"]
            formula = formula_cache.get(text)
            if formula is None:
                formula = parse_formula(text)
                formula_cache[text] = formula
            labels = record["retained_trace"]["reported_labels"]
            trace = LabeledTrace(labels, terminated=bool(record["terminated"]), truncated=bool(record["truncated"]))
            result = assess_prefix(formula, trace)
            decision_origins[result.decision_origin.value] += 1
            identity = {
                "task_id": record["task_id"],
                "subject_id": record["subject_id"],
                "episode_index": record["episode_index"],
            }
            if not result.decision_complete:
                incomplete_decisions.append(
                    {
                        **identity,
                        "status": result.status.value,
                        "decision_origin": result.decision_origin.value,
                    }
                )
            if result.status.value != record["prefix_status"]:
                mismatches.append(
                    {
                        **identity,
                        "stored_prefix_status": record["prefix_status"],
                        "recomputed_status": result.status.value,
                    }
                )
    return {
        "source": POINTLTL_RECORDS.name,
        "source_sha256": hashlib.sha256(POINTLTL_RECORDS.read_bytes()).hexdigest(),
        "total_records": total,
        "mismatches": mismatches,
        "decision_origin_counts": dict(sorted(decision_origins.items())),
        "incomplete_decision_count": len(incomplete_decisions),
        "incomplete_decisions": incomplete_decisions,
    }


def main() -> None:
    corpus_sweep = verify_corpus_sweep()
    pointltl_replay = verify_pointltl_record_replay()

    result = {
        "schema_version": "prefix-exact-completeness-verification/2",
        "purpose": (
            "Confirms the exact continuation-aware status procedure (the "
            "fix for the G(Xp)-style completeness defect found via the "
            "operator-generality stress test) agrees with a separately "
            "implemented reachability checker across all real corpus "
            "formulas, and reproduces the "
            "stored prefix_status for every frozen PointLTL confirmatory "
            "record. The checker shares the production progression and "
            "reference-evaluation primitives but implements the reachability "
            "traversal separately. Mismatches, checker bound hits, or "
            "incomplete production decisions are verification failures."
        ),
        "corpus_sweep": corpus_sweep,
        "pointltl_record_replay": pointltl_replay,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_bytes((json.dumps(result, indent=2) + "\n").encode("utf-8"))
    digest = hashlib.sha256(OUTPUT_PATH.read_bytes()).hexdigest()
    (OUTPUT_PATH.parent / f"{OUTPUT_PATH.name}.sha256").write_bytes(
        f"{digest}  {OUTPUT_PATH.name}\n".encode("utf-8")
    )

    print(f"wrote {OUTPUT_PATH} ({digest})")
    print(f"corpus sweep: {corpus_sweep['total_trials']} trials, {corpus_sweep['total_mismatches']} mismatches")
    print(
        "  completeness:",
        f"checker_bound_hits={corpus_sweep['checker_bound_hit_count']},",
        f"incomplete_decisions={corpus_sweep['incomplete_decision_count']},",
        f"origins={corpus_sweep['decision_origin_counts']}",
    )
    for name, stats in corpus_sweep["per_corpus"].items():
        print(f"  {name}: {stats['num_formulas']} formulas, {stats['trials']} trials")
    if pointltl_replay.get("skipped"):
        print("PointLTL record replay: skipped -", pointltl_replay["reason"])
    else:
        print(
            f"PointLTL record replay: {pointltl_replay['total_records']} records, "
            f"{len(pointltl_replay['mismatches'])} mismatches, "
            f"{pointltl_replay['incomplete_decision_count']} incomplete, "
            f"origins={pointltl_replay['decision_origin_counts']}"
        )

    failed = (
        corpus_sweep["total_mismatches"] > 0
        or corpus_sweep["checker_bound_hit_count"] > 0
        or corpus_sweep["incomplete_decision_count"] > 0
        or pointltl_replay.get("skipped", False)
        or len(pointltl_replay.get("mismatches", [])) > 0
        or pointltl_replay.get("incomplete_decision_count", 0) > 0
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
