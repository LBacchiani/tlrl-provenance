"""Stress test: how close does real progression come to the residual-size
guard, and does the guard degrade correctly (to a marked, computationally
unresolved OPEN, never a manufactured verdict) when forced?

Pure post-hoc analysis over formula/trace pairs; no trained checkpoints or
rollouts needed, so it runs under the ordinary project environment:
    python scripts/stress_test_residual_size_guard.py

Three checks:
1. Progress every real corpus formula (all three benchmarks) through an
   adversarial synthetic trace at that benchmark's own official horizon,
   and record the maximum continuation-residual node count reached.
2. Two deliberately adversarial formula constructions -- a wide IFF chain
   and a wide conjunction of independent temporal clauses, designed to
   defeat the simplifier's structural-equality collapse -- stress the same
   guard over a long synthetic trace.
3. Force the guard with an artificially small threshold and confirm
   PrefixAssessment reports it as a guard fallback (decision_complete is
   False, decision_origin is RESIDUAL_SIZE_GUARD), not a proven OPEN.
"""

from __future__ import annotations

import os
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tlrl_provenance import DecisionOrigin, LabeledTrace, PrefixStatus, assess_prefix, parse_formula  # noqa: E402
from tlrl_provenance.prefix import DEFAULT_RESIDUAL_NODE_LIMIT, progress_open  # noqa: E402

OUTPUT_PATH = ROOT / "evidence" / "residual_size_guard_stress_test.json"
if os.environ.get("TLRL_OUTPUT_DIR"):  # redirect generated evidence; frozen files stay untouched
    OUTPUT_PATH = pathlib.Path(os.environ["TLRL_OUTPUT_DIR"]) / OUTPUT_PATH.name

CORPORA = {
    "deepltl_point": (ROOT / "src/tlrl_benchmarks/deepltl_point/tasks.txt", 1000),
    "letter_env": (ROOT / "src/tlrl_benchmarks/letter_env/tasks.txt", 75),
    "flatworld": (ROOT / "src/tlrl_benchmarks/flatworld/tasks.txt", 500),
}


def _node_count(formula) -> int:
    stack = [formula]
    seen: set[int] = set()
    count = 0
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        count += 1
        stack.extend(node.args)
    return count


def _adversarial_trace(atoms: list[str], horizon: int) -> list[frozenset[str]]:
    """Alternates an empty label with a rotating single-atom label -- never
    lets any one atom stay constant, defeating the simplifier's cheapest
    idempotence-based collapse rules step over step."""

    labels = []
    for t in range(horizon):
        labels.append(frozenset({atoms[t % len(atoms)]}) if t % 2 else frozenset())
    return labels


def corpus_worst_case() -> dict:
    per_corpus = {}
    overall_max = 0
    for name, (path, horizon) in CORPORA.items():
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        worst = 0
        worst_formula = None
        for line in lines:
            formula = parse_formula(line)
            atoms = sorted(formula.atoms())
            if not atoms:
                continue
            residual = formula
            local_max = _node_count(residual)
            for labels in _adversarial_trace(atoms, horizon):
                residual = progress_open(residual, labels)
                local_max = max(local_max, _node_count(residual))
            if local_max > worst:
                worst = local_max
                worst_formula = line
        overall_max = max(overall_max, worst)
        per_corpus[name] = {"max_residual_nodes": worst, "worst_formula": worst_formula, "horizon": horizon, "num_formulas": len(lines)}
    return {"per_corpus": per_corpus, "overall_max_residual_nodes": overall_max, "guard_threshold": DEFAULT_RESIDUAL_NODE_LIMIT}


def adversarial_constructions() -> dict:
    results = {}

    # Wide IFF-chain of independent eventually-clauses.
    atoms = [f"p{i}" for i in range(80)]
    formula = parse_formula("(" + " <-> ".join(f"F {a}" for a in atoms) + ")")
    residual = formula
    max_nodes = _node_count(residual)
    for labels in _adversarial_trace(atoms, 2000):
        residual = progress_open(residual, labels)
        max_nodes = max(max_nodes, _node_count(residual))
    results["wide_iff_chain_of_eventually"] = {"atoms": len(atoms), "steps": 2000, "max_residual_nodes": max_nodes}

    # Wide conjunction of independent Until clauses, held permanently open.
    n_clauses = 60
    clauses = [f"(p{i} U q{i})" for i in range(n_clauses)]
    formula2 = parse_formula("(" + " & ".join(clauses) + ")")
    labels_const = frozenset(f"p{i}" for i in range(n_clauses))
    residual2 = formula2
    max_nodes2 = _node_count(residual2)
    for _ in range(2000):
        residual2 = progress_open(residual2, labels_const)
        max_nodes2 = max(max_nodes2, _node_count(residual2))
    results["wide_conjunction_of_until_held_open"] = {"clauses": n_clauses, "steps": 2000, "max_residual_nodes": max_nodes2}

    return results


def forced_guard_check() -> dict:
    formula = parse_formula("(!magenta U (blue & (!green U yellow)))")
    trace = LabeledTrace([{"blue"}] * 50, terminated=False, truncated=True)
    result = assess_prefix(formula, trace, residual_node_limit=1)
    return {
        "status": result.status.value,
        "decision_origin": result.decision_origin.value,
        "decision_complete": result.decision_complete,
        "correctly_marked_as_guard_fallback": (
            result.status is PrefixStatus.OPEN
            and result.decision_origin is DecisionOrigin.RESIDUAL_SIZE_GUARD
            and result.decision_complete is False
        ),
    }


def main() -> None:
    result = {
        "schema_version": "residual-size-guard-stress-test/1",
        "purpose": (
            "Checks how close real corpus formulas come to the residual-size "
            "guard under adversarial synthetic traces, whether deliberately "
            "adversarial formula constructions can drive it further, and "
            "that forcing the guard is reported as an explicit, marked "
            "fallback rather than a proven OPEN."
        ),
        "corpus_worst_case": corpus_worst_case(),
        "adversarial_constructions": adversarial_constructions(),
        "forced_guard_check": forced_guard_check(),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_bytes((json.dumps(result, indent=2) + "\n").encode("utf-8"))
    digest = hashlib.sha256(OUTPUT_PATH.read_bytes()).hexdigest()
    (OUTPUT_PATH.parent / f"{OUTPUT_PATH.name}.sha256").write_bytes(
        f"{digest}  {OUTPUT_PATH.name}\n".encode("utf-8")
    )

    print(f"wrote {OUTPUT_PATH} ({digest})")
    print("corpus worst case:", result["corpus_worst_case"]["overall_max_residual_nodes"], "/", DEFAULT_RESIDUAL_NODE_LIMIT)
    for name, stats in result["corpus_worst_case"]["per_corpus"].items():
        print(f"  {name}: max={stats['max_residual_nodes']}")
    for name, stats in result["adversarial_constructions"].items():
        print(f"  {name}: max={stats['max_residual_nodes']}")
    print("forced guard check:", result["forced_guard_check"])


if __name__ == "__main__":
    main()
