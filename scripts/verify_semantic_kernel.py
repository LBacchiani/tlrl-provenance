"""Independent, benchmark-free verification run for the semantic kernel."""

from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tlrl_provenance import (  # noqa: E402
    LabeledTrace,
    SupportEnumerationMode,
    evaluate_with_provenance,
    parse_formula,
)
from tlrl_provenance.logic import Atom, Op, binary, unary  # noqa: E402
from tlrl_provenance.verification import verify_certificate  # noqa: E402


FORMULAS = (
    "top",
    "bottom",
    "p",
    "not p",
    "p and q",
    "p or q",
    "p -> q",
    "p <-> q",
    "X p",
    "Xw p",
    "F p",
    "G p",
    "p U q",
    "p R q",
    "p W q",
    "G(p -> F(q or not p))",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--random-cases", type=int, default=500)
    args = parser.parse_args()
    if args.random_cases < 0:
        parser.error("--random-cases must be non-negative")

    valuations = tuple(
        frozenset(atom for atom, enabled in zip(("p", "q"), bits) if enabled)
        for bits in product((False, True), repeat=2)
    )
    certificate_count = 0
    cell_count = 0
    support_count = 0
    exhaustive_trace_count = 0
    for source in FORMULAS:
        formula = parse_formula(source)
        for length in (1, 2, 3):
            for labels in product(valuations, repeat=length):
                exhaustive_trace_count += 1
                certificate = evaluate_with_provenance(
                    formula,
                    LabeledTrace(labels),
                    verify_reference="direct",
                )
                report = verify_certificate(certificate, exhaustive_atom_time_limit=6)
                if not report.valid:
                    raise AssertionError((source, labels, report.errors))
                certificate_count += 1
                cell_count += report.checked_cells
                support_count += report.checked_supports

    rng = random.Random(0x5E6A71C)
    for _ in range(args.random_cases):
        formula = _random_formula(rng, 4)
        labels = tuple(rng.choice(valuations) for _ in range(rng.randint(1, 4)))
        certificate = evaluate_with_provenance(
            formula,
            LabeledTrace(labels),
            verify_reference="direct",
        )
        report = verify_certificate(certificate, exhaustive_atom_time_limit=8)
        if not report.valid:
            raise AssertionError((formula.to_source(), labels, report.errors))
        certificate_count += 1
        cell_count += report.checked_cells
        support_count += report.checked_supports

    stress_formula = "F(a) and F(b) and F(c) and F(d)"
    stress_certificate = evaluate_with_provenance(
        parse_formula(stress_formula),
        LabeledTrace([{"a", "b", "c", "d"}] * 40),
    )
    stress_started = time.perf_counter()
    stress_enumeration = stress_certificate.minimal_supports(
        limit=10_000,
        work_limit=100_000,
    )
    stress_elapsed = time.perf_counter() - stress_started
    if (
        stress_enumeration.complete
        or len(stress_enumeration.supports) != 10_000
        or stress_enumeration.work_done > stress_enumeration.work_limit
        or stress_enumeration.mode is not SupportEnumerationMode.EXACT
        or stress_enumeration.truncation_reasons != ("output_limit",)
    ):
        raise AssertionError(stress_enumeration)
    partial_stress = stress_certificate.minimal_supports(
        limit=10_000,
        work_limit=100_000,
        mode=SupportEnumerationMode.BOUNDED_PARTIAL,
    )
    if (
        not partial_stress.supports
        or partial_stress.complete
        or partial_stress.intermediate_limit != 16
        or partial_stress.truncation_reasons != ("intermediate_limit",)
    ):
        raise AssertionError(partial_stress)
    tiny_work = stress_certificate.minimal_supports(limit=10_000, work_limit=50)
    if (
        tiny_work.complete
        or tiny_work.supports
        or tiny_work.work_done != 50
        or tiny_work.truncation_reasons != ("work_limit",)
    ):
        raise AssertionError(tiny_work)

    linear_temporal_supports = {}
    for source, labels, expected in (
        ("F p", [{"p"}] * 40, 40),
        ("G(not p)", [{"p"}] * 40, 40),
        ("p U q", [{"p", "q"}] * 40, 40),
        ("p W q", [{"p", "q"}] * 40, 41),
    ):
        certificate = evaluate_with_provenance(
            parse_formula(source),
            LabeledTrace(labels),
        )
        mode_results = {}
        for mode in (
            SupportEnumerationMode.EXACT,
            SupportEnumerationMode.BOUNDED_PARTIAL,
        ):
            enumeration = certificate.minimal_supports(limit=10_000, mode=mode)
            if (
                not enumeration.complete
                or len(enumeration.supports) != expected
                or enumeration.truncation_reasons
            ):
                raise AssertionError((source, mode.value, enumeration))
            mode_results[mode.value] = {
                "returned_supports": len(enumeration.supports),
                "complete": enumeration.complete,
                "work_done": enumeration.work_done,
            }
        linear_temporal_supports[source] = mode_results

    # Regression for the discovery made against real cached Highway DQN
    # rollouts: G(trigger -> F_leq_K(persist_H(...))) folded over an
    # ordinary, mostly-safe ~20-step episode used to exhaust the default
    # work budget completely (0 supports), because every non-triggered
    # G-level still offers a genuine antecedent-false/consequent-true
    # branching alternative that compounds across the whole chain -- a
    # fundamentally different shape from a handful of sibling top-level
    # ANDs, and not caught by the stress case above.
    def x_pow(n, phi):
        for _ in range(n):
            phi = unary(Op.NEXT, phi)
        return phi

    def persist(phi, h):
        acc = phi
        for i in range(1, h):
            acc = binary(Op.AND, acc, x_pow(i, phi))
        return acc

    def within(phi, k):
        acc = phi
        for i in range(1, k + 1):
            acc = binary(Op.OR, acc, x_pow(i, phi))
        return acc

    safe_atom, hazard_atom, collision_atom = Atom("safe"), Atom("hazard"), Atom("collision")
    response_ok = within(persist(safe_atom, 2), 4)
    gfold_formula = binary(
        Op.AND,
        unary(Op.GLOBALLY, unary(Op.NOT, collision_atom)),
        unary(Op.GLOBALLY, binary(Op.IMPLIES, hazard_atom, response_ok)),
    )
    gfold_labels = [{"hazard"}] + [{"safe"}] * 20
    gfold_certificate = evaluate_with_provenance(gfold_formula, LabeledTrace(gfold_labels))
    gfold_started = time.perf_counter()
    gfold_enumeration = gfold_certificate.minimal_supports(
        limit=10_000,
        mode=SupportEnumerationMode.BOUNDED_PARTIAL,
    )
    gfold_elapsed = time.perf_counter() - gfold_started
    if (
        not gfold_enumeration.supports
        or gfold_enumeration.mode is not SupportEnumerationMode.BOUNDED_PARTIAL
        or gfold_enumeration.truncation_reasons != ("intermediate_limit",)
        or gfold_enumeration.work_done >= gfold_enumeration.work_limit
        or gfold_elapsed > 2.0
    ):
        raise AssertionError(("gfold_regression", gfold_enumeration, gfold_elapsed))

    result = {
        "schema": "tlrl-semantic-kernel-verification/3",
        "status": "pass",
        "formula_operator_suite_count": len(FORMULAS),
        "exhaustive_trace_count_per_formula": exhaustive_trace_count // len(FORMULAS),
        "random_seed": 0x5E6A71C,
        "random_nested_formula_count": args.random_cases,
        "verified_certificate_count": certificate_count,
        "verified_cell_count": cell_count,
        "verified_support_count": support_count,
        "support_enumeration_stress": {
            "formula": stress_formula,
            "trace_positions": 40,
            "returned_supports": len(stress_enumeration.supports),
            "complete": stress_enumeration.complete,
            "mode": stress_enumeration.mode.value,
            "intermediate_limit": stress_enumeration.intermediate_limit,
            "work_done": stress_enumeration.work_done,
            "work_limit": stress_enumeration.work_limit,
            "truncation_reasons": list(stress_enumeration.truncation_reasons),
            "truncated_node_ids": list(stress_enumeration.truncated_node_ids),
            "elapsed_seconds": stress_elapsed,
            "tiny_work_limit_checked": tiny_work.work_limit,
        },
        "bounded_partial_support_enumeration_stress": {
            "formula": stress_formula,
            "trace_positions": 40,
            "returned_supports": len(partial_stress.supports),
            "complete": partial_stress.complete,
            "mode": partial_stress.mode.value,
            "intermediate_limit": partial_stress.intermediate_limit,
            "work_done": partial_stress.work_done,
            "work_limit": partial_stress.work_limit,
            "truncation_reasons": list(partial_stress.truncation_reasons),
            "truncated_node_ids": list(partial_stress.truncated_node_ids),
        },
        "linear_temporal_support_regression": linear_temporal_supports,
        "gfold_support_enumeration_stress": {
            "formula": gfold_formula.to_source(),
            "trace_positions": len(gfold_labels),
            "returned_supports": len(gfold_enumeration.supports),
            "complete": gfold_enumeration.complete,
            "mode": gfold_enumeration.mode.value,
            "intermediate_limit": gfold_enumeration.intermediate_limit,
            "work_done": gfold_enumeration.work_done,
            "work_limit": gfold_enumeration.work_limit,
            "truncation_reasons": list(gfold_enumeration.truncation_reasons),
            "truncated_node_ids": list(gfold_enumeration.truncated_node_ids),
            "elapsed_seconds": gfold_elapsed,
        },
        "source_sha256": _source_digest(),
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
    }
    payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(args.output, payload)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        _atomic_write(
            args.output.with_suffix(args.output.suffix + ".sha256"),
            f"{digest}  {args.output.name}\n",
        )
    return 0


def _random_formula(rng: random.Random, depth: int):
    if depth == 0 or rng.random() < 0.18:
        return Atom(rng.choice(("p", "q")))
    if rng.random() < 0.4:
        return unary(
            rng.choice((Op.NOT, Op.NEXT, Op.WEAK_NEXT, Op.EVENTUALLY, Op.GLOBALLY)),
            _random_formula(rng, depth - 1),
        )
    return binary(
        rng.choice(
            (
                Op.AND,
                Op.OR,
                Op.IMPLIES,
                Op.IFF,
                Op.UNTIL,
                Op.RELEASE,
                Op.WEAK_UNTIL,
            )
        ),
        _random_formula(rng, depth - 1),
        _random_formula(rng, depth - 1),
    )


def _source_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
