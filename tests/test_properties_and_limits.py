from itertools import product
import random

import pytest

from tlrl_provenance import EvaluationLimits, LabeledTrace, evaluate_with_provenance, parse_formula
from tlrl_provenance.logic import Atom, Formula, Op, binary, unary
from tlrl_provenance.prefix import PrefixStatus, assess_prefix
from tlrl_provenance.reference import evaluate_reference
from tlrl_provenance.verification import verify_certificate


def random_formula(rng, depth):
    atoms = [Atom("p"), Atom("q")]
    if depth == 0 or rng.random() < 0.2:
        return rng.choice(atoms)
    unary_ops = [Op.NOT, Op.NEXT, Op.WEAK_NEXT, Op.EVENTUALLY, Op.GLOBALLY]
    binary_ops = [
        Op.AND,
        Op.OR,
        Op.IMPLIES,
        Op.IFF,
        Op.UNTIL,
        Op.RELEASE,
        Op.WEAK_UNTIL,
    ]
    if rng.random() < 0.4:
        return unary(rng.choice(unary_ops), random_formula(rng, depth - 1))
    return binary(
        rng.choice(binary_ops),
        random_formula(rng, depth - 1),
        random_formula(rng, depth - 1),
    )


def random_trace(rng, length):
    return LabeledTrace(
        [frozenset(atom for atom in ("p", "q") if rng.choice((False, True))) for _ in range(length)]
    )


def test_random_nested_formulas_match_reference_and_verify():
    rng = random.Random(928_341)
    for _ in range(250):
        formula = random_formula(rng, 3)
        trace = random_trace(rng, rng.randint(1, 4))
        certificate = evaluate_with_provenance(formula, trace)
        assert certificate.satisfied == evaluate_reference(formula, trace.labels)
        report = verify_certificate(certificate, exhaustive_atom_time_limit=8)
        assert report.valid, (formula, trace.labels, report.errors)


def test_release_and_weak_until_match_standard_dualities():
    release = parse_formula("p R q")
    release_dual = parse_formula("not ((not p) U (not q))")
    weak_until = parse_formula("p W q")
    weak_until_expanded = parse_formula("(p U q) or G p")
    valuations = (set(), {"p"}, {"q"}, {"p", "q"})
    for length in (1, 2, 3):
        for labels in product(valuations, repeat=length):
            assert evaluate_reference(release, labels) == evaluate_reference(release_dual, labels)
            assert evaluate_reference(weak_until, labels) == evaluate_reference(weak_until_expanded, labels)


def test_definite_prefix_classifications_are_sound_for_short_extensions():
    formulas = [parse_formula(item) for item in ("F p", "G p", "X p", "Xw p", "p U q", "p R q")]
    valuations = (frozenset(), frozenset({"p"}), frozenset({"q"}), frozenset({"p", "q"}))
    for formula in formulas:
        for first in valuations:
            prefix = LabeledTrace([first], terminated=False, truncated=True)
            assessment = assess_prefix(formula, prefix)
            if assessment.status is PrefixStatus.OPEN:
                continue
            expected = assessment.status is PrefixStatus.DEFINITELY_SATISFIED
            assert evaluate_reference(formula, prefix.labels) == expected
            for extra_length in (1, 2):
                for extension in product(valuations, repeat=extra_length):
                    assert evaluate_reference(formula, prefix.labels + extension) == expected


def test_resource_limits_fail_explicitly():
    formula = parse_formula("G(p or q)")
    trace = LabeledTrace([set(), set()])
    with pytest.raises(RuntimeError, match="position limit"):
        evaluate_with_provenance(
            formula,
            trace,
            limits=EvaluationLimits(max_trace_positions=1),
        )
    with pytest.raises(RuntimeError, match="cells"):
        evaluate_with_provenance(
            formula,
            trace,
            limits=EvaluationLimits(max_cells=2),
        )


def test_long_global_certificate_queries_are_stack_safe():
    position_count = 2_000
    certificate = evaluate_with_provenance(
        parse_formula("G p"),
        LabeledTrace([{"p"}] * position_count),
    )
    assert len(certificate.necessary_evidence) == position_count
    enumeration = certificate.minimal_supports(limit=2)
    assert enumeration.complete
    assert len(enumeration.supports) == 1
    assert len(enumeration.supports[0]) == position_count
