from itertools import product

import pytest

from tlrl_provenance import (
    LabeledTrace,
    evaluate_with_provenance,
    parse_formula,
    temporal_landmarks,
    verify_certificate,
)
from tlrl_provenance.reference import evaluate_reference


def all_traces(atoms, length):
    valuations = [
        frozenset(atom for atom, enabled in zip(atoms, bits) if enabled)
        for bits in product((False, True), repeat=len(atoms))
    ]
    for steps in product(valuations, repeat=length):
        yield LabeledTrace(steps)


FORMULAS = [
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
    "G(not p)",
    "p U q",
    "p R q",
    "p W q",
    "G(p -> F(q or not p))",
]


@pytest.mark.parametrize("source", FORMULAS)
def test_provenance_matches_independent_semantics_exhaustively(source):
    formula = parse_formula(source)
    for length in (1, 2, 3):
        for trace in all_traces(("p", "q"), length):
            certificate = evaluate_with_provenance(formula, trace, verify_reference="direct")
            assert certificate.satisfied == evaluate_reference(formula, trace.labels)
            report = verify_certificate(certificate, exhaustive_atom_time_limit=6)
            assert report.valid, report.errors


def test_global_safety_pass_requires_every_negative_observation():
    certificate = evaluate_with_provenance(
        parse_formula("G(not a)"),
        LabeledTrace([set(), set(), set()]),
    )
    supports = certificate.minimal_supports()
    assert supports.complete
    assert len(supports.supports) == 1
    assert {fact.to_source() for fact in supports.supports[0]} == {
        "not a@0",
        "not a@1",
        "not a@2",
    }


def test_global_safety_failure_exposes_each_counterexample_as_alternative():
    certificate = evaluate_with_provenance(
        parse_formula("G(not a)"),
        LabeledTrace([{"a"}, set(), {"a"}]),
    )
    assert not certificate.satisfied
    supports = certificate.minimal_supports()
    assert supports.complete
    assert {frozenset(fact.to_source() for fact in support) for support in supports.supports} == {
        frozenset({"a@0"}),
        frozenset({"a@2"}),
    }


def test_eventuality_records_all_observed_witness_alternatives():
    certificate = evaluate_with_provenance(
        parse_formula("F(p or q)"),
        LabeledTrace([set(), {"p", "q"}, {"p"}]),
    )
    root_decision = temporal_landmarks(certificate, "oroot", 0)
    assert root_decision.kind == "eventuality_witness"
    assert root_decision.alternatives == ("time@1", "time@2")
    supports = certificate.minimal_supports()
    assert {frozenset(fact.to_source() for fact in support) for support in supports.supports} == {
        frozenset({"p@1"}),
        frozenset({"q@1"}),
        frozenset({"p@2"}),
    }


def test_support_cap_is_explicit():
    certificate = evaluate_with_provenance(
        parse_formula("(p or q) and (r or s)"),
        LabeledTrace([{"p", "q", "r", "s"}]),
    )
    enumeration = certificate.minimal_supports(limit=2)
    assert len(enumeration.supports) == 2
    assert not enumeration.complete


def test_failed_weak_until_has_operator_specific_decision_kind():
    certificate = evaluate_with_provenance(
        parse_formula("p W q"),
        LabeledTrace([set()]),
    )
    decisions = [item for item in certificate.decisions if item.occurrence_id == "oroot"]
    assert decisions[0].kind == "weak_until_unfulfilled"
    assert temporal_landmarks(certificate, "oroot", 0).kind == "weak_until_unfulfilled"
