import time
from itertools import product

import pytest

from tlrl_provenance import (
    AuditConfig,
    LabeledTrace,
    SupportEnumerationMode,
    audit_traces,
    evaluate_with_provenance,
    parse_formula,
)
from tlrl_provenance.circuit import CircuitBuilder, NodeKind
from tlrl_provenance.trace import EvidenceFact, FactKind
from tlrl_provenance.logic import Atom, Op, binary, unary
from tlrl_provenance.verification import support_entails_verdict, verify_certificate


def _bounded_response_formula(k_steps, h_steps):
    """G(not collision) and G(hazard -> F_{<=k}(persist_h(safe))), built by
    bounded X-unrolling -- the real shape discovered (on actual cached
    Highway DQN rollouts) to compound multiplicatively across a G-fold over
    an ordinary, mostly-safe trace, not just across sibling top-level ANDs."""

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

    safe, hazard, collision = Atom("safe"), Atom("hazard"), Atom("collision")
    response_ok = within(persist(safe, h_steps), k_steps)
    return binary(
        Op.AND,
        unary(Op.GLOBALLY, unary(Op.NOT, collision)),
        unary(Op.GLOBALLY, binary(Op.IMPLIES, hazard, response_ok)),
    )


def _mostly_safe_trace(length):
    """One trigger at t=0, resolved and stable for the rest of the trace --
    an ordinary, unremarkable "policy behaved fine" episode, not a crafted
    adversarial one."""

    labels = [{"hazard"}] + [{"safe"}] * (length - 1)
    return LabeledTrace(labels)


def branching_obligation_certificate(obligation_count=4, trace_length=40):
    formula = parse_formula(
        " and ".join(f"F({chr(ord('a') + index)})" for index in range(obligation_count))
    )
    labels = {chr(ord("a") + index) for index in range(obligation_count)}
    return evaluate_with_provenance(formula, LabeledTrace([labels] * trace_length))


def test_exact_mode_bounds_realistic_cartesian_product_at_output_limit():
    """Raw certificate queries are exact-until-a-declared-hard-bound by default."""
    certificate = branching_obligation_certificate()

    started = time.perf_counter()
    enumeration = certificate.minimal_supports(limit=10_000)
    elapsed = time.perf_counter() - started

    assert elapsed < 5.0
    assert len(enumeration.supports) == 10_000
    assert not enumeration.complete
    assert enumeration.mode is SupportEnumerationMode.EXACT
    assert enumeration.intermediate_limit is None
    assert enumeration.truncation_reasons == ("output_limit",)
    assert enumeration.truncated_node_ids
    assert enumeration.work_done <= enumeration.work_limit == 100_000
    assert {len(support) for support in enumeration.supports} == {4}


def test_bounded_partial_mode_is_explicit_and_records_intermediate_truncation():
    certificate = branching_obligation_certificate()

    enumeration = certificate.minimal_supports(
        limit=10_000,
        mode=SupportEnumerationMode.BOUNDED_PARTIAL,
    )

    assert 0 < len(enumeration.supports) < 10_000
    assert not enumeration.complete
    assert enumeration.mode is SupportEnumerationMode.BOUNDED_PARTIAL
    assert enumeration.intermediate_limit == 16
    assert enumeration.truncation_reasons == ("intermediate_limit",)
    assert enumeration.truncated_node_ids
    assert {len(support) for support in enumeration.supports} == {4}


def test_raising_partial_intermediate_limit_reaches_requested_output_limit():
    certificate = branching_obligation_certificate()

    enumeration = certificate.minimal_supports(
        limit=10_000,
        work_limit=2_000_000,
        mode=SupportEnumerationMode.BOUNDED_PARTIAL,
        intermediate_limit=10_000,
    )

    assert len(enumeration.supports) == 10_000
    assert not enumeration.complete
    assert enumeration.intermediate_limit == 10_000
    assert enumeration.truncation_reasons == ("output_limit",)


def test_tiny_work_limit_stops_before_intermediate_product_expansion():
    certificate = branching_obligation_certificate()

    enumeration = certificate.minimal_supports(limit=10_000, work_limit=50)

    assert not enumeration.complete
    assert enumeration.supports == ()
    assert enumeration.truncation_reasons == ("work_limit",)
    assert enumeration.truncated_node_ids
    assert enumeration.work_done == enumeration.work_limit == 50


def test_complete_antichain_is_retained_when_full_expansion_fits_both_bounds():
    certificate = branching_obligation_certificate(obligation_count=3, trace_length=3)

    enumeration = certificate.minimal_supports(limit=100, work_limit=10_000)

    assert enumeration.complete
    assert enumeration.truncation_reasons == ()
    assert len(enumeration.supports) == 27
    assert {len(support) for support in enumeration.supports} == {3}
    assert all(support_entails_verdict(certificate, support) for support in enumeration.supports)


def test_verification_report_exposes_mode_and_every_active_bound():
    certificate = branching_obligation_certificate()

    report = verify_certificate(certificate, support_limit=500, support_work_limit=10_000)

    assert report.valid
    assert report.checked_supports == 500
    assert not report.support_enumeration_complete
    assert report.support_enumeration_mode is SupportEnumerationMode.BOUNDED_PARTIAL
    assert report.support_intermediate_limit == 16
    assert report.support_truncation_reasons == ("intermediate_limit", "output_limit")
    assert report.support_truncated_node_ids
    assert report.support_work_done <= report.support_work_limit == 10_000


def test_verify_certificate_intermediate_limit_reaches_declared_support_limit():
    """Explicitly overriding `support_intermediate_limit` through
    `verify_certificate` reaches the full declared `support_limit`, matching
    the raw circuit API's tunability."""
    certificate = branching_obligation_certificate()

    report = verify_certificate(
        certificate,
        support_limit=500,
        support_work_limit=100_000,
        support_enumeration_mode=SupportEnumerationMode.BOUNDED_PARTIAL,
        support_intermediate_limit=500,
    )

    assert report.valid
    assert report.checked_supports == 500
    assert not report.support_enumeration_complete
    assert report.support_intermediate_limit == 500
    assert report.support_truncation_reasons == ("output_limit",)


def test_complete_enumerations_match_independent_bruteforce_circuit_oracle():
    cases = (
        ("p or (p and q)", [{"p", "q"}]),
        ("(p and q) or p", [{"p", "q"}]),
        ("F(p or q) and F(r or s)", [set(), {"p", "q", "r", "s"}]),
        ("G(p -> F(q or r))", [{"p"}, {"q", "r"}, set()]),
        ("(p U q) or (r W s)", [{"p", "r"}, {"q", "s"}]),
    )
    for source, labels in cases:
        certificate = evaluate_with_provenance(parse_formula(source), LabeledTrace(labels))
        expected = _bruteforce_minimal_supports(certificate.circuit)
        actual = certificate.minimal_supports(limit=10_000, work_limit=100_000)
        assert actual.complete, source
        assert frozenset(actual.supports) == expected, source


def test_bounded_response_gfold_over_ordinary_short_episode_is_fast_and_nonempty():
    """Regression for the discovery made against real cached Highway DQN
    rollouts: G(trigger -> F_leq_K(persist_H(...))) folded over an ordinary
    ~20-step episode used to exhaust the default work budget completely
    (0 supports) because every "boring" (non-triggered) G-level still offers
    a genuine antecedent-false/consequent-true branching alternative,
    compounding across the whole chain -- a fundamentally different shape
    from a handful of sibling top-level ANDs."""

    formula = _bounded_response_formula(k_steps=4, h_steps=2)
    trace = _mostly_safe_trace(length=21)
    certificate = evaluate_with_provenance(formula, trace)

    started = time.perf_counter()
    enumeration = certificate.minimal_supports(
        limit=10_000,
        mode=SupportEnumerationMode.BOUNDED_PARTIAL,
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 2.0
    assert len(enumeration.supports) > 0
    assert enumeration.work_done < 100_000


def test_bounded_response_gfold_scales_to_long_traces_without_default_tuning():
    """The same formula shape must stay tractable well past a 20-step
    episode -- other domains run far longer than Highway's ~20-40 steps.
    Confirms bounded-partial mode (`intermediate_limit=16`) generalizes rather than
    being tuned to fit only the one episode that surfaced the bug."""

    formula = _bounded_response_formula(k_steps=4, h_steps=2)
    for length in (100, 500, 1000):
        trace = _mostly_safe_trace(length=length)
        certificate = evaluate_with_provenance(formula, trace)
        enumeration = certificate.minimal_supports(
            limit=1000,
            mode=SupportEnumerationMode.BOUNDED_PARTIAL,
        )
        assert len(enumeration.supports) > 0, length
        assert enumeration.work_done <= enumeration.work_limit, length


@pytest.mark.parametrize(
    ("source", "labels", "expected"),
    (
        ("F p", [{"p"}] * 40, 40),
        ("G(not p)", [{"p"}] * 40, 40),
        ("p U q", [{"p", "q"}] * 40, 40),
        ("p W q", [{"p", "q"}] * 40, 41),
    ),
)
def test_linear_temporal_alternatives_are_not_cut_by_partial_and_beam(source, labels, expected):
    certificate = evaluate_with_provenance(parse_formula(source), LabeledTrace(labels))
    for mode in (SupportEnumerationMode.EXACT, SupportEnumerationMode.BOUNDED_PARTIAL):
        enumeration = certificate.minimal_supports(limit=10_000, mode=mode)
        assert enumeration.complete, (source, mode)
        assert len(enumeration.supports) == expected, (source, mode)
        assert enumeration.truncation_reasons == ()


def test_exact_mode_rejects_hidden_intermediate_cap():
    certificate = branching_obligation_certificate(obligation_count=2, trace_length=2)
    with pytest.raises(ValueError, match="bounded_partial"):
        certificate.minimal_supports(intermediate_limit=16)


def test_overlap_counterexample_is_documented_as_partial_not_minimal():
    def fact(name):
        return EvidenceFact(FactKind.ATOM, 0, value=True, proposition=name)

    builder = CircuitBuilder()
    a = builder.evidence(fact("a"))
    b = builder.evidence(fact("b"))
    x = builder.evidence(fact("x"))
    y = builder.evidence(fact("y"))
    shared = builder.all_of((a, b))
    left = builder.any_of((x, shared))
    right = builder.any_of((y, shared))
    circuit = builder.freeze(builder.all_of((left, right)))

    partial = circuit.minimal_supports(limit=2)
    complete = circuit.minimal_supports(limit=3)

    assert not partial.complete
    assert partial.mode is SupportEnumerationMode.EXACT
    assert partial.truncation_reasons == ("output_limit",)
    assert complete.complete
    assert {
        frozenset(item.proposition for item in support) for support in complete.supports
    } == {frozenset({"a", "b"}), frozenset({"x", "y"})}


def test_fail_closed_audit_reports_the_actual_incomplete_bound():
    formula = parse_formula("F(a) and F(b) and F(c) and F(d)")
    trace = LabeledTrace([{"a", "b", "c", "d"}] * 40)
    with pytest.raises(RuntimeError, match="intermediate_limit") as error:
        audit_traces(
            formula,
            [trace],
            config=AuditConfig(
                require_grounding=False,
                support_limit=500,
                support_enumeration_mode=SupportEnumerationMode.BOUNDED_PARTIAL,
                require_complete_support_enumeration=True,
            ),
        )
    assert "nodes=" in str(error.value)


def _bruteforce_minimal_supports(circuit):
    memo = []
    for node in circuit.nodes:
        if node.kind is NodeKind.TRUE:
            supports = {frozenset()}
        elif node.kind is NodeKind.EVIDENCE:
            supports = {frozenset((node.evidence,))}
        elif node.kind is NodeKind.OR:
            supports = set().union(*(memo[child] for child in node.children))
        else:
            supports = {
                frozenset().union(*choices)
                for choices in product(*(memo[child] for child in node.children))
            }
        memo.append(
            {
                support
                for support in supports
                if not any(other < support for other in supports)
            }
        )
    return frozenset(memo[circuit.root])
