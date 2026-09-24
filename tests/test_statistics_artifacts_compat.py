from pathlib import Path
from types import SimpleNamespace

import pytest

from tlrl_provenance import (
    GroundingRecord,
    LabeledTrace,
    aggregate_certificates,
    certificate_digest,
    evaluate_with_provenance,
    engine_source_digest,
    from_legacy_formula,
    from_v2_episode,
    occurrence_mutation_query,
    parse_formula,
    policy_cluster_bootstrap_event,
    policy_cluster_bootstrap_satisfaction,
    replay_certificate_artifact,
    write_certificate_artifact,
)
from tlrl_provenance.statistics import certificate_events


def grounded_trace(value, policy_id):
    labels = [{"p"}] if value else [set()]
    return LabeledTrace(
        labels,
        groundings=[GroundingRecord("p", 0, value, "p_rule", "1")],
        policy_id=policy_id,
    )


def test_profile_and_policy_cluster_bootstrap():
    formula = parse_formula("p")
    certificates = [
        evaluate_with_provenance(formula, grounded_trace(True, "a")),
        evaluate_with_provenance(formula, grounded_trace(True, "a")),
        evaluate_with_provenance(formula, grounded_trace(False, "b")),
    ]
    profile = aggregate_certificates(certificates)
    assert profile.trace_count == 3
    assert profile.policy_count == 2
    assert profile.satisfaction.value == pytest.approx(2 / 3)
    estimate = policy_cluster_bootstrap_satisfaction(certificates, bootstrap_samples=200, seed=4)
    assert estimate.value == pytest.approx(0.5)
    assert estimate.lower <= estimate.value <= estimate.upper
    event = next(key for key, _ in profile.event_prevalence if key.operator == "atom") if profile.event_prevalence else None
    # Atomic cells have no decision events; use a temporal formula below for event inference.
    assert event is None


def test_policy_cluster_bootstrap_for_semantic_event_normalizes_time():
    formula = parse_formula("F p")
    certificates = [
        evaluate_with_provenance(formula, LabeledTrace([{"p"}], policy_id="a")),
        evaluate_with_provenance(formula, LabeledTrace([set(), {"p"}], policy_id="b")),
    ]
    profile = aggregate_certificates(certificates)
    event = next(
        key
        for key, _ in profile.event_prevalence
        if key.kind == "eventuality_progress" and key.alternative == "witness_now"
    )
    estimate = policy_cluster_bootstrap_event(certificates, event, bootstrap_samples=100, seed=2)
    assert estimate.value == 1.0


def test_default_events_exclude_semantically_irrelevant_cells():
    certificate = evaluate_with_provenance(
        parse_formula("(p or q) and r"),
        LabeledTrace([{"p"}], policy_id="a"),
    )
    verdict_kinds = {event.kind for event in certificate_events(certificate)}
    all_kinds = {event.kind for event in certificate_events(certificate, scope="all_cells")}
    assert "satisfying_disjunct" not in verdict_kinds
    assert "satisfying_disjunct" in all_kinds


def test_artifact_is_deterministic_and_replayable(tmp_path: Path):
    certificate = evaluate_with_provenance(parse_formula("G p"), grounded_trace(True, "policy"))
    path = tmp_path / "certificate.json"
    digest = write_certificate_artifact(path, certificate)
    replayed = replay_certificate_artifact(path)
    assert replayed.verdict == certificate.verdict
    assert certificate_digest(replayed) == certificate_digest(certificate)
    assert len(digest) == 64
    assert len(engine_source_digest()) == 64
    path.write_text(path.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        replay_certificate_artifact(path)


def test_v1_formula_adapter_preserves_semantics():
    legacy_logic = pytest.importorskip("tlrl.logic", reason="optional v1 integration check")
    parse_v1_formula = legacy_logic.parse_formula
    legacy = parse_v1_formula("G(p -> F q)")
    converted = from_legacy_formula(legacy)
    assert converted == parse_formula("G(p -> F q)")


@pytest.mark.parametrize("source", ["F(p or q)", "G(not p)", "p U q"])
def test_v1_verdict_and_pass_relevance_are_exactly_recovered(source):
    legacy_logic = pytest.importorskip("tlrl.logic", reason="optional v1 integration check")
    parse_v1_formula = legacy_logic.parse_formula
    iter_v1_occurrences = legacy_logic.iter_occurrences
    v1_trace_relevance = legacy_logic.trace_relevance
    legacy = parse_v1_formula(source)
    converted = from_legacy_formula(legacy)
    v3_occurrences = {item.path: item.id for item in converted_index(converted).occurrences}
    valuations = (set(), {"p"}, {"q"}, {"p", "q"})
    for first in valuations:
        for second in valuations:
            labels = [first, second]
            trace = LabeledTrace(labels)
            assert legacy.evaluate(labels) == evaluate_with_provenance(converted, trace).satisfied
            if not legacy.evaluate(labels):
                continue
            for old_occurrence in iter_v1_occurrences(legacy):
                query = occurrence_mutation_query(
                    converted,
                    trace,
                    v3_occurrences[old_occurrence.path],
                )
                assert query.pass_necessary == v1_trace_relevance(
                    legacy,
                    old_occurrence.id,
                    labels,
                )


def test_v2_trace_adapter_copies_only_declared_observable_fields():
    steps = [
        SimpleNamespace(labels={"p"}, state={"x": 1}, observation=(1.0,)),
        SimpleNamespace(labels=set(), state={"x": 2}, observation=(2.0,)),
    ]
    legacy = SimpleNamespace(
        steps=steps,
        actions=(1,),
        rewards=(0.5,),
        policy_id="policy",
        environment_id="environment",
        seed=7,
        metadata={"terminated": False, "truncated": True, "campaign": "development"},
    )

    converted = from_v2_episode(legacy)

    assert converted.labels == (frozenset({"p"}), frozenset())
    assert converted.states == ({"x": 1}, {"x": 2})
    assert converted.observations == ((1.0,), (2.0,))
    assert converted.actions == (1,)
    assert converted.rewards == (0.5,)
    assert converted.policy_id == "policy"
    assert converted.environment_id == "environment"
    assert converted.seed == 7
    assert not converted.terminated
    assert converted.truncated
    assert converted.metadata["campaign"] == "development"


def converted_index(formula):
    from tlrl_provenance.logic import IndexedFormula

    return IndexedFormula.build(formula)
