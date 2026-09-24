from dataclasses import dataclass

import pytest

from tlrl_provenance import (
    AuditConfig,
    PropositionRule,
    audit_traces,
    ground_raw_steps,
    verify_grounding_reconstruction,
    parse_formula,
)


@dataclass(frozen=True)
class State:
    speed: float
    crashed: bool


def crashed(state, time):
    return state.crashed


def fast(state, time):
    return state.speed >= 10.0


RULES = (
    PropositionRule("crash", "crash_rule", "1", crashed, ("crashed",)),
    PropositionRule("fast", "speed_rule", "2", fast, ("speed",)),
)


def test_rules_generate_complete_labels_and_replay_independently():
    trace = ground_raw_steps(
        [State(5.0, False), State(12.0, True)],
        RULES,
        policy_id="policy",
    )
    assert trace.labels == (frozenset(), frozenset({"fast", "crash"}))
    assert trace.grounding_coverage(("fast", "crash")) == (4, 4)
    report = verify_grounding_reconstruction(trace, RULES)
    assert report.valid
    assert report.checked_records == 4
    audit = audit_traces(
        parse_formula("G(not crash) or F fast"),
        [trace],
        config=AuditConfig(require_grounding=True),
        grounding_rules=RULES,
    )
    assert audit.grounding_verification[0].valid


def test_replay_detects_different_rule_version_or_raw_data():
    trace = ground_raw_steps([State(12.0, False)], RULES)
    changed = (
        RULES[0],
        PropositionRule("fast", "speed_rule", "3", fast, ("speed",)),
    )
    report = verify_grounding_reconstruction(trace, changed)
    assert not report.valid
    assert "provenance mismatch" in " ".join(report.errors)


def test_rule_registry_rejects_non_boolean_outputs_and_duplicates():
    bad = PropositionRule("bad", "bad_rule", "1", lambda state, time: 1)
    with pytest.raises(TypeError, match="expected bool"):
        ground_raw_steps([State(0.0, False)], [bad])
    with pytest.raises(ValueError, match="unique proposition"):
        ground_raw_steps([State(0.0, False)], [RULES[0], RULES[0]])
