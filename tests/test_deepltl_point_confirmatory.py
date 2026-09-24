from __future__ import annotations

import pytest

from tlrl_benchmarks.deepltl_point.confirmatory import (
    _retained_trace_payload,
    compare_official_verdict,
)
from tlrl_benchmarks.deepltl_point import build_point_trace
from test_deepltl_point_benchmark import record


@pytest.mark.parametrize(
    ("success", "violation", "verdict", "disposition", "official", "mismatch"),
    [
        (True, False, "satisfied", "decided_satisfaction", "satisfied", False),
        (False, True, "violated", "decided_violation", "violated", False),
        # Both official flags false means undecided, not an asserted negation.
        (False, False, "violated", "censored", "undecided", False),
        (False, False, "violated", "environment_termination", "undecided", False),
        (False, False, "satisfied", "environment_termination", "undecided", False),
        # An adapter cannot silently invent a decided disposition without an
        # official terminal flag.
        (False, False, "violated", "decided_violation", "undecided", True),
        (True, False, "violated", "decided_satisfaction", "satisfied", True),
        (False, True, "satisfied", "decided_violation", "violated", True),
    ],
)
def test_official_verdict_is_tri_state(
    success: bool,
    violation: bool,
    verdict: str,
    disposition: str,
    official: str,
    mismatch: bool,
) -> None:
    assert compare_official_verdict(
        official_success=success,
        official_violation=violation,
        independent_verdict=verdict,
        disposition=disposition,
    ) == (official, mismatch)


def test_contradictory_official_flags_fail_closed() -> None:
    with pytest.raises(ValueError, match="cannot both"):
        compare_official_verdict(
            official_success=True,
            official_violation=True,
            independent_verdict="satisfied",
            disposition="decided_satisfaction",
        )


def test_retained_trace_payload_is_compact_complete_and_deterministic() -> None:
    trace = build_point_trace(
        [
            record((1.0, 1.0), set(), action=[0.1, 0.2], reward=0.0),
            record((0.0, 0.0), {"green"}, action=[0.0, 0.0], reward=1.0),
        ],
        terminated=True,
    )
    payload, digest = _retained_trace_payload(trace)
    second_payload, second_digest = _retained_trace_payload(trace)
    assert payload == second_payload
    assert digest == second_digest
    assert len(payload["zone_layout"]) == 4
    assert payload["agent_xy"] == [[1.0, 1.0], [0.0, 0.0]]
    assert payload["reported_labels"] == [[], ["green"]]
    assert payload["actions"] == [[0.1, 0.2], [0.0, 0.0]]
    assert payload["rewards"] == [0.0, 1.0]
