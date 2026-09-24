from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tlrl_benchmarks.deepltl_point import (
    ATOMS,
    EXPECTED_TASKS_SHA256,
    EpisodeDisposition,
    PointStepRecord,
    PointTask,
    PointZoneStep,
    ZoneGeometry,
    assess_point_episode,
    build_point_trace,
    capture_official_step,
    load_tasks,
    point_zone_rules,
    preserve_termination_and_truncation,
)
from tlrl_benchmarks.deepltl_point.adapter import tasks_path
from tlrl_provenance import PrefixStatus, parse_formula, verify_grounding_reconstruction


def zones(*, green=(0.0, 0.0)):
    return (
        ZoneGeometry("blue", ((-4.0, -4.0), (-4.0, -2.0))),
        ZoneGeometry("green", (green, (0.0, 2.0))),
        ZoneGeometry("magenta", ((2.0, -4.0), (2.0, -2.0))),
        ZoneGeometry("yellow", ((4.0, 2.0), (4.0, 4.0))),
    )


def record(xy, reported, *, action=None, reward=0.0):
    return PointStepRecord(
        PointZoneStep(xy, zones(), frozenset(reported)),
        action=action,
        reward=reward,
    )


def task(source):
    return PointTask("test", source, parse_formula(source))


def test_official_task_corpus_is_frozen_complete_and_directly_parseable():
    payload = tasks_path().read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(payload).hexdigest() == EXPECTED_TASKS_SHA256
    tasks = load_tasks()
    assert len(tasks) == 50
    assert [item.task_id for item in tasks] == [f"deepltl_point_{i:03d}" for i in range(50)]
    assert frozenset(atom for item in tasks for atom in item.formula.atoms()) == ATOMS
    assert sum(item.source.lstrip().startswith("F") for item in tasks) == 20
    assert sum(" U " in item.source for item in tasks) == 30


def test_task_loader_rejects_even_semantically_small_corpus_drift(tmp_path):
    changed = tmp_path / "tasks.txt"
    changed.write_text(tasks_path().read_text(encoding="utf-8").replace("green", "blue", 1))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_tasks(changed)


def test_geometry_rule_includes_exact_boundary_and_excludes_outside():
    boundary = PointZoneStep((0.4, 0.0), zones(), frozenset({"green"}))
    outside = PointZoneStep((0.4000001, 0.0), zones(), frozenset())
    assert boundary.grounded_propositions == frozenset({"green"})
    assert outside.grounded_propositions == frozenset()


def test_geometry_rejects_layout_that_breaks_zero_or_one_assignment_contract():
    overlapping = list(zones())
    overlapping[-1] = ZoneGeometry("yellow", ((0.7, 0.0), (4.0, 4.0)))
    with pytest.raises(ValueError, match="zero-or-one-proposition"):
        PointZoneStep((0.0, 0.0), tuple(overlapping), frozenset({"green"}))


def test_official_label_mismatch_fails_before_semantic_evaluation():
    bad = PointStepRecord(PointZoneStep((0.0, 0.0), zones(), frozenset()))
    with pytest.raises(AssertionError, match="disagrees with independent geometry"):
        build_point_trace([bad])


def test_trace_retains_post_action_payload_and_replays_all_groundings():
    trace = build_point_trace(
        [
            record((1.0, 1.0), set(), action=(0.1, 0.2), reward=0.0),
            record((0.0, 0.0), {"green"}, action=(0.0, 0.0), reward=1.0),
        ],
        policy_id="ppo-seed-1",
        seed=1,
        task_id="deepltl_point_000",
    )
    assert trace.labels == (frozenset(), frozenset({"green"}))
    assert not trace.actions
    assert trace.states[1].action == (0.0, 0.0)
    assert trace.metadata["label_timing"] == "post_action_before_ldba_transition"
    report = verify_grounding_reconstruction(trace, point_zone_rules())
    assert report.valid
    assert report.checked_records == 8


def test_official_success_is_independently_decided_and_verified():
    trace = build_point_trace(
        [record((1.0, 1.0), set()), record((0.0, 0.0), {"green"})],
        terminated=True,
    )
    result = assess_point_episode(task("F green"), trace, official_success=True)
    assert result.disposition is EpisodeDisposition.DECIDED_SATISFACTION
    assert result.certificate.satisfied
    assert result.prefix.status is PrefixStatus.DEFINITELY_SATISFIED


def test_official_until_violation_is_independently_decided_and_verified():
    trace = build_point_trace([record((0.0, 0.0), {"green"})], terminated=True)
    result = assess_point_episode(
        task("not green U yellow"),
        trace,
        official_violation=True,
    )
    assert result.disposition is EpisodeDisposition.DECIDED_VIOLATION
    assert not result.certificate.satisfied
    assert result.prefix.status is PrefixStatus.DEFINITELY_VIOLATED


def test_truncated_unfulfilled_eventuality_is_censored_not_a_violation():
    trace = build_point_trace(
        [record((1.0, 1.0), set()), record((1.1, 1.1), set())],
        terminated=False,
        truncated=True,
    )
    result = assess_point_episode(task("F yellow"), trace)
    assert result.disposition is EpisodeDisposition.CENSORED
    assert not result.decided
    assert not result.certificate.satisfied
    assert result.prefix.status is PrefixStatus.OPEN


def test_non_tl_environment_termination_stays_separate():
    trace = build_point_trace([record((1.0, 1.0), set())], terminated=True)
    result = assess_point_episode(task("F yellow"), trace)
    assert result.disposition is EpisodeDisposition.ENVIRONMENT_TERMINATION
    assert not result.decided


def test_missing_official_violation_is_promoted_by_independent_prefix():
    trace = build_point_trace([record((0.0, 0.0), {"green"})], terminated=True)
    result = assess_point_episode(task("not green U yellow"), trace)
    assert result.disposition is EpisodeDisposition.DECIDED_VIOLATION
    assert result.prefix.status is PrefixStatus.DEFINITELY_VIOLATED


def test_missing_official_satisfaction_is_promoted_by_independent_prefix():
    trace = build_point_trace([record((0.0, 0.0), {"green"})], terminated=True)
    result = assess_point_episode(task("F green"), trace)
    assert result.disposition is EpisodeDisposition.DECIDED_SATISFACTION
    assert result.prefix.status is PrefixStatus.DEFINITELY_SATISFIED


def test_fast_mode_requires_an_asserted_official_outcome():
    trace = build_point_trace([record((1.0, 1.0), set())], terminated=True)
    with pytest.raises(ValueError, match="compute_prefix=False"):
        assess_point_episode(task("F yellow"), trace, compute_prefix=False)

    decided = build_point_trace(
        [record((0.0, 0.0), {"green"})],
        terminated=True,
    )
    result = assess_point_episode(
        task("F green"),
        decided,
        official_success=True,
        compute_prefix=False,
    )
    assert result.disposition is EpisodeDisposition.DECIDED_SATISFACTION


def test_inconsistent_official_terminal_flags_fail_closed():
    trace = build_point_trace([record((1.0, 1.0), set())], terminated=True)
    with pytest.raises(AssertionError, match="official success disagrees"):
        assess_point_episode(task("F yellow"), trace, official_success=True)
    with pytest.raises(ValueError, match="both"):
        assess_point_episode(
            task("F yellow"), trace, official_success=True, official_violation=True
        )


class FakeAgent:
    pos = (0.0, 0.0, 0.25)


class FakeGeom:
    def __init__(self, color, centers):
        self.color_name = color
        self.size = 0.4
        self.pos = [(*center, 0.02) for center in centers]


class FakeTask:
    agent = FakeAgent()
    _geoms = {
        "blue": FakeGeom("blue", ((-4.0, -4.0), (-4.0, -2.0))),
        "green": FakeGeom("green", ((0.0, 0.0), (0.0, 2.0))),
        "magenta": FakeGeom("magenta", ((2.0, -4.0), (2.0, -2.0))),
        "yellow": FakeGeom("yellow", ((4.0, 2.0), (4.0, 4.0))),
        "wall": object(),
    }


class FakeUnwrapped:
    task = FakeTask()


class FakeEnv:
    unwrapped = FakeUnwrapped()


def test_official_environment_capture_is_dependency_free_and_fail_closed():
    captured = capture_official_step(
        FakeEnv(), {"propositions": {"green"}}, action=(0.2, -0.1), reward=0.5
    )
    assert captured.geometry.grounded_propositions == frozenset({"green"})
    assert captured.action == (0.2, -0.1)
    with pytest.raises(KeyError, match="propositions"):
        capture_official_step(FakeEnv(), {})


class TimeLimit:
    _max_episode_steps = 1000


class RemoveTruncWrapper:
    env = TimeLimit()


def test_auditable_execution_bypasses_only_the_known_done_collapsing_wrapper():
    assert preserve_termination_and_truncation(RemoveTruncWrapper())._max_episode_steps == 1000
    with pytest.raises(TypeError, match="RemoveTruncWrapper"):
        preserve_termination_and_truncation(TimeLimit())
    bad = RemoveTruncWrapper()
    bad.env._max_episode_steps = 999
    with pytest.raises(ValueError, match="must be 1000"):
        preserve_termination_and_truncation(bad)


def test_frozen_manifest_and_sidecar_agree_when_present():
    package = tasks_path().parent
    manifest_path = package / "manifest.json"
    sidecar_path = package / "manifest.json.sha256"
    if not manifest_path.exists() or not sidecar_path.exists():
        pytest.skip("manifest is generated after adapter tests")
    payload = manifest_path.read_bytes()
    expected = sidecar_path.read_text(encoding="ascii").split()[0]
    assert hashlib.sha256(payload).hexdigest() == expected
    manifest = json.loads(payload)
    assert manifest["contamination_controls"]["v2_dependency"] is False
    assert manifest["contamination_controls"]["utility_outcomes_inspected"] is False
    assert manifest["formula_corpus"]["count"] == 50
