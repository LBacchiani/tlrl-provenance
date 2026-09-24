from __future__ import annotations

import hashlib
import json

import pytest

from tlrl_benchmarks.flatworld import (
    ATOMS,
    CIRCLES,
    EXPECTED_MAX_STEPS,
    EXPECTED_TASK_COUNT,
    EXPECTED_TASKS_SHA256,
    EpisodeDisposition,
    FlatworldPositionStep,
    FlatworldStepRecord,
    FlatworldTask,
    assess_flatworld_episode,
    build_flatworld_trace,
    capture_official_step,
    flatworld_zone_rules,
    load_tasks,
    preserve_termination_and_truncation,
)
from tlrl_benchmarks.flatworld.adapter import tasks_path
from tlrl_provenance import PrefixStatus, parse_formula, verify_grounding_reconstruction


def step(*colors: str, action: int = 0, reward: float = 0.0) -> FlatworldStepRecord:
    """Build a step record whose position sits inside every requested circle.

    Only combinations that are actually simultaneously reachable in the
    frozen geometry are usable here (mirroring ``get_possible_assignments``):
    single circles, or ``blue``+``green``, ``green``+``aqua``, ``blue``+
    ``aqua``, or all three together, or ``red``+``magenta``.
    """

    # Positions independently verified (by direct circle-containment search,
    # not by inspection) to ground to exactly the requested proposition set
    # under the frozen geometry -- several circle centers themselves lie
    # inside a different-colored circle (e.g. aqua's center is inside blue's
    # circle), so a "pure" single-color point cannot always be a center.
    single_color_positions = {
        "aqua": (0.9702791534208636, 0.08198763564735649),
        "red": (-1.4187224366781153, 0.616705631564025),
        "magenta": (-0.6918871148046649, 1.1046868558173903),
        "yellow": (-1.1308972933601777, -1.0465174775056656),
        "orange": (-1.454243842127788, -0.659675941528038),
        "blue": (-0.2037638890890678, 0.36773079721620583),
        "green": (1.0988382879679934, 0.8839839319154412),
    }
    if not colors:
        # A point far outside every circle but inside the domain bound.
        position = (-2.3, -2.3)
    elif len(colors) == 1 and colors[0] in single_color_positions:
        position = single_color_positions[colors[0]]
    else:
        raise ValueError(f"no known reachable position for combination {colors}")
    geometry = FlatworldPositionStep(position)
    reported = geometry.grounded_propositions
    assert reported == frozenset(colors), (reported, colors)
    return FlatworldStepRecord(geometry, action=action, reward=reward, reported_propositions=reported)


def task(source: str) -> FlatworldTask:
    return FlatworldTask("test", source, parse_formula(source))


def test_official_task_corpus_is_frozen_complete_and_parseable():
    payload = tasks_path().read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(payload).hexdigest() == EXPECTED_TASKS_SHA256
    tasks = load_tasks()
    assert len(tasks) == EXPECTED_TASK_COUNT
    assert [item.task_id for item in tasks] == [f"flatworld_{index:03d}" for index in range(EXPECTED_TASK_COUNT)]
    assert frozenset(atom for item in tasks for atom in item.formula.atoms()) <= ATOMS
    assert sum(item.source.lstrip().startswith("F") for item in tasks) == 23
    assert sum(" U " in item.source for item in tasks) == 26


def test_task_loader_rejects_corpus_drift(tmp_path):
    changed = tmp_path / "tasks.txt"
    changed.write_text(tasks_path().read_text(encoding="utf-8").replace("green", "azure", 1))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_tasks(changed)


def test_circle_geometry_has_nine_circles_covering_every_atom():
    assert len(CIRCLES) == 9
    assert frozenset(color for color, _, _, _ in CIRCLES) == ATOMS
    colors = [color for color, _, _, _ in CIRCLES]
    assert colors.count("red") == 2
    assert colors.count("green") == 2


def test_grounding_allows_multiple_simultaneous_propositions():
    single = FlatworldPositionStep((0.1, 0.0))  # blue circle center
    assert single.grounded_propositions == frozenset({"blue"})
    triple = FlatworldPositionStep((0.75, 0.35))
    assert triple.grounded_propositions == frozenset({"blue", "green", "aqua"})


def test_position_step_rejects_nonfinite_and_out_of_bounds_coordinates():
    with pytest.raises(ValueError, match="finite"):
        FlatworldPositionStep((float("nan"), 0.0))
    with pytest.raises(ValueError, match="exceeds"):
        FlatworldPositionStep((10.0, 0.0))


def test_reported_and_grounded_propositions_must_agree():
    record = step("blue")
    record.require_reported_agreement()
    bad = FlatworldStepRecord(FlatworldPositionStep((0.1, 0.0)), reported_propositions=frozenset())
    with pytest.raises(AssertionError, match="disagrees"):
        bad.require_reported_agreement()


def test_trace_retains_transition_payload_and_replays_grounding():
    trace = build_flatworld_trace(
        [step(action=1), step("blue", action=3, reward=1.0)],
        policy_id="flatworld-policy-1",
        seed=1,
        task_id="flatworld_000",
    )
    assert trace.labels == (frozenset(), frozenset({"blue"}))
    assert trace.states[1].action == 3
    assert trace.states[1].reward == 1.0
    assert trace.metadata["label_timing"] == "post_action_before_ldba_transition"
    report = verify_grounding_reconstruction(trace, flatworld_zone_rules())
    assert report.valid
    assert report.checked_records == len(ATOMS) * 2


def test_official_success_and_violation_are_independently_verified():
    success_trace = build_flatworld_trace([step("blue")], terminated=True)
    success = assess_flatworld_episode(task("F blue"), success_trace, official_success=True)
    assert success.disposition is EpisodeDisposition.DECIDED_SATISFACTION
    assert success.prefix.status is PrefixStatus.DEFINITELY_SATISFIED
    assert not success.official_independent_mismatch

    violation_trace = build_flatworld_trace([step("blue")], terminated=True)
    violation = assess_flatworld_episode(
        task("not blue U aqua"), violation_trace, official_violation=True
    )
    assert violation.disposition is EpisodeDisposition.DECIDED_VIOLATION
    assert violation.prefix.status is PrefixStatus.DEFINITELY_VIOLATED
    assert not violation.official_independent_mismatch


def test_logically_satisfied_horizon_edge_is_not_mislabeled_censored():
    trace = build_flatworld_trace([step("blue"), step("aqua")], terminated=False, truncated=True)
    result = assess_flatworld_episode(task("F (blue & F aqua)"), trace)
    assert result.disposition is EpisodeDisposition.DECIDED_SATISFACTION
    assert result.certificate.satisfied
    assert result.prefix.status is PrefixStatus.DEFINITELY_SATISFIED
    assert result.official_independent_mismatch


def test_open_truncation_is_censored_and_open_termination_is_separate():
    truncated = build_flatworld_trace([step()], terminated=False, truncated=True)
    censored = assess_flatworld_episode(task("F blue"), truncated)
    assert censored.disposition is EpisodeDisposition.CENSORED
    assert censored.prefix.status is PrefixStatus.OPEN
    assert not censored.official_independent_mismatch

    terminated = build_flatworld_trace([step()], terminated=True)
    environment = assess_flatworld_episode(task("F blue"), terminated)
    assert environment.disposition is EpisodeDisposition.ENVIRONMENT_TERMINATION
    assert environment.prefix.status is PrefixStatus.OPEN
    assert not environment.official_independent_mismatch


def test_missing_official_violation_is_retained_as_a_mismatch():
    trace = build_flatworld_trace([step("blue")], terminated=True)
    result = assess_flatworld_episode(task("not blue U aqua"), trace)
    assert result.disposition is EpisodeDisposition.DECIDED_VIOLATION
    assert result.official_independent_mismatch


def test_fast_mode_requires_an_asserted_terminal_outcome():
    trace = build_flatworld_trace([step()], terminated=True)
    with pytest.raises(ValueError, match="asserted official terminal outcome"):
        assess_flatworld_episode(task("F blue"), trace, compute_prefix=False)


class FakeUnwrapped:
    def __init__(self, position):
        self.agent_pos = position


class FakeEnv:
    def __init__(self, position):
        self.unwrapped = FakeUnwrapped(position)


def test_capture_official_step_is_dependency_free_and_fail_closed():
    captured = capture_official_step(FakeEnv((0.1, 0.0)), {"propositions": {"blue"}}, action=2, reward=0.5)
    assert captured.geometry.grounded_propositions == frozenset({"blue"})
    with pytest.raises(KeyError, match="propositions"):
        capture_official_step(FakeEnv((0.1, 0.0)), {})
    with pytest.raises(AssertionError, match="disagrees"):
        capture_official_step(FakeEnv((0.1, 0.0)), {"propositions": {"aqua"}})


class TimeLimit:
    _max_episode_steps = EXPECTED_MAX_STEPS


class RemoveTruncWrapper:
    env = TimeLimit()


def test_auditable_execution_bypasses_only_the_done_collapsing_wrapper():
    assert preserve_termination_and_truncation(RemoveTruncWrapper())._max_episode_steps == EXPECTED_MAX_STEPS
    with pytest.raises(TypeError, match="RemoveTruncWrapper"):
        preserve_termination_and_truncation(TimeLimit())
    bad = RemoveTruncWrapper()
    bad.env = TimeLimit()
    bad.env._max_episode_steps = EXPECTED_MAX_STEPS - 1
    with pytest.raises(ValueError, match=f"must be {EXPECTED_MAX_STEPS}"):
        preserve_termination_and_truncation(bad)


def test_frozen_corpus_manifest_and_sidecar_agree_when_present():
    package = tasks_path().parent
    manifest_path = package / "corpus_manifest.json"
    sidecar_path = package / "corpus_manifest.json.sha256"
    if not manifest_path.exists() or not sidecar_path.exists():
        pytest.skip("corpus manifest is generated by scripts/freeze_flatworld_corpus.py")
    payload = manifest_path.read_bytes()
    expected = sidecar_path.read_text(encoding="ascii").split()[0]
    assert hashlib.sha256(payload).hexdigest() == expected
    manifest = json.loads(payload)
    assert manifest["formula_corpus"]["count"] == EXPECTED_TASK_COUNT
    assert manifest["circle_geometry"]["count"] == 9
    assert manifest["circle_geometry"]["matches_live_upstream_class_attribute"] is True
    assert manifest["outcomes_inspected"] is False
    assert manifest["upstream"]["repository"] == "https://github.com/mathiasj33/deep-ltl"

