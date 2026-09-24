from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import pytest

from tlrl_benchmarks.letter_env import (
    ATOMS,
    EXPECTED_LETTER_COPIES,
    EXPECTED_OCCUPIED_CELLS,
    EXPECTED_TASKS_SHA256,
    EpisodeDisposition,
    LetterGridStep,
    LetterStepRecord,
    LetterTask,
    assess_letter_episode,
    build_letter_trace,
    canonical_world_digest,
    capture_official_step,
    load_tasks,
    load_world_layout,
    preserve_termination_and_truncation,
    validate_world_layout,
    world_path,
)
from tlrl_benchmarks.letter_env.adapter import tasks_path
from tlrl_provenance import PrefixStatus, parse_formula, verify_grounding_reconstruction
from tlrl_benchmarks.letter_env import letter_grid_rules


def official_layout() -> dict[tuple[int, int], str]:
    cells = [(row, column) for row in range(7) for column in range(7) if (row, column) != (0, 0)]
    letters = [letter for letter in sorted(ATOMS) for _ in range(EXPECTED_LETTER_COPIES)]
    return dict(zip(cells[:EXPECTED_OCCUPIED_CELLS], letters, strict=True))


def step(letter: str | None, *, action: int = 0, reward: float = 0.0) -> LetterStepRecord:
    layout = official_layout()
    if letter is None:
        position = (0, 0)
        reported = frozenset()
    else:
        position = next(cell for cell, value in layout.items() if value == letter)
        reported = frozenset({letter})
    return LetterStepRecord(
        LetterGridStep(position, layout, reported),
        action=action,
        reward=reward,
    )


def task(source: str) -> LetterTask:
    return LetterTask("test", source, parse_formula(source), 0)


def test_official_task_corpus_is_frozen_complete_and_parseable():
    payload = tasks_path().read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(payload).hexdigest() == EXPECTED_TASKS_SHA256
    tasks = load_tasks()
    assert len(tasks) == 50
    assert [item.task_id for item in tasks] == [f"letter_env_{index:03d}" for index in range(50)]
    assert [item.world_index for item in tasks] == list(range(50))
    assert frozenset(atom for item in tasks for atom in item.formula.atoms()) == ATOMS
    assert sum(item.source.lstrip().startswith("F") for item in tasks) == 23
    assert sum(" U " in item.source for item in tasks) == 27


def test_task_loader_rejects_corpus_drift(tmp_path: Path):
    changed = tmp_path / "tasks.txt"
    changed.write_text(tasks_path().read_text(encoding="utf-8").replace("g", "a", 1))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_tasks(changed)


def test_world_layout_requires_exactly_two_copies_of_every_letter():
    layout = official_layout()
    assert len(validate_world_layout(layout)) == 24
    missing = dict(layout)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="24 occupied cells"):
        validate_world_layout(missing)
    wrong_counts = dict(layout)
    a_cell = next(cell for cell, value in wrong_counts.items() if value == "a")
    wrong_counts[a_cell] = "b"
    with pytest.raises(ValueError, match="two copies|2 copies"):
        validate_world_layout(wrong_counts)


def test_world_layout_rejects_nonintegral_and_out_of_range_cells():
    layout = official_layout()
    value = layout.pop((0, 1))
    layout[(0.5, 1)] = value
    with pytest.raises(ValueError, match="integral"):
        validate_world_layout(layout)
    layout = official_layout()
    value = layout.pop((0, 1))
    layout[(7, 1)] = value
    with pytest.raises(ValueError, match="outside"):
        validate_world_layout(layout)


def test_world_path_is_bounded_and_world_loader_validates_pickle(tmp_path: Path):
    directory = tmp_path / ".runtime" / "deep-ltl" / "eval_datasets" / "LetterEnv-v0" / "worlds"
    directory.mkdir(parents=True)
    path = directory / "world_info_0.pkl"
    with path.open("wb") as handle:
        pickle.dump(official_layout(), handle)
    loaded = load_world_layout(tmp_path, 0)
    assert loaded == dict(sorted(official_layout().items()))
    assert canonical_world_digest(loaded) == canonical_world_digest(dict(reversed(list(loaded.items()))))
    assert world_path(tmp_path, 0) == path.resolve()
    with pytest.raises(ValueError, match="world index"):
        world_path(tmp_path, 50)
    with pytest.raises(TypeError, match="world index"):
        world_path(tmp_path, True)


def test_grid_lookup_and_reported_label_must_agree():
    record = step("a")
    assert record.geometry.grounded_propositions == frozenset({"a"})
    bad = LetterGridStep(record.geometry.agent_pos, official_layout(), frozenset())
    with pytest.raises(AssertionError, match="disagrees"):
        bad.require_reported_agreement()


def test_trace_retains_transition_payload_and_replays_grounding():
    trace = build_letter_trace(
        [step(None, action=1), step("a", action=3, reward=1.0)],
        policy_id="dqn-seed-1",
        seed=1,
        task_id="letter_env_000",
    )
    assert trace.labels == (frozenset(), frozenset({"a"}))
    assert trace.states[1].action == 3
    assert trace.states[1].reward == 1.0
    assert trace.metadata["label_timing"] == "post_action_before_ldba_transition"
    report = verify_grounding_reconstruction(trace, letter_grid_rules())
    assert report.valid
    assert report.checked_records == 24


def test_official_success_and_violation_are_independently_verified():
    success_trace = build_letter_trace([step("a")], terminated=True)
    success = assess_letter_episode(task("F a"), success_trace, official_success=True)
    assert success.disposition is EpisodeDisposition.DECIDED_SATISFACTION
    assert success.prefix.status is PrefixStatus.DEFINITELY_SATISFIED
    assert not success.official_independent_mismatch

    violation_trace = build_letter_trace([step("a")], terminated=True)
    violation = assess_letter_episode(
        task("not a U b"), violation_trace, official_violation=True
    )
    assert violation.disposition is EpisodeDisposition.DECIDED_VIOLATION
    assert violation.prefix.status is PrefixStatus.DEFINITELY_VIOLATED
    assert not violation.official_independent_mismatch


def test_logically_satisfied_horizon_edge_is_not_mislabeled_censored():
    trace = build_letter_trace([step("a"), step("b")], terminated=False, truncated=True)
    result = assess_letter_episode(task("F (a & F b)"), trace)
    assert result.disposition is EpisodeDisposition.DECIDED_SATISFACTION
    assert result.certificate.satisfied
    assert result.prefix.status is PrefixStatus.DEFINITELY_SATISFIED
    assert result.official_independent_mismatch


def test_open_truncation_is_censored_and_open_termination_is_separate():
    truncated = build_letter_trace([step(None)], terminated=False, truncated=True)
    censored = assess_letter_episode(task("F a"), truncated)
    assert censored.disposition is EpisodeDisposition.CENSORED
    assert censored.prefix.status is PrefixStatus.OPEN
    assert not censored.official_independent_mismatch

    terminated = build_letter_trace([step(None)], terminated=True)
    environment = assess_letter_episode(task("F a"), terminated)
    assert environment.disposition is EpisodeDisposition.ENVIRONMENT_TERMINATION
    assert environment.prefix.status is PrefixStatus.OPEN
    assert not environment.official_independent_mismatch


def test_missing_official_violation_is_retained_as_a_mismatch():
    trace = build_letter_trace([step("a")], terminated=True)
    result = assess_letter_episode(task("not a U b"), trace)
    assert result.disposition is EpisodeDisposition.DECIDED_VIOLATION
    assert result.official_independent_mismatch


def test_fast_mode_requires_an_asserted_terminal_outcome():
    trace = build_letter_trace([step(None)], terminated=True)
    with pytest.raises(ValueError, match="asserted official terminal outcome"):
        assess_letter_episode(task("F a"), trace, compute_prefix=False)


class FakeAgent:
    pass


class FakeUnwrapped:
    def __init__(self):
        self.map = official_layout()
        self.agent = next(cell for cell, value in self.map.items() if value == "a")


class FakeEnv:
    def __init__(self):
        self.unwrapped = FakeUnwrapped()


def test_capture_official_step_is_dependency_free_and_fail_closed():
    captured = capture_official_step(FakeEnv(), {"propositions": {"a"}}, action=2, reward=0.5)
    assert captured.geometry.grounded_propositions == frozenset({"a"})
    with pytest.raises(KeyError, match="propositions"):
        capture_official_step(FakeEnv(), {})
    with pytest.raises(TypeError, match="must be strings"):
        capture_official_step(FakeEnv(), {"propositions": {1}})


class TimeLimit:
    _max_episode_steps = 75


class RemoveTruncWrapper:
    env = TimeLimit()


def test_auditable_execution_bypasses_only_the_done_collapsing_wrapper():
    assert preserve_termination_and_truncation(RemoveTruncWrapper())._max_episode_steps == 75
    with pytest.raises(TypeError, match="RemoveTruncWrapper"):
        preserve_termination_and_truncation(TimeLimit())
    bad = RemoveTruncWrapper()
    bad.env = TimeLimit()
    bad.env._max_episode_steps = 74
    with pytest.raises(ValueError, match="must be 75"):
        preserve_termination_and_truncation(bad)


def test_frozen_corpus_manifest_and_sidecar_agree_when_present():
    package = tasks_path().parent
    manifest_path = package / "corpus_manifest.json"
    sidecar_path = package / "corpus_manifest.json.sha256"
    if not manifest_path.exists() or not sidecar_path.exists():
        pytest.skip("corpus manifest is generated after adapter tests")
    payload = manifest_path.read_bytes()
    expected = sidecar_path.read_text(encoding="ascii").split()[0]
    assert hashlib.sha256(payload).hexdigest() == expected
    manifest = json.loads(payload)
    assert manifest["formula_corpus"]["count"] == 50
    assert manifest["world_corpus"]["count"] == 50
    assert len(manifest["pairs"]) == 50
    assert manifest["outcomes_inspected"] is False
    assert manifest["upstream"]["repository"] == "https://github.com/mathiasj33/deep-ltl"
