"""Regression coverage for the LetterEnv DQN confirmatory adapter.

Torch/DeepLTL are not installed in this lightweight test environment (see
``tlrl_benchmarks.letter_env.dqn``'s own docstring for the same convention),
so this file covers everything that does not require them: adapter/subject
resolution, the RABINIZER_JAR self-sufficiency check, formula-family
classification, and -- the most safety-critical new logic -- the campaign
engine's scenario_seed pairing. The torch-dependent session behavior
(sequence refresh, DQN value estimation, a complete episode) is covered by
``scripts/smoke_letter_env_confirmatory_session.py``, run under the
DeepLTL-compatible interpreter, matching this benchmark's existing
smoke/pytest split.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from tlrl_benchmarks.letter_env.confirmatory import (
    LetterEnvDQNConfirmatoryAdapter,
    _formula_family,
)
from tlrl_provenance.campaign import AuditSubject, AuditTask, build_plan


def _write(path: Path, payload: bytes = b"stub") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _adapter_config(root: Path, *, rabinizer_jar: str = "rabinizer.jar") -> dict:
    return {
        "upstream": ".",
        "rabinizer_jar": rabinizer_jar,
        "subjects": [
            {
                "subject_id": "dqn_a",
                "training_seed": 930001,
                "experiment": "exp",
                "checkpoint_steps": 100,
            }
        ],
    }


def test_formula_family_matches_deepltl_operator_grammar():
    assert _formula_family("F (a & F b)") == "eventually_chain"
    assert _formula_family("(!a U (b & (!c U d)))") == "until_chain"
    assert _formula_family("G a") == "other"


def test_adapter_resolves_subjects_and_sets_rabinizer_jar(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("RABINIZER_JAR", raising=False)
    root = tmp_path
    _write(root / "src" / "envs" / "__init__.py")
    _write(root / "rabinizer.jar", b"fake-jar")
    run = root / "experiments" / "dqn" / "LetterEnv-v0" / "exp" / "930001"
    _write(run / "eval" / "100.pth")
    _write(run / "vocab.pkl")

    adapter = LetterEnvDQNConfirmatoryAdapter(config=_adapter_config(root), root=root)

    assert [s.subject_id for s in adapter.subjects()] == ["dqn_a"]
    subject = adapter.subjects()[0]
    assert subject.metadata["algorithm"] == "double_dqn"
    assert subject.metadata["checkpoint_sha256"] == hashlib.sha256(b"stub").hexdigest()
    assert len(adapter.tasks()) == 50
    # Self-sufficiency: the adapter must not depend on the caller having
    # already configured the environment correctly.
    assert os.environ["RABINIZER_JAR"] == str((root / "rabinizer.jar").resolve())


def test_adapter_fails_closed_on_missing_checkpoint(tmp_path: Path):
    root = tmp_path
    _write(root / "src" / "envs" / "__init__.py")
    _write(root / "rabinizer.jar")
    # No checkpoint/vocab written.
    with pytest.raises(FileNotFoundError):
        LetterEnvDQNConfirmatoryAdapter(config=_adapter_config(root), root=root)


def test_adapter_fails_closed_on_missing_rabinizer_jar(tmp_path: Path):
    root = tmp_path
    _write(root / "src" / "envs" / "__init__.py")
    run = root / "experiments" / "dqn" / "LetterEnv-v0" / "exp" / "930001"
    _write(run / "eval" / "100.pth")
    _write(run / "vocab.pkl")
    # No rabinizer.jar written.
    with pytest.raises(FileNotFoundError, match="RABINIZER_JAR"):
        LetterEnvDQNConfirmatoryAdapter(config=_adapter_config(root), root=root)


def test_tasks_carry_official_canonical_world_index_as_informational_metadata(tmp_path: Path):
    root = tmp_path
    _write(root / "src" / "envs" / "__init__.py")
    _write(root / "rabinizer.jar")
    run = root / "experiments" / "dqn" / "LetterEnv-v0" / "exp" / "930001"
    _write(run / "eval" / "100.pth")
    _write(run / "vocab.pkl")

    adapter = LetterEnvDQNConfirmatoryAdapter(config=_adapter_config(root), root=root)
    by_id = {t.task_id: t for t in adapter.tasks()}
    assert by_id["letter_env_000"].metadata["official_canonical_world_index"] == 0
    assert by_id["letter_env_049"].metadata["official_canonical_world_index"] == 49


# --- scenario_seed pairing: the campaign-engine-level fix for cross-policy
# comparability. This is the most safety-critical new logic in this change,
# and it is fully testable without torch since campaign.py has no such
# dependency.


class _FakeAdapter:
    def __init__(self, num_subjects, num_tasks):
        self._subjects = tuple(AuditSubject(f"s{i}", {}) for i in range(num_subjects))
        self._tasks = tuple(AuditTask(f"t{i}", formula="F a") for i in range(num_tasks))

    def subjects(self):
        return self._subjects

    def tasks(self):
        return self._tasks

    def provenance(self):
        return {}


def _base_config(**overrides) -> dict:
    config = {
        "campaign_id": "test",
        "benchmark_id": "test",
        "adapter": "fake:adapter",
        "design": {"episodes_per_task": 3},
        "seed_namespace": {"base": 1000, "subject_stride": 100, "task_stride": 10, "episode_stride": 1},
        "outputs": {},
    }
    config.update(overrides)
    return config


def test_without_scenario_namespace_cells_have_no_scenario_seed_key():
    plan = build_plan(_base_config(), _FakeAdapter(3, 2))
    assert all("scenario_seed" not in cell for cell in plan["cells"])
    assert plan["scenario_namespace"] is None


def test_scenario_seed_is_shared_across_subjects_for_the_same_task_and_episode():
    config = _base_config(scenario_namespace={"base": 5000, "task_stride": 20, "episode_stride": 1})
    plan = build_plan(config, _FakeAdapter(3, 2))
    by_key = {(c["subject_id"], c["task_id"], c["episode_index"]): c["scenario_seed"] for c in plan["cells"]}
    for task_id in ("t0", "t1"):
        for episode_index in range(3):
            values = {by_key[(f"s{i}", task_id, episode_index)] for i in range(3)}
            assert len(values) == 1, f"scenario_seed must be identical across subjects for {task_id}/{episode_index}"


def test_scenario_seed_varies_by_task_and_episode_but_evaluation_seed_stays_unique():
    config = _base_config(scenario_namespace={"base": 5000, "task_stride": 20, "episode_stride": 1})
    plan = build_plan(config, _FakeAdapter(2, 2))
    scenario_seeds = {(c["task_id"], c["episode_index"]): c["scenario_seed"] for c in plan["cells"]}
    assert len(set(scenario_seeds.values())) == 2 * 3  # num_tasks * episodes_per_task, all distinct
    evaluation_seeds = [c["evaluation_seed"] for c in plan["cells"]]
    assert len(set(evaluation_seeds)) == len(evaluation_seeds)  # still globally unique


def test_scenario_namespace_rejects_negative_or_zero_episode_stride():
    with pytest.raises(ValueError, match="scenario namespace"):
        build_plan(
            _base_config(scenario_namespace={"base": 5000, "task_stride": 20, "episode_stride": 0}),
            _FakeAdapter(1, 1),
        )
    with pytest.raises(ValueError, match="scenario namespace"):
        build_plan(
            _base_config(scenario_namespace={"base": -1, "task_stride": 20}),
            _FakeAdapter(1, 1),
        )


def test_scenario_namespace_rejects_aliasing_between_task_episode_cells():
    with pytest.raises(ValueError, match="scenario namespace aliases"):
        build_plan(
            _base_config(scenario_namespace={"base": 5000, "task_stride": 2, "episode_stride": 1}),
            _FakeAdapter(2, 2),
        )
