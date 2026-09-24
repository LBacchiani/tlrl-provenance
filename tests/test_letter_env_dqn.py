from __future__ import annotations

import json
import random

import pytest

from scripts import train_letter_env_dqn as trainer
from tlrl_benchmarks.letter_env.dqn import (
    DQNConfig,
    EPSILON_ACTION,
    ReplayBuffer,
    Transition,
    action_index_to_env_action,
    choose_epsilon_greedy_index,
    env_action_to_action_index,
    epsilon_at_step,
    nonterminal_indices,
    valid_action_indices,
)


def test_dqn_config_rejects_invalid_training_protocol_values():
    with pytest.raises(ValueError, match="total_steps"):
        DQNConfig(total_steps=0)
    with pytest.raises(ValueError, match="batch_size cannot exceed"):
        DQNConfig(buffer_size=10, batch_size=11)
    with pytest.raises(ValueError, match="epsilon bounds"):
        DQNConfig(initial_epsilon=0.1, final_epsilon=0.2)
    with pytest.raises(ValueError, match="device"):
        DQNConfig(device="gpu")
    with pytest.raises(ValueError, match="replay_preprocessing"):
        DQNConfig(replay_preprocessing="sometimes")


def test_epsilon_schedule_is_linear_and_clamped():
    config = DQNConfig(initial_epsilon=1.0, final_epsilon=0.1, epsilon_decay_steps=100)
    assert epsilon_at_step(config, -1) == 1.0
    assert epsilon_at_step(config, 0) == 1.0
    assert epsilon_at_step(config, 50) == pytest.approx(0.55)
    assert epsilon_at_step(config, 100) == pytest.approx(0.1)
    assert epsilon_at_step(config, 1000) == pytest.approx(0.1)


def test_replay_buffer_is_bounded_and_samples_deterministically():
    first = ReplayBuffer(3, seed=7)
    second = ReplayBuffer(3, seed=7)
    transitions = [
        Transition(observation=index, action=index % 4, reward=float(index), next_observation=index + 1, done=False)
        for index in range(5)
    ]
    first.extend(transitions)
    second.extend(transitions)
    assert len(first) == 3
    assert first.sample(2) == second.sample(2)
    with pytest.raises(ValueError, match="cannot sample"):
        first.sample(4)


def test_action_mapping_keeps_epsilon_as_extra_q_category():
    assert valid_action_indices(4, epsilon_enabled=False) == (0, 1, 2, 3)
    assert valid_action_indices(4, epsilon_enabled=True) == (0, 1, 2, 3, 4)
    assert action_index_to_env_action(4, 4) == EPSILON_ACTION
    assert env_action_to_action_index(EPSILON_ACTION, 4) == 4
    assert action_index_to_env_action(2, 4) == 2
    assert env_action_to_action_index(2, 4) == 2
    with pytest.raises(ValueError, match="invalid action index"):
        action_index_to_env_action(5, 4)
    with pytest.raises(ValueError, match="invalid environment action"):
        env_action_to_action_index(9, 4)


def test_epsilon_greedy_never_selects_disabled_epsilon_greedily():
    rng = random.Random(1)
    action = choose_epsilon_greedy_index(
        [0.0, 1.0, 2.0, 3.0, 999.0],
        action_dim=4,
        epsilon_enabled=False,
        epsilon=0.0,
        rng=rng,
    )
    assert action == 3


def test_epsilon_greedy_can_select_enabled_epsilon():
    rng = random.Random(1)
    action = choose_epsilon_greedy_index(
        [0.0, 1.0, 2.0, 3.0, 999.0],
        action_dim=4,
        epsilon_enabled=True,
        epsilon=0.0,
        rng=rng,
    )
    assert action == 4


def test_transition_validates_action_and_reward():
    with pytest.raises(TypeError, match="action"):
        Transition(None, action=1.5, reward=0.0, next_observation=None, done=False)
    with pytest.raises(ValueError, match="reward"):
        Transition(None, action=1, reward=float("nan"), next_observation=None, done=False)


def test_nonterminal_indices_skip_terminal_next_observations():
    batch = [
        Transition("obs0", action=0, reward=0.0, next_observation="live0", done=False),
        Transition("obs1", action=1, reward=1.0, next_observation=[], done=True),
        Transition("obs2", action=2, reward=0.0, next_observation="live2", done=False),
    ]
    assert nonterminal_indices(batch) == (0, 2)


def test_training_protocol_seed_is_per_subject_not_last_launched_seed():
    seeds = (930001, 930002, 930003, 930004, 930005)
    first = trainer.build_plan(DQNConfig(seed=930001, device="cuda"), seeds=seeds)
    last = trainer.build_plan(DQNConfig(seed=930005, device="cuda"), seeds=seeds)
    assert first == last
    assert first["config"]["seed"] == "per_subject"
    assert first["subjects"]["seeds"] == list(seeds)


def test_training_protocol_freeze_rejects_silent_rewrites(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "PROTOCOL", tmp_path)
    seeds = (930001, 930002)
    trainer.freeze_plan(DQNConfig(seed=930001, device="cuda"), seeds=seeds)

    # Launching a different member of the same predeclared subject set must
    # be idempotent; changing the actual protocol must fail closed.
    trainer.freeze_plan(DQNConfig(seed=930002, device="cuda"), seeds=seeds)
    with pytest.raises(RuntimeError, match="frozen DQN training plan differs"):
        trainer.freeze_plan(DQNConfig(seed=930001, device="cpu"), seeds=seeds)

    plan_path = trainer.training_plan_path(DQNConfig(seed=930001, device="cuda"))
    payload = json.loads(plan_path.read_text())
    assert payload["config"]["device"] == "cuda"


def test_optimized_protocol_does_not_overwrite_legacy_v1_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "PROTOCOL", tmp_path)
    legacy = trainer.training_plan_path(DQNConfig(name="letter_env_dqn_v1"))
    optimized = trainer.training_plan_path(DQNConfig())
    assert legacy.name == "letter_env_dqn_training_plan.json"
    assert optimized.name == "letter_env_dqn_training_plan_letter_env_dqn_v2_cached_curriculum_recency_fix.json"
    assert optimized != legacy


# --- Pilot-then-confirmatory gate -------------------------------------------
#
# Mirrors the deepltl_point PPO precedent (dedicated pilot seed, inspected
# before the confirmatory seeds are authorized). These tests cover the state
# machine itself: pilot seeds are never gated, confirmatory seeds are
# refused before authorization and permitted after, and authorization
# itself is refused before the pilot has ever launched.


def test_pilot_seed_is_disjoint_from_confirmatory_seeds():
    assert trainer.PILOT_SEED not in trainer.DEFAULT_SEEDS


def test_confirmatory_seed_refused_before_pilot_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "PROTOCOL", tmp_path)
    with pytest.raises(RuntimeError, match="pilot seed .* has not been inspected"):
        trainer.require_seed_authorized(trainer.DEFAULT_SEEDS[0])


def test_pilot_seed_never_requires_authorization(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "PROTOCOL", tmp_path)
    trainer.require_seed_authorized(trainer.PILOT_SEED)  # must not raise
    assert not trainer.gate_path().exists()


def test_authorize_confirmatory_training_refuses_before_pilot_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "PROTOCOL", tmp_path)
    with pytest.raises(RuntimeError, match="pilot seed .* has not been launched"):
        trainer.authorize_confirmatory_training()


def test_full_gate_lifecycle_unlocks_confirmatory_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "PROTOCOL", tmp_path)

    trainer.record_pilot_launch()
    with pytest.raises(RuntimeError, match="has not been inspected"):
        trainer.require_seed_authorized(trainer.DEFAULT_SEEDS[0])

    gate = trainer.authorize_confirmatory_training()
    assert gate["status"] == "confirmatory_authorized"
    assert gate["contamination_controls"]["pilot_outcomes_inspected"] is True
    assert gate["contamination_controls"]["confirmatory_training_authorized"] is True

    for seed in trainer.DEFAULT_SEEDS:
        trainer.require_seed_authorized(seed)  # must not raise

    payload = json.loads(trainer.gate_path().read_text())
    statuses = [entry["status"] for entry in payload["history"]]
    assert statuses == ["pilot_not_launched", "pilot_launched", "confirmatory_authorized"]


def test_gate_file_hash_sidecar_matches_content(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "PROTOCOL", tmp_path)
    trainer.ensure_gate()
    gate_file = trainer.gate_path()
    sidecar = gate_file.with_suffix(gate_file.suffix + ".sha256")
    expected = trainer.sha256_bytes(gate_file.read_bytes())
    assert sidecar.read_text().split()[0] == expected
