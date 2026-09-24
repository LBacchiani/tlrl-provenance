#!/usr/bin/env python
"""Train a DQN subject on DeepLTL's frozen LetterEnv environment.

By default this script freezes and prints a training plan but does not execute
learning. Pass ``--execute`` for a real run, or ``--smoke-steps`` for a short
implementation check.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEEPLTL = ROOT / ".runtime" / "deep-ltl" / "src"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(DEEPLTL))

from tlrl_benchmarks.letter_env import (  # noqa: E402
    ENVIRONMENT_ID,
    EXPECTED_MAX_STEPS,
    EXPECTED_TASK_COUNT,
)
from tlrl_benchmarks.letter_env.dqn import (  # noqa: E402
    DQNConfig,
    ReplayBuffer,
    Transition,
    action_index_to_env_action,
    build_q_network,
    collate_cached_observations,
    choose_epsilon_greedy_index,
    encode_observation,
    env_action_to_action_index,
    epsilon_at_step,
    nonterminal_indices,
    resolve_device,
)


PLAN_SCHEMA = "tlrl-letter-env-dqn-training-plan/2"
GATE_SCHEMA = "tlrl-letter-env-dqn-pilot-gate/2"
PILOT_SEED = 930000
DEFAULT_SEEDS = (930001, 930002, 930003, 930004, 930005)
PROTOCOL = ROOT / "evidence" / "letter_env" / "dqn" / "development" / "training"
EXPERIMENTS = ROOT / "experiments" / "dqn" / ENVIRONMENT_ID

if PILOT_SEED in DEFAULT_SEEDS:
    raise RuntimeError("PILOT_SEED must be disjoint from the confirmatory DEFAULT_SEEDS")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def make_env(seed: int, *, curriculum: Any | None = None, initialize_vocab: bool = True):
    import preprocessing
    import utils
    from envs import make_env as deepltl_make_env
    from envs import get_env_attr
    from sequence.samplers import CurriculumSampler, curricula

    utils.set_seed(seed)
    if curriculum is None:
        curriculum = copy.deepcopy(curricula[ENVIRONMENT_ID])
    sampler = CurriculumSampler.partial(curriculum)
    env = deepltl_make_env(ENVIRONMENT_ID, sampler, sequence=True, max_steps=EXPECTED_MAX_STEPS)
    env.reset(seed=100 * seed)
    assignments = get_env_attr(env, "get_possible_assignments")()
    if initialize_vocab:
        preprocessing.reset_vocab()
        preprocessing.init_vocab(assignments)
    return env, curriculum, assignments


def experiment_dir(config: DQNConfig) -> Path:
    return EXPERIMENTS / config.name / str(config.seed)


def training_plan_path(config: DQNConfig) -> Path:
    """Keep the original v1 freeze immutable while versioning new trainers."""

    if config.name == "letter_env_dqn_v1":
        return PROTOCOL / "letter_env_dqn_training_plan.json"
    safe_name = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in config.name
    )
    return PROTOCOL / f"letter_env_dqn_training_plan_{safe_name}.json"


def build_plan(config: DQNConfig, *, seeds: tuple[int, ...]) -> dict[str, Any]:
    corpus_manifest = ROOT / "src" / "tlrl_benchmarks" / "letter_env" / "corpus_manifest.json"
    freeze_script = ROOT / "scripts" / "freeze_letter_env_corpus.py"
    benchmark_script = ROOT / "scripts" / "benchmark_letter_env_dqn_preprocessing.py"
    train_script = Path(__file__)
    protocol_config = config.to_jsonable()
    protocol_config["seed"] = "per_subject"
    return {
        "schema_version": PLAN_SCHEMA,
        "benchmark_id": "deepltl-letter-env-dqn",
        "status": "training_protocol_frozen",
        "training_executed": False,
        "algorithm": {
            "name": "Double DQN" if config.double_dqn else "DQN",
            "implementation": "v3.0/scripts/train_letter_env_dqn.py",
            "network": "DeepLTL LetterEnv convolutional encoder plus LTL sequence encoder plus Q head",
            "reward": "official DeepLTL SequenceWrapper reach-avoid reward",
            "environment": ENVIRONMENT_ID,
            "curriculum": "official DeepLTL LETTER_CURRICULUM",
            "replay_preprocessing": config.replay_preprocessing,
        },
        "subjects": {
            "seeds": list(seeds),
            "count": len(seeds),
            "selection_rule": "all listed seeds are trained; final checkpoint is primary",
        },
        "config": protocol_config,
        "evaluation_corpus": {
            "frozen_manifest": str(corpus_manifest.relative_to(ROOT)).replace("\\", "/"),
            "frozen_manifest_sha256": sha256_bytes(corpus_manifest.read_bytes()),
            "formula_count": EXPECTED_TASK_COUNT,
            "episode_horizon": EXPECTED_MAX_STEPS,
            "outcomes_inspected": False,
        },
        "source_hashes": {
            "train_script_sha256": sha256_bytes(train_script.read_bytes()),
            "dqn_module_sha256": sha256_bytes(
                (ROOT / "src" / "tlrl_benchmarks" / "letter_env" / "dqn.py").read_bytes()
            ),
            "corpus_freeze_script_sha256": sha256_bytes(freeze_script.read_bytes()),
            "preprocessing_benchmark_sha256": sha256_bytes(benchmark_script.read_bytes()),
        },
        "contamination_controls": {
            "deepltl_point_policy_dependency": False,
            "v2_dependency": False,
            "confirmatory_outcomes_inspected_before_training": False,
            "best_checkpoint_selected_after_outcomes": False,
        },
        "implementation_amendment": {
            "reason": "v1 replay preprocessing made the declared training budget operationally infeasible",
            "changes_learning_rule": False,
            "changes_environment_distribution": False,
            "changes_network_or_hyperparameters": False,
            "validation": "stock and cached collators produce bit-identical tensors and Q outputs",
            "supersedes_plan": "letter_env_dqn_training_plan.json",
        },
    }


def freeze_plan(config: DQNConfig, *, seeds: tuple[int, ...]) -> dict[str, Any]:
    plan = build_plan(config, seeds=seeds)
    payload = canonical_json(plan)
    output = training_plan_path(config)
    if output.exists() and output.read_bytes() != payload:
        raise RuntimeError(
            f"frozen DQN training plan differs from current config/inventory: {output}"
        )
    write_bytes(output, payload)
    write_bytes(
        output.with_suffix(output.suffix + ".sha256"),
        f"{sha256_bytes(payload)}  {output.name}\n".encode("ascii"),
    )
    return {
        "status": plan["status"],
        "plan_path": display_path(output),
        "plan_sha256": sha256_bytes(payload),
        "seeds": list(seeds),
        "total_steps_per_seed": config.total_steps,
    }


# --- Pilot-then-confirmatory gate ------------------------------------------
#
# Mirrors the deepltl_point PPO precedent: a dedicated pilot seed (disjoint
# from the confirmatory seed set) is trained and inspected first; the
# confirmatory seeds are refused until that inspection has been explicitly
# recorded via --authorize-confirmatory-training. Unlike the training plan
# above (a single frozen, byte-identical document), this file's whole
# purpose is to record a state transition over time, so it is read/written
# rather than freeze-once-and-match.


def gate_path() -> Path:
    return PROTOCOL / "letter_env_dqn_v2_cached_pilot_gate.json"


def _write_gate(gate: dict[str, Any]) -> None:
    payload = canonical_json(gate)
    output = gate_path()
    write_bytes(output, payload)
    write_bytes(
        output.with_suffix(output.suffix + ".sha256"),
        f"{sha256_bytes(payload)}  {output.name}\n".encode("ascii"),
    )


def read_gate() -> dict[str, Any] | None:
    output = gate_path()
    if not output.exists():
        return None
    return json.loads(output.read_text(encoding="utf-8"))


def ensure_gate() -> dict[str, Any]:
    """Create the gate file in its initial state if it does not exist yet."""

    gate = read_gate()
    if gate is not None:
        return gate
    gate = {
        "schema_version": GATE_SCHEMA,
        "pilot_seed": PILOT_SEED,
        "confirmatory_seeds": list(DEFAULT_SEEDS),
        "status": "pilot_not_launched",
        "contamination_controls": {
            "pilot_outcomes_inspected": False,
            "confirmatory_training_authorized": False,
        },
        "history": [{"status": "pilot_not_launched", "epoch_seconds": time.time()}],
    }
    _write_gate(gate)
    return gate


def record_pilot_launch() -> dict[str, Any]:
    gate = ensure_gate()
    if gate["status"] == "pilot_not_launched":
        gate["status"] = "pilot_launched"
        gate["history"].append({"status": "pilot_launched", "epoch_seconds": time.time()})
        _write_gate(gate)
    return gate


def authorize_confirmatory_training() -> dict[str, Any]:
    gate = ensure_gate()
    if gate["status"] == "pilot_not_launched":
        raise RuntimeError(
            "cannot authorize confirmatory training: the pilot seed "
            f"({PILOT_SEED}) has not been launched yet"
        )
    if gate["status"] != "confirmatory_authorized":
        gate["status"] = "confirmatory_authorized"
        gate["contamination_controls"]["pilot_outcomes_inspected"] = True
        gate["contamination_controls"]["confirmatory_training_authorized"] = True
        gate["history"].append({"status": "confirmatory_authorized", "epoch_seconds": time.time()})
        _write_gate(gate)
    return gate


def require_seed_authorized(seed: int) -> None:
    if seed == PILOT_SEED:
        return
    if seed not in DEFAULT_SEEDS:
        return
    gate = ensure_gate()
    if not gate["contamination_controls"]["confirmatory_training_authorized"]:
        raise RuntimeError(
            f"refusing to train confirmatory seed {seed}: pilot seed {PILOT_SEED} "
            "has not been inspected and authorized yet. Train the pilot first "
            "(--execute --seed 930000), inspect its outcomes, then run again "
            "with --authorize-confirmatory-training."
        )


class EpisodeTracker:
    def __init__(self, window: int = 500):
        self.window = window
        self.episodes: list[dict[str, Any]] = []
        self.task_success: dict[Any, list[int]] = {}
        # Curriculum stages sampled from a large or unbounded task space (e.g.
        # RandomCurriculumStage) never resample most distinct goals often
        # enough for their per-goal average to refresh: a goal touched once,
        # early, keeps contributing that stale value to curriculum_success()
        # forever, since task_success itself is never pruned. This tracks the
        # episode index at which each goal was last recorded, so
        # curriculum_success() can restrict itself to goals verified recently
        # -- reusing `window` as the recency horizon, the same constant
        # already used to bound how much history a single goal retains.
        self._episode_count = 0
        self._last_seen: dict[Any, int] = {}

    def record(self, *, goal: Any, reward: float, steps: int, success: bool, violation: bool) -> None:
        self.episodes.append(
            {
                "return": reward,
                "steps": steps,
                "success": int(success),
                "violation": int(violation),
            }
        )
        bucket = self.task_success.setdefault(goal, [])
        bucket.append(int(success))
        if len(bucket) > self.window:
            del bucket[:-self.window]
        self._episode_count += 1
        self._last_seen[goal] = self._episode_count

    def curriculum_success(self) -> dict[Any, float]:
        return {
            goal: sum(values) / len(values)
            for goal, values in self.task_success.items()
            if self._episode_count - self._last_seen[goal] < self.window
        }

    def recent_summary(self) -> dict[str, float]:
        recent = self.episodes[-self.window :]
        if not recent:
            return {"episodes": 0, "success_rate": 0.0, "violation_rate": 0.0, "mean_return": 0.0}
        return {
            "episodes": float(len(recent)),
            "success_rate": sum(row["success"] for row in recent) / len(recent),
            "violation_rate": sum(row["violation"] for row in recent) / len(recent),
            "mean_return": sum(row["return"] for row in recent) / len(recent),
            "mean_steps": sum(row["steps"] for row in recent) / len(recent),
        }


def save_status(
    path: Path,
    *,
    model: Any,
    target_model: Any,
    optimizer: Any,
    config: DQNConfig,
    num_steps: int,
    num_updates: int,
    curriculum_stage: int,
) -> None:
    import torch
    import preprocessing

    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "algorithm": "double_dqn" if config.double_dqn else "dqn",
            "config": config.to_jsonable(),
            "num_steps": num_steps,
            "num_updates": num_updates,
            "model_state": model.state_dict(),
            "target_model_state": target_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "curriculum_stage": curriculum_stage,
        },
        path / "status.pth",
    )
    with (path / "vocab.pkl").open("wb") as handle:
        pickle.dump(preprocessing.VOCAB, handle)
    eval_dir = path / "eval"
    eval_dir.mkdir(exist_ok=True)
    torch.save(
        {
            "algorithm": "double_dqn" if config.double_dqn else "dqn",
            "config": config.to_jsonable(),
            "num_steps": num_steps,
            "model_state": model.state_dict(),
            "curriculum_stage": curriculum_stage,
        },
        eval_dir / f"{num_steps}.pth",
    )


def train_one_seed(config: DQNConfig, *, smoke_steps: int | None = None) -> dict[str, Any]:
    if config.num_threads is not None:
        # Must happen before numpy/torch are first imported in this process:
        # OMP/MKL read these once at native-library init time, not per-call.
        # Running several training processes concurrently without this caps
        # each one to a default of "use every core", causing severe
        # cross-process CPU contention instead of real parallel throughput.
        os.environ.setdefault("OMP_NUM_THREADS", str(config.num_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(config.num_threads))

    import numpy as np
    import torch
    import torch.nn.functional as F
    import preprocessing

    if config.num_threads is not None:
        torch.set_num_threads(config.num_threads)

    device = resolve_device(config.device)
    total_steps = smoke_steps if smoke_steps is not None else config.total_steps
    first_env, curriculum, _assignments = make_env(config.seed)
    envs = [first_env]
    for index in range(1, config.num_envs):
        env, _curriculum, _assignments = make_env(
            config.seed + index,
            curriculum=curriculum,
            initialize_vocab=False,
        )
        envs.append(env)
    action_dim = int(first_env.action_space.n)
    propositions = set(first_env.unwrapped.get_propositions())
    rng = random.Random(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    model = build_q_network(first_env, config).to(device)
    target_model = build_q_network(first_env, config).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=config.adam_eps)
    buffer = ReplayBuffer(config.buffer_size, seed=config.seed)
    output_dir = experiment_dir(config)
    log_path = output_dir / "training_log.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker = EpisodeTracker()

    observations = []
    cached_observations = []
    episode_returns = []
    episode_lengths = []
    for index in range(config.num_envs):
        obs = envs[index].reset(seed=100 * config.seed + index)
        observations.append(obs)
        if config.replay_preprocessing == "cached":
            cached_observations.append(encode_observation(obs, propositions))
        episode_returns.append(0.0)
        episode_lengths.append(0)

    num_steps = 0
    num_updates = 0
    started = time.time()
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "num_steps",
                "num_updates",
                "epsilon",
                "loss",
                "success_rate",
                "violation_rate",
                "mean_return",
                "mean_steps",
                "curriculum_stage",
                "sps",
            ],
        )
        writer.writeheader()
        last_loss = float("nan")
        while num_steps < total_steps:
            # Batch all num_envs current observations into one forward pass
            # instead of num_envs separate batch-size-1 calls: each call pays
            # a full CPU<->GPU round trip and kernel-launch overhead that
            # barely depends on batch size, so doing it once per step instead
            # of once per (step, env) is the dominant rollout-throughput cost.
            epsilon = epsilon_at_step(config, num_steps)
            with torch.no_grad():
                if config.replay_preprocessing == "cached":
                    processed = collate_cached_observations(cached_observations, device=device)
                else:
                    processed = preprocessing.preprocess_obss(
                        observations,
                        propositions,
                        device=device,
                    )
                batch_q_values = model(processed).detach().cpu().tolist()
            epsilon_mask = processed.epsilon_mask.tolist()
            for env_index in range(config.num_envs):
                q_values = batch_q_values[env_index]
                action_index = choose_epsilon_greedy_index(
                    q_values,
                    action_dim=action_dim,
                    epsilon_enabled=bool(epsilon_mask[env_index]),
                    epsilon=epsilon,
                    rng=rng,
                )
                env_action = np.int64(action_index_to_env_action(action_index, action_dim))
                next_obs, reward, done, info = envs[env_index].step(env_action)
                cached_next_obs = None
                if config.replay_preprocessing == "cached" and not done:
                    cached_next_obs = encode_observation(next_obs, propositions)
                buffer.append(
                    Transition(
                        cached_observations[env_index]
                        if config.replay_preprocessing == "cached"
                        else observations[env_index],
                        int(env_action),
                        float(reward),
                        cached_next_obs
                        if config.replay_preprocessing == "cached"
                        else next_obs,
                        bool(done),
                    )
                )
                episode_returns[env_index] += float(reward)
                episode_lengths[env_index] += 1
                if done:
                    tracker.record(
                        goal=observations[env_index]["initial_goal"],
                        reward=episode_returns[env_index],
                        steps=episode_lengths[env_index],
                        success="success" in info,
                        violation="violation" in info,
                    )
                    obs = envs[env_index].reset(seed=100 * config.seed + num_steps + env_index + 1)
                    observations[env_index] = obs
                    if config.replay_preprocessing == "cached":
                        cached_observations[env_index] = encode_observation(obs, propositions)
                    episode_returns[env_index] = 0.0
                    episode_lengths[env_index] = 0
                else:
                    observations[env_index] = next_obs
                    if config.replay_preprocessing == "cached":
                        cached_observations[env_index] = cached_next_obs
                num_steps += 1

                if len(buffer) >= config.learning_starts and num_steps % config.train_frequency == 0:
                    batch = buffer.sample(config.batch_size)
                    if config.replay_preprocessing == "cached":
                        obs_batch = collate_cached_observations(
                            [row.observation for row in batch],
                            device=device,
                        )
                    else:
                        obs_batch = preprocessing.preprocess_obss(
                            [row.observation for row in batch],
                            propositions,
                            device=device,
                        )
                    actions = torch.tensor(
                        [env_action_to_action_index(row.action, action_dim) for row in batch],
                        dtype=torch.long,
                        device=device,
                    )
                    rewards = torch.tensor(
                        [row.reward for row in batch],
                        dtype=torch.float32,
                        device=device,
                    )
                    dones = torch.tensor(
                        [row.done for row in batch],
                        dtype=torch.float32,
                        device=device,
                    )
                    live_indices = nonterminal_indices(batch)
                    q_selected = model(obs_batch).gather(1, actions.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        next_values = torch.zeros(len(batch), dtype=torch.float32, device=device)
                        if live_indices:
                            if config.replay_preprocessing == "cached":
                                next_obs_batch = collate_cached_observations(
                                    [batch[index].next_observation for index in live_indices],
                                    device=device,
                                )
                            else:
                                next_obs_batch = preprocessing.preprocess_obss(
                                    [batch[index].next_observation for index in live_indices],
                                    propositions,
                                    device=device,
                                )
                            if config.double_dqn:
                                next_actions = model(next_obs_batch).argmax(dim=1)
                                live_next_values = target_model(next_obs_batch).gather(
                                    1, next_actions.unsqueeze(1)
                                ).squeeze(1)
                            else:
                                live_next_values = target_model(next_obs_batch).max(dim=1).values
                            next_values[
                                torch.tensor(live_indices, dtype=torch.long, device=device)
                            ] = live_next_values
                        targets = rewards + config.discount * (1.0 - dones) * next_values
                    loss = F.smooth_l1_loss(q_selected, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    last_loss = float(loss.item())
                    num_updates += 1

                if num_steps % config.target_update_frequency == 0:
                    target_model.load_state_dict(model.state_dict())
                if tracker.task_success and num_steps % config.log_interval == 0:
                    curriculum.update_task_success(tracker.curriculum_success(), verbose=True)
                if num_steps % config.log_interval == 0 or num_steps >= total_steps:
                    elapsed = max(time.time() - started, 1e-9)
                    summary = tracker.recent_summary()
                    writer.writerow(
                        {
                            "num_steps": num_steps,
                            "num_updates": num_updates,
                            "epsilon": epsilon_at_step(config, num_steps),
                            "loss": last_loss,
                            "success_rate": summary["success_rate"],
                            "violation_rate": summary["violation_rate"],
                            "mean_return": summary["mean_return"],
                            "mean_steps": summary.get("mean_steps", 0.0),
                            "curriculum_stage": curriculum.stage_index,
                            "sps": num_steps / elapsed,
                        }
                    )
                    handle.flush()
                if num_steps % config.checkpoint_interval == 0 or num_steps >= total_steps:
                    save_status(
                        output_dir,
                        model=model,
                        target_model=target_model,
                        optimizer=optimizer,
                        config=config,
                        num_steps=num_steps,
                        num_updates=num_updates,
                        curriculum_stage=curriculum.stage_index,
                    )
                if num_steps >= total_steps:
                    break

    save_status(
        output_dir,
        model=model,
        target_model=target_model,
        optimizer=optimizer,
        config=config,
        num_steps=num_steps,
        num_updates=num_updates,
        curriculum_stage=curriculum.stage_index,
    )
    return {
        "status": "smoke_complete" if smoke_steps is not None else "training_complete",
        "seed": config.seed,
        "num_steps": num_steps,
        "num_updates": num_updates,
        "output_dir": display_path(output_dir),
        "recent": tracker.recent_summary(),
    }


def parse_args() -> argparse.Namespace:
    defaults = DQNConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-all", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(DEFAULT_SEEDS))
    parser.add_argument("--total-steps", type=int, default=defaults.total_steps)
    parser.add_argument("--num-envs", type=int, default=defaults.num_envs)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default=defaults.device)
    parser.add_argument("--name", default=defaults.name)
    parser.add_argument(
        "--replay-preprocessing",
        choices=["cached", "upstream"],
        default=defaults.replay_preprocessing,
        help=(
            "Cache invariant logical encodings in replay (recommended) or "
            "re-run DeepLTL's stock preprocessing on every sample."
        ),
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=defaults.num_threads,
        help=(
            "Cap OMP/MKL/PyTorch intra-op threads for this process. Required "
            "when running multiple seeds concurrently on one machine -- "
            "PyTorch otherwise defaults each process to using every core, "
            "and N unthrottled processes contend rather than parallelize."
        ),
    )
    parser.add_argument(
        "--authorize-confirmatory-training",
        action="store_true",
        help=(
            "Deliberate, standalone action: after inspecting the pilot "
            f"seed's ({PILOT_SEED}) outcomes, authorize training the "
            "confirmatory seeds. Refuses if the pilot has not been "
            "launched yet. Cannot be combined with --execute/--execute-all."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.authorize_confirmatory_training:
        if args.execute or args.execute_all or args.smoke_steps is not None:
            raise ValueError(
                "--authorize-confirmatory-training cannot be combined with "
                "--execute, --execute-all, or --smoke-steps"
            )
        gate = authorize_confirmatory_training()
        print(json.dumps(gate, indent=2, sort_keys=True))
        return
    if args.execute_all and args.smoke_steps is not None:
        raise ValueError("--execute-all cannot be combined with --smoke-steps")
    if args.execute and args.smoke_steps is None:
        require_seed_authorized(args.seed)
    if args.execute_all:
        for seed in args.seeds:
            require_seed_authorized(seed)
    config = DQNConfig(
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        seed=args.seed,
        name=args.name,
        device=args.device,
        num_threads=args.num_threads,
        replay_preprocessing=args.replay_preprocessing,
    )
    if args.smoke_steps is not None:
        plan_result = {
            "status": "smoke_protocol_not_frozen",
            "message": "Smoke runs do not overwrite the frozen DQN training plan.",
            "seeds": list(args.seeds),
            "total_steps_per_seed": config.total_steps,
        }
    else:
        plan_result = freeze_plan(config, seeds=tuple(args.seeds))
    if not args.execute and not args.execute_all and args.smoke_steps is None:
        print(json.dumps({**plan_result, "message": "Re-run with --execute to train."}, indent=2))
        return
    if args.execute_all:
        runs = []
        for seed in args.seeds:
            seed_config = DQNConfig(
                total_steps=args.total_steps,
                num_envs=args.num_envs,
                seed=seed,
                name=args.name,
                device=args.device,
                num_threads=args.num_threads,
                replay_preprocessing=args.replay_preprocessing,
            )
            if seed == PILOT_SEED:
                record_pilot_launch()
            runs.append(train_one_seed(seed_config))
        print(json.dumps({**plan_result, "runs": runs}, indent=2, sort_keys=True))
    else:
        if args.execute and args.seed == PILOT_SEED:
            record_pilot_launch()
        result = train_one_seed(config, smoke_steps=args.smoke_steps)
        print(json.dumps({**plan_result, "run": result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
