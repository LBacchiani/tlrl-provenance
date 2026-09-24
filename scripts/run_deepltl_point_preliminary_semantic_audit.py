#!/usr/bin/env python
"""Run a small development-only semantic audit for DeepLTL PointWorld."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / ".runtime" / "deep-ltl"
UPSTREAM_SRC = UPSTREAM / "src"
V3_SRC = ROOT / "src"
DEFAULT_EXPERIMENT = "v3_dev_ppo_5m_seed_910001"
DEFAULT_SEED = 910001
DEFAULT_CHECKPOINT_STEPS = 4_456_448
DEFAULT_EPISODES_PER_FORMULA = 2
DEFAULT_OUTPUT = ROOT / "evidence" / "deepltl_point" / "development" / "semantic" / "deepltl_point_preliminary_semantic_audit.json"

for candidate in (str(V3_SRC), str(UPSTREAM_SRC)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from envs import make_env  # noqa: E402
from ltl import FixedSampler  # noqa: E402
from model.agent import Agent  # noqa: E402
from model.model import build_model  # noqa: E402
from ltl.automata import LDBASequence  # noqa: E402
from sequence.search import SequenceSearch  # noqa: E402
from utils.model_store import ModelStore  # noqa: E402
from config import model_configs  # noqa: E402

from tlrl_benchmarks.deepltl_point.adapter import (  # noqa: E402
    ENVIRONMENT_ID,
    assess_point_episode,
    build_point_trace,
    capture_official_step,
    load_tasks,
    preserve_termination_and_truncation,
)
from tlrl_provenance import EvaluationLimits  # noqa: E402
from tlrl_provenance.statistics import certificate_events  # noqa: E402


class RobustFiniteSearch(SequenceSearch):
    """Shortest finite accepting sequence search for development auditing.

    DeepLTL trains PointLtl2 through explicit reach-avoid sequences. Its helper
    evaluator searches those sequences from LDBA states, but can return an empty
    candidate list after some finite-state updates. The audit collector needs a
    nonempty sequence whenever the environment has not terminated, so this local
    search handles accepting states and finite accepting paths directly.
    """

    def __call__(self, ldba: Any, ldba_state: int, obs: dict[str, Any]) -> LDBASequence:
        candidates = self._sequences(ldba, ldba_state)
        if not candidates:
            raise RuntimeError(f"no accepting LDBA sequence from state {ldba_state}")
        return max(candidates, key=lambda sequence: self.get_value(sequence, obs))

    def _sequences(self, ldba: Any, ldba_state: int) -> list[LDBASequence]:
        accepting_self = self._accepting_self_loop(ldba, ldba_state)
        if accepting_self is not None:
            return [LDBASequence((accepting_self,))]

        queue: list[tuple[int, tuple[tuple[frozenset[Any] | type, frozenset[Any]], ...], frozenset[int]]] = [
            (ldba_state, (), frozenset())
        ]
        shortest: int | None = None
        results: list[LDBASequence] = []
        while queue:
            state, sequence, visited = queue.pop(0)
            if shortest is not None and len(sequence) >= shortest:
                continue
            avoid = self._avoid_assignments(ldba, state, visited)
            for transition in ldba.state_to_transitions[state]:
                if transition.target in visited:
                    continue
                target_scc = ldba.state_to_scc[transition.target]
                if target_scc.bottom and not target_scc.accepting:
                    continue
                reach = (
                    LDBASequence.EPSILON
                    if transition.is_epsilon()
                    else frozenset(transition.valid_assignments)
                )
                next_sequence = sequence + ((reach, avoid),)
                if transition.accepting or target_scc.accepting:
                    shortest = len(next_sequence)
                    results.append(LDBASequence(next_sequence))
                    continue
                queue.append((transition.target, next_sequence, visited | {state}))
        return results

    def _accepting_self_loop(
        self,
        ldba: Any,
        state: int,
    ) -> tuple[frozenset[Any] | type, frozenset[Any]] | None:
        scc = ldba.state_to_scc[state]
        if not scc.accepting:
            return None
        for transition in ldba.state_to_transitions[state]:
            if transition.target == state and transition.accepting:
                reach = (
                    LDBASequence.EPSILON
                    if transition.is_epsilon()
                    else frozenset(transition.valid_assignments)
                )
                return reach, frozenset()
        return None

    def _avoid_assignments(self, ldba: Any, state: int, visited: frozenset[int]) -> frozenset[Any]:
        avoid_sets = []
        for transition in ldba.state_to_transitions[state]:
            if transition.source == transition.target:
                continue
            scc = ldba.state_to_scc[transition.target]
            if (scc.bottom and not scc.accepting) or transition.target in visited:
                avoid_sets.append(transition.valid_assignments)
        if not avoid_sets:
            return frozenset()
        return frozenset().union(*(frozenset(items) for items in avoid_sets))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
    temporary.replace(path)


def json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, Counter):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(json_ready(item) for item in value)
    return str(value)


def checkpoint_path(experiment: str, seed: int, checkpoint_steps: int) -> Path:
    path = (
        UPSTREAM
        / "experiments"
        / "ppo"
        / ENVIRONMENT_ID
        / experiment
        / str(seed)
        / "eval"
        / f"{checkpoint_steps}.pth"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_agent(experiment: str, seed: int, checkpoint: Path, formula: str) -> tuple[Any, Agent]:
    sampler = FixedSampler.partial(formula)
    outer_env = make_env(ENVIRONMENT_ID, sampler, render_mode=None)
    env = preserve_termination_and_truncation(outer_env)
    model_store = ModelStore(ENVIRONMENT_ID, experiment, seed)
    model_store.load_vocab()
    status = torch.load(checkpoint, map_location="cpu")
    model = build_model(env, status, model_configs[ENVIRONMENT_ID])
    model.eval()
    props = set(env.get_propositions())
    search = RobustFiniteSearch(model, props)
    agent = Agent(model, search=search, propositions=props, verbose=False)
    return env, agent


def run_episode(
    *,
    env: Any,
    agent: Agent,
    task_id: str,
    formula: str,
    task_by_id: dict[str, Any],
    policy_id: str,
    seed: int,
    deterministic: bool,
    evaluation_limits: EvaluationLimits = EvaluationLimits(),
    verify_structural_certificate: bool = True,
    verification_kwargs: dict[str, Any] | None = None,
    collect_semantic_events: bool = True,
    compute_prefix: bool = True,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs, info = env.reset(seed=seed)
    agent.reset()
    steps = []
    terminated = False
    truncated = False
    official_success = False
    official_violation = False
    total_reward = 0.0

    while not (terminated or truncated):
        action = agent.get_action(obs, info, deterministic=deterministic)
        action = action.flatten()
        if action.shape == (1,):
            action = action[0]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps.append(capture_official_step(env, info, action=action.tolist(), reward=float(reward)))
        official_success = official_success or bool(info.get("success", False))
        official_violation = official_violation or bool(info.get("violation", False))
        if len(steps) > 1000:
            raise RuntimeError("PointLtl2 TimeLimit failed to truncate at 1000 steps")

    trace = build_point_trace(
        steps,
        policy_id=policy_id,
        seed=seed,
        task_id=task_id,
        terminated=terminated,
        truncated=truncated,
    )
    assessment = assess_point_episode(
        task_by_id[task_id],
        trace,
        official_success=official_success,
        official_violation=official_violation,
        evaluation_limits=evaluation_limits,
        verify_structural_certificate=verify_structural_certificate,
        verification_kwargs=verification_kwargs,
        compute_prefix=compute_prefix,
    )
    certificate = assessment.certificate
    event_counts = (
        Counter(
            f"{event.occurrence_id}:{event.operator}:{event.kind}:{event.alternative}"
            for event in certificate_events(certificate)
        )
        if collect_semantic_events
        else Counter()
    )
    atom_counts = Counter(atom for labels in trace.labels for atom in labels)
    return {
        "task_id": task_id,
        "formula": formula,
        "seed": seed,
        "steps": len(steps),
        "terminated": terminated,
        "truncated": truncated,
        "official_success": official_success,
        "official_violation": official_violation,
        "independent_verdict": certificate.verdict.value,
        "disposition": assessment.disposition.value,
        "return_sum": total_reward,
        "mean_possible_evidence": len(certificate.possible_evidence),
        "mean_necessary_evidence": len(certificate.necessary_evidence),
        "atom_counts": dict(sorted(atom_counts.items())),
        "semantic_event_counts": dict(sorted(event_counts.items())),
    }


def summarize_formula(records: list[dict[str, Any]]) -> dict[str, Any]:
    trials = len(records)
    disposition_counts = Counter(record["disposition"] for record in records)
    verdict_counts = Counter(record["independent_verdict"] for record in records)
    atom_presence = Counter()
    event_presence = Counter()
    for record in records:
        atom_presence.update(record["atom_counts"].keys())
        event_presence.update(record["semantic_event_counts"].keys())
    return {
        "episodes": trials,
        "dispositions": dict(sorted(disposition_counts.items())),
        "independent_verdicts": dict(sorted(verdict_counts.items())),
        "satisfied_rate": verdict_counts["satisfied"] / trials,
        "censored_rate": disposition_counts["censored"] / trials,
        "mean_steps": sum(record["steps"] for record in records) / trials,
        "mean_possible_evidence": sum(record["mean_possible_evidence"] for record in records) / trials,
        "mean_necessary_evidence": sum(record["mean_necessary_evidence"] for record in records) / trials,
        "atoms_seen_in_any_episode": sorted(atom_presence),
        "semantic_events_seen": len(event_presence),
        "top_semantic_events": event_presence.most_common(8),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--training-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--checkpoint-steps", type=int, default=DEFAULT_CHECKPOINT_STEPS)
    parser.add_argument("--episodes-per-formula", type=int, default=DEFAULT_EPISODES_PER_FORMULA)
    parser.add_argument("--max-formulas", type=int, default=None)
    parser.add_argument("--eval-seed-base", type=int, default=730_000)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.episodes_per_formula < 1:
        raise ValueError("episodes-per-formula must be positive")
    if not args.output.is_absolute():
        args.output = (Path.cwd() / args.output).resolve()

    os.chdir(UPSTREAM)
    checkpoint = checkpoint_path(args.experiment, args.training_seed, args.checkpoint_steps)
    tasks = load_tasks()
    if args.max_formulas is not None:
        if args.max_formulas < 1:
            raise ValueError("max-formulas must be positive when provided")
        tasks = tasks[: args.max_formulas]
    task_by_id = {task.task_id: task for task in tasks}
    all_records: list[dict[str, Any]] = []
    started = time.time()
    for index, task in enumerate(tasks):
        print(
            f"[{index + 1}/{len(tasks)}] auditing {task.task_id}: {task.source}",
            flush=True,
        )
        env, agent = load_agent(args.experiment, args.training_seed, checkpoint, task.source)
        try:
            for episode in range(args.episodes_per_formula):
                seed = args.eval_seed_base + index * 100 + episode
                all_records.append(
                    run_episode(
                        env=env,
                        agent=agent,
                        task_id=task.task_id,
                        formula=task.source,
                        task_by_id=task_by_id,
                        policy_id=f"{args.experiment}:{args.training_seed}:{args.checkpoint_steps}",
                        seed=seed,
                        deterministic=args.deterministic,
                    )
                )
                print(
                    f"[{index + 1}/{len(tasks)}] episode {episode + 1}/"
                    f"{args.episodes_per_formula} complete",
                    flush=True,
                )
        finally:
            env.close()

    by_formula: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in all_records:
        by_formula[record["task_id"]].append(record)
    formula_summaries = {
        task.task_id: {
            "formula": task.source,
            **summarize_formula(by_formula[task.task_id]),
        }
        for task in tasks
    }
    disposition_counts = Counter(record["disposition"] for record in all_records)
    verdict_counts = Counter(record["independent_verdict"] for record in all_records)
    result = {
        "schema_version": "tlrl-deepltl-point-preliminary-semantic-audit/1",
        "status": "completed",
        "scope": "development_only_preliminary_semantic_audit_not_confirmatory",
        "benchmark_id": "deepltl-pointworld-official-eval-50",
        "experiment": args.experiment,
        "training_seed": args.training_seed,
        "checkpoint_steps": args.checkpoint_steps,
        "checkpoint_path": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "episodes_per_formula": args.episodes_per_formula,
        "formula_count": len(tasks),
        "episode_count": len(all_records),
        "deterministic": args.deterministic,
        "elapsed_seconds": time.time() - started,
        "aggregate": {
            "dispositions": dict(sorted(disposition_counts.items())),
            "independent_verdicts": dict(sorted(verdict_counts.items())),
            "satisfied_rate": verdict_counts["satisfied"] / len(all_records),
            "censored_rate": disposition_counts["censored"] / len(all_records),
            "mean_steps": sum(record["steps"] for record in all_records) / len(all_records),
        },
        "formulas": formula_summaries,
        "records": all_records,
        "contamination_controls": {
            "uses_intermediate_development_checkpoint": True,
            "confirmatory_usefulness_evaluation_authorized": False,
            "not_used_for_budget_or_benchmark_selection": True,
        },
    }
    payload = (json.dumps(json_ready(result), indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write(args.output, payload)
    atomic_write(
        args.output.with_suffix(args.output.suffix + ".sha256"),
        f"{hashlib.sha256(payload).hexdigest()}  {args.output.name}\n".encode("ascii"),
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(args.output),
                "episodes": len(all_records),
                "satisfied_rate": result["aggregate"]["satisfied_rate"],
                "censored_rate": result["aggregate"]["censored_rate"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
