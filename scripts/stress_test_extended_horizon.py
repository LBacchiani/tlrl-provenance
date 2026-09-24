"""Stress test: does the audit behave correctly on traces far longer than the
official evaluated horizon (1000 steps for PointLtl2), and does the residual
formula stay far from the residual-size guard even on real (not synthetic)
long rollouts?

The confirmatory campaign uses PointLtl2's official 1000-step TimeLimit only
(a declared, disclosed scope boundary -- see paper \\S Threats to Validity).
This script does not extend that scope claim or produce a P_sat estimate; it
reuses one already-trained PPO checkpoint (seed 920001) and deliberately
overrides the environment's TimeLimit to 10x the official horizon (10,000
steps) to answer two narrow, checkable questions: (1) does the audit engine,
grounding replay, and agreement checking still behave correctly (no crash,
no silent misclassification) at 10x the evaluated trace length, and (2) do
episodes that were censored at the official 1000-step cutoff resolve to a
decided verdict given more time, or genuinely remain open -- i.e. is 1000
steps enough for this environment's dynamics, or an arbitrary cutoff that
manufactures censoring.

Must be run under the DeepLTL-compatible interpreter:
    python scripts/stress_test_extended_horizon.py
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
UPSTREAM = ROOT / ".runtime" / "deep-ltl"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_deepltl_point_preliminary_semantic_audit import (  # noqa: E402
    checkpoint_path,
    load_agent,
)
from tlrl_benchmarks.deepltl_point.adapter import (  # noqa: E402
    assess_point_episode,
    build_point_trace,
    capture_official_step,
    load_tasks,
)
from tlrl_provenance.prefix import DEFAULT_RESIDUAL_NODE_LIMIT, assess_prefix  # noqa: E402

EXPERIMENT = "v3_confirm_ppo_5m_seed_920001"
TRAINING_SEED = 920001
CHECKPOINT_STEPS = 4_980_736
NUM_FORMULAS = 10
EPISODES_PER_FORMULA = 3
EVAL_SEED_BASE = 8_970_000
OFFICIAL_HORIZON = 1000
EXTENDED_HORIZON = 10_000
OUTPUT_PATH = (
    ROOT
    / "evidence"
    / "deepltl_point"
    / "confirmatory"
    / "semantic"
    / "deepltl_point_extended_horizon_stress_test.json"
)
if os.environ.get("TLRL_OUTPUT_DIR"):  # redirect generated evidence; frozen files stay untouched
    OUTPUT_PATH = Path(os.environ["TLRL_OUTPUT_DIR"]) / OUTPUT_PATH.name


def _node_count(formula: Any) -> int:
    stack = [formula]
    seen: set[int] = set()
    count = 0
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        count += 1
        stack.extend(node.args)
    return count


def _set_horizon(env: Any, steps: int) -> None:
    # load_agent() already returns preserve_termination_and_truncation()'s
    # result, i.e. the inner Gymnasium TimeLimit wrapper directly (the outer
    # RemoveTruncWrapper is already stripped) -- not the RemoveTruncWrapper
    # itself.
    if type(env).__name__ != "TimeLimit":
        raise TypeError(f"expected the Gymnasium TimeLimit wrapper directly, found {type(env).__name__}")
    if getattr(env, "_max_episode_steps", None) not in (OFFICIAL_HORIZON, EXTENDED_HORIZON):
        raise ValueError("unexpected horizon value before overriding it")
    env._max_episode_steps = steps


def run_one_episode(env: Any, agent: Any, task: Any, seed: int, policy_id: str, horizon: int) -> dict[str, Any]:
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
    while not (terminated or truncated):
        action = agent.get_action(obs, info, deterministic=True)
        action = action.flatten()
        if action.shape == (1,):
            action = action[0]
        obs, reward, terminated, truncated, info = env.step(action)
        steps.append(capture_official_step(env, info, action=action.tolist(), reward=float(reward)))
        official_success = official_success or bool(info.get("success", False))
        official_violation = official_violation or bool(info.get("violation", False))
        if len(steps) > horizon:
            raise RuntimeError(f"TimeLimit failed to truncate at {horizon} steps")

    trace = build_point_trace(
        steps,
        policy_id=policy_id,
        seed=seed,
        task_id=task.task_id,
        terminated=terminated,
        truncated=truncated,
    )
    assessment = assess_point_episode(
        task,
        trace,
        official_success=official_success,
        official_violation=official_violation,
    )
    residual_assessment = assess_prefix(
        task.formula, trace, residual_node_limit=DEFAULT_RESIDUAL_NODE_LIMIT
    )
    return {
        "task_id": task.task_id,
        "steps": len(steps),
        "official_success": official_success,
        "official_violation": official_violation,
        "disposition": assessment.disposition.value,
        "prefix_status": assessment.prefix.status.value,
        "residual_node_count": _node_count(residual_assessment.continuation_residual),
    }


def main() -> None:
    os.chdir(UPSTREAM)
    checkpoint = checkpoint_path(EXPERIMENT, TRAINING_SEED, CHECKPOINT_STEPS)
    tasks = load_tasks()[:NUM_FORMULAS]

    official_results: list[dict[str, Any]] = []
    extended_results: list[dict[str, Any]] = []
    for task in tasks:
        env, agent = load_agent(EXPERIMENT, TRAINING_SEED, checkpoint, task.source)
        try:
            for episode_index in range(EPISODES_PER_FORMULA):
                seed = EVAL_SEED_BASE + episode_index
                # Official horizon first (env starts at its official 1000-step
                # TimeLimit), then override and rerun the identical seed so the
                # only varying factor is the horizon available to the policy.
                official_results.append(
                    run_one_episode(env, agent, task, seed, f"ppo_seed_{TRAINING_SEED}_official_horizon", OFFICIAL_HORIZON)
                )
                _set_horizon(env, EXTENDED_HORIZON)
                extended_results.append(
                    run_one_episode(env, agent, task, seed, f"ppo_seed_{TRAINING_SEED}_extended_horizon", EXTENDED_HORIZON)
                )
                _set_horizon(env, OFFICIAL_HORIZON)
        finally:
            env.close()

    def disposition_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["disposition"]] = counts.get(row["disposition"], 0) + 1
        return counts

    resolved_by_extension = 0
    still_censored = 0
    for official, extended in zip(official_results, extended_results):
        if official["disposition"] == "censored":
            if extended["disposition"] != "censored":
                resolved_by_extension += 1
            else:
                still_censored += 1

    result = {
        "schema_version": "deepltl-point-extended-horizon-stress-test/1",
        "purpose": (
            "Checks audit correctness and residual-guard margin on traces "
            "10x the official evaluated horizon, and whether officially "
            "censored episodes resolve given more time. Not a claim about "
            "P_sat at an extended horizon or a replacement confirmatory "
            "result."
        ),
        "checkpoint": {
            "experiment": EXPERIMENT,
            "training_seed": TRAINING_SEED,
            "checkpoint_steps": CHECKPOINT_STEPS,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
        "official_horizon": OFFICIAL_HORIZON,
        "extended_horizon": EXTENDED_HORIZON,
        "num_formulas": NUM_FORMULAS,
        "episodes_per_formula": EPISODES_PER_FORMULA,
        "num_episode_pairs": len(official_results),
        "official_horizon_disposition_counts": disposition_counts(official_results),
        "extended_horizon_disposition_counts": disposition_counts(extended_results),
        "censored_at_official_horizon_resolved_by_extension": resolved_by_extension,
        "censored_at_official_horizon_still_censored_at_extension": still_censored,
        "max_residual_node_count_official_horizon": max((r["residual_node_count"] for r in official_results), default=0),
        "max_residual_node_count_extended_horizon": max((r["residual_node_count"] for r in extended_results), default=0),
        "residual_node_limit": DEFAULT_RESIDUAL_NODE_LIMIT,
        "official_horizon_episodes": official_results,
        "extended_horizon_episodes": extended_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(OUTPUT_PATH.read_bytes()).hexdigest()
    (OUTPUT_PATH.parent / f"{OUTPUT_PATH.name}.sha256").write_text(
        f"{digest}  {OUTPUT_PATH.name}\n", encoding="utf-8"
    )

    print(f"wrote {OUTPUT_PATH} ({digest})")
    print(f"official horizon dispositions: {disposition_counts(official_results)}")
    print(f"extended horizon dispositions: {disposition_counts(extended_results)}")
    print(f"censored-at-1000 resolved by 10,000-step extension: {resolved_by_extension}")
    print(f"censored-at-1000 still censored at 10,000: {still_censored}")
    print(f"max residual node count: official={result['max_residual_node_count_official_horizon']}, "
          f"extended={result['max_residual_node_count_extended_horizon']} (limit={DEFAULT_RESIDUAL_NODE_LIMIT})")


if __name__ == "__main__":
    main()
