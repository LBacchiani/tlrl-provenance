"""Stress test: does the audit's fail-closed agreement guarantee survive
stochastic (non-deterministic) policy execution, not just the deterministic
rollouts the confirmatory campaigns used?

The confirmatory campaigns enforce ``deterministic=True`` throughout (a
declared, disclosed scope boundary -- see paper \\S Threats to Validity).
This script does not extend that scope claim; it answers a narrower,
genuinely checkable question: does anything in the audit engine or the
adapter's grounding/agreement checks implicitly assume determinism, such
that stochastic rollouts would silently misbehave rather than being audited
correctly (or failing loudly)? It reuses one already-trained PointLTL PPO
checkpoint (seed 920001, same as the audit-overhead measurement) with
``deterministic=False`` and lets ``assess_point_episode``'s existing
fail-closed assertions be the judge: any official/independent disagreement
raises immediately rather than being silently accepted.

Must be run under the DeepLTL-compatible interpreter:
    python scripts/stress_test_stochastic_execution.py
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import traceback
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

EXPERIMENT = "v3_confirm_ppo_5m_seed_920001"
TRAINING_SEED = 920001
CHECKPOINT_STEPS = 4_980_736
NUM_FORMULAS = 10
EPISODES_PER_FORMULA = 5
EVAL_SEED_BASE = 8_950_000
OUTPUT_PATH = (
    ROOT
    / "evidence"
    / "deepltl_point"
    / "confirmatory"
    / "semantic"
    / "deepltl_point_stochastic_execution_stress_test.json"
)
if os.environ.get("TLRL_OUTPUT_DIR"):  # redirect generated evidence; frozen files stay untouched
    OUTPUT_PATH = Path(os.environ["TLRL_OUTPUT_DIR"]) / OUTPUT_PATH.name


def run_one_episode(env: Any, agent: Any, task: Any, seed: int, policy_id: str) -> dict[str, Any]:
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
        # The only change from the confirmatory/overhead harnesses:
        # deterministic=False samples from the policy distribution instead
        # of taking its mode/argmax.
        action = agent.get_action(obs, info, deterministic=False)
        action = action.flatten()
        if action.shape == (1,):
            action = action[0]
        obs, reward, terminated, truncated, info = env.step(action)
        steps.append(capture_official_step(env, info, action=action.tolist(), reward=float(reward)))
        official_success = official_success or bool(info.get("success", False))
        official_violation = official_violation or bool(info.get("violation", False))
        if len(steps) > 1000:
            raise RuntimeError("PointLtl2 TimeLimit failed to truncate at 1000 steps")

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
    return {
        "task_id": task.task_id,
        "steps": len(steps),
        "official_success": official_success,
        "official_violation": official_violation,
        "prefix_status": assessment.prefix.status.value,
        "disposition": assessment.disposition.value,
    }


def main() -> None:
    os.chdir(UPSTREAM)
    checkpoint = checkpoint_path(EXPERIMENT, TRAINING_SEED, CHECKPOINT_STEPS)
    tasks = load_tasks()[:NUM_FORMULAS]

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for task in tasks:
        env, agent = load_agent(EXPERIMENT, TRAINING_SEED, checkpoint, task.source)
        try:
            for episode_index in range(EPISODES_PER_FORMULA):
                seed = EVAL_SEED_BASE + episode_index
                try:
                    results.append(run_one_episode(env, agent, task, seed, policy_id=f"ppo_seed_{TRAINING_SEED}_stochastic"))
                except AssertionError as exc:
                    failures.append(
                        {
                            "task_id": task.task_id,
                            "episode_index": episode_index,
                            "seed": seed,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
        finally:
            env.close()

    dispositions: dict[str, int] = {}
    for row in results:
        dispositions[row["disposition"]] = dispositions.get(row["disposition"], 0) + 1

    result = {
        "schema_version": "deepltl-point-stochastic-execution-stress-test/1",
        "purpose": (
            "Checks whether the audit's fail-closed official/independent "
            "agreement guarantee (assess_point_episode) holds under "
            "deterministic=False (stochastic) policy execution, which the "
            "confirmatory campaigns never use. Not a claim about stochastic "
            "P_sat or a replacement confirmatory result."
        ),
        "checkpoint": {
            "experiment": EXPERIMENT,
            "training_seed": TRAINING_SEED,
            "checkpoint_steps": CHECKPOINT_STEPS,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
        "deterministic": False,
        "num_formulas": NUM_FORMULAS,
        "episodes_per_formula": EPISODES_PER_FORMULA,
        "num_episodes_attempted": NUM_FORMULAS * EPISODES_PER_FORMULA,
        "num_episodes_completed_without_assertion_failure": len(results),
        "num_assertion_failures": len(failures),
        "disposition_counts": dispositions,
        "failures": failures,
        "per_episode": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(OUTPUT_PATH.read_bytes()).hexdigest()
    (OUTPUT_PATH.parent / f"{OUTPUT_PATH.name}.sha256").write_text(
        f"{digest}  {OUTPUT_PATH.name}\n", encoding="utf-8"
    )

    print(f"wrote {OUTPUT_PATH} ({digest})")
    print(f"episodes attempted: {NUM_FORMULAS * EPISODES_PER_FORMULA}")
    print(f"completed without assertion failure: {len(results)}")
    print(f"assertion failures (official/independent disagreement): {len(failures)}")
    print(f"disposition counts: {dispositions}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
