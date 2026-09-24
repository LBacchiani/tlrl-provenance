"""Wall-clock overhead of the semantic audit, measured against real rollouts.

Answers a critique the paired PointWorld report did not previously address:
no runtime/overhead data existed anywhere in this work. This script splits
each episode's wall-clock time into two real, separately timed phases using
the exact same PPO checkpoint and adapter functions the confirmatory
campaign used (not a stripped-down substitute):

1. "rollout": policy inference + environment stepping, from ``env.reset``
   through the episode's terminal step -- i.e. the cost that exists with or
   without any semantic audit.
2. "audit": ``build_point_trace`` + ``assess_point_episode`` (grounding
   replay verification, provenance-circuit construction via
   ``evaluate_with_provenance``, and structural certificate verification --
   the exact same call the confirmatory campaign made for every one of its
   5,000 records) -- the audit's own incremental cost.

Must be run under the DeepLTL-compatible interpreter (torch/gymnasium):
    python scripts/measure_deepltl_point_audit_overhead.py

This is a read-only development-analysis script: it does not touch the
frozen confirmatory campaign's plan, records, or summary. It reuses one of
the same five checkpoints and writes a new, separately hash-anchored
artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import statistics as pystats
import sys
import time
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
EPISODES_PER_FORMULA = 3
EVAL_SEED_BASE = 8_900_000
OUTPUT_PATH = (
    ROOT
    / "evidence"
    / "deepltl_point"
    / "confirmatory"
    / "semantic"
    / "deepltl_point_audit_overhead_measurement.json"
)
if os.environ.get("TLRL_OUTPUT_DIR"):  # redirect generated evidence; frozen files stay untouched
    OUTPUT_PATH = Path(os.environ["TLRL_OUTPUT_DIR"]) / OUTPUT_PATH.name


def time_one_episode(env: Any, agent: Any, task: Any, seed: int, policy_id: str) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    rollout_start = time.perf_counter()
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
        if len(steps) > 1000:
            raise RuntimeError("PointLtl2 TimeLimit failed to truncate at 1000 steps")
    rollout_seconds = time.perf_counter() - rollout_start

    audit_start = time.perf_counter()
    trace = build_point_trace(
        steps,
        policy_id=policy_id,
        seed=seed,
        task_id=task.task_id,
        terminated=terminated,
        truncated=truncated,
    )
    assess_point_episode(
        task,
        trace,
        official_success=official_success,
        official_violation=official_violation,
    )
    audit_seconds = time.perf_counter() - audit_start

    return {
        "task_id": task.task_id,
        "steps": len(steps),
        "rollout_seconds": rollout_seconds,
        "audit_seconds": audit_seconds,
        "audit_to_rollout_ratio": audit_seconds / rollout_seconds if rollout_seconds > 0 else float("inf"),
    }


def main() -> None:
    os.chdir(UPSTREAM)
    checkpoint = checkpoint_path(EXPERIMENT, TRAINING_SEED, CHECKPOINT_STEPS)
    tasks = load_tasks()[:NUM_FORMULAS]

    measurements: list[dict[str, Any]] = []
    for task in tasks:
        env, agent = load_agent(EXPERIMENT, TRAINING_SEED, checkpoint, task.source)
        try:
            for episode_index in range(EPISODES_PER_FORMULA):
                seed = EVAL_SEED_BASE + episode_index
                measurements.append(
                    time_one_episode(env, agent, task, seed, policy_id=f"ppo_seed_{TRAINING_SEED}")
                )
        finally:
            env.close()

    rollout_seconds = [m["rollout_seconds"] for m in measurements]
    audit_seconds = [m["audit_seconds"] for m in measurements]
    ratios = [m["audit_to_rollout_ratio"] for m in measurements]

    result = {
        "schema_version": "deepltl-point-audit-overhead-measurement/1",
        "checkpoint": {
            "experiment": EXPERIMENT,
            "training_seed": TRAINING_SEED,
            "checkpoint_steps": CHECKPOINT_STEPS,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
        "device": "cpu",
        "num_episodes": len(measurements),
        "num_formulas": NUM_FORMULAS,
        "episodes_per_formula": EPISODES_PER_FORMULA,
        "rollout_seconds": {
            "mean": pystats.fmean(rollout_seconds),
            "median": pystats.median(rollout_seconds),
            "min": min(rollout_seconds),
            "max": max(rollout_seconds),
        },
        "audit_seconds": {
            "mean": pystats.fmean(audit_seconds),
            "median": pystats.median(audit_seconds),
            "min": min(audit_seconds),
            "max": max(audit_seconds),
        },
        "audit_to_rollout_ratio": {
            "mean": pystats.fmean(ratios),
            "median": pystats.median(ratios),
            "min": min(ratios),
            "max": max(ratios),
        },
        "per_episode": measurements,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_bytes((json.dumps(result, indent=2) + "\n").encode("utf-8"))
    digest = hashlib.sha256(OUTPUT_PATH.read_bytes()).hexdigest()
    (OUTPUT_PATH.parent / f"{OUTPUT_PATH.name}.sha256").write_bytes(
        f"{digest}  {OUTPUT_PATH.name}\n".encode("utf-8")
    )

    print(f"wrote {OUTPUT_PATH} ({digest})")
    print(f"episodes timed: {len(measurements)}")
    print(f"mean rollout: {result['rollout_seconds']['mean']*1000:.1f} ms")
    print(f"mean audit:   {result['audit_seconds']['mean']*1000:.1f} ms")
    print(f"mean ratio (audit/rollout): {result['audit_to_rollout_ratio']['mean']:.3f}")


if __name__ == "__main__":
    main()
