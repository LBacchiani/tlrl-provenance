#!/usr/bin/env python
"""Run a development-only real-environment grounding smoke check.

This script deliberately bypasses the LDBA and policy. It checks only the
PointLtl2-v0 simulator schema and proposition reconstruction, so it neither
requires Rabinizer nor observes a methodology-usefulness outcome.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tlrl_benchmarks.deepltl_point import capture_official_step  # noqa: E402


UPSTREAM_COMMIT = "3157200b5910cd7f5c493ffba8913994f2207eaf"


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evidence" / "deepltl_point" / "validation" / "deepltl_point_environment_smoke.json",
    )
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    upstream = args.upstream.resolve()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=upstream, text=True
    ).strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"upstream commit mismatch: {commit}")
    sys.path.insert(0, str(upstream / "src"))

    import gymnasium  # noqa: E402
    import mujoco  # noqa: E402
    import numpy as np  # noqa: E402
    import safety_gymnasium  # noqa: E402
    from envs.zones.safety_gym_wrapper import SafetyGymWrapper  # noqa: E402

    env = SafetyGymWrapper(safety_gymnasium.make("PointLtl2-v0", render_mode=None))
    labels = Counter()
    contact_probes: dict[str, list[str]] = {}
    episodes = 0
    checked = 0
    try:
        _observation, _info = env.reset(seed=args.seed)
        env.action_space.seed(args.seed)
        for _ in range(args.steps):
            action = env.action_space.sample()
            _observation, reward, terminated, truncated, info = env.step(action)
            captured = capture_official_step(env, info, action=action.tolist(), reward=reward)
            labels[",".join(sorted(captured.geometry.grounded_propositions)) or "<empty>"] += 1
            checked += 1
            if terminated or truncated:
                episodes += 1
                _observation, _info = env.reset()
        episodes += 1

        # Development-only instrumentation probes exercise each positive label.
        # They are not controller episodes and can never enter usefulness data.
        for color in sorted(("blue", "green", "magenta", "yellow")):
            _observation, _info = env.reset(seed=args.seed)
            task = env.unwrapped.task
            geom = next(item for item in task._geoms.values() if getattr(item, "color_name", None) == color)
            target = np.asarray(geom.pos[0][:2], dtype=float)
            agent = task.agent
            displacement = target - np.asarray(agent.pos[:2], dtype=float)
            agent_body = agent.engine.model.body("agent")
            original_body_position = agent_body.pos.copy()
            agent_body.pos[:2] += displacement
            agent.engine.data.qvel[:] = 0.0
            mujoco.mj_forward(agent.engine.model, agent.engine.data)
            pre_step_distance = float(agent.dist_xy(target))
            pre_step_cost = dict(geom.cal_cost())
            zero_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
            _observation, reward, terminated, truncated, info = env.step(zero_action)
            captured = capture_official_step(
                env, info, action=zero_action.tolist(), reward=reward
            )
            observed = sorted(captured.geometry.grounded_propositions)
            if observed != [color]:
                raise AssertionError(
                    f"positive contact probe for {color} produced {observed}; "
                    f"pre_step_distance={pre_step_distance}, pre_step_cost={pre_step_cost}, "
                    f"displacement={displacement.tolist()}, "
                    f"post_step_distance={float(agent.dist_xy(target))}, info={info}"
                )
            contact_probes[color] = observed
            agent_body.pos[:] = original_body_position
            mujoco.mj_forward(agent.engine.model, agent.engine.data)
    finally:
        env.close()

    result = {
        "schema_version": "tlrl-deepltl-point-environment-smoke/1",
        "status": "passed",
        "scope": "development_only_schema_and_grounding_no_policy_no_utility",
        "upstream_commit": commit,
        "environment_id": "PointLtl2-v0",
        "seed": args.seed,
        "requested_steps": args.steps,
        "checked_steps": checked,
        "episodes_touched": episodes,
        "geometry_info_mismatches": 0,
        "label_counts": dict(sorted(labels.items())),
        "positive_contact_probes": contact_probes,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "gymnasium": gymnasium.__version__,
            "mujoco": mujoco.__version__,
            "safety_gymnasium": safety_gymnasium.__version__,
        },
    }
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write(args.output, payload)
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    atomic_write(
        sidecar,
        f"{hashlib.sha256(payload).hexdigest()}  {args.output.name}\n".encode("ascii"),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
