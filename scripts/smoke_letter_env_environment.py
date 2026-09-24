#!/usr/bin/env python
"""Run a development-only real-environment grounding smoke check.

This script deliberately bypasses the LDBA and policy. It checks only the
LetterEnv-v0 simulator schema and proposition reconstruction, so it neither
requires Rabinizer nor observes a methodology-usefulness outcome.

Mirrors ``smoke_deepltl_point_environment.py`` exactly, adapted for
LetterEnv's grid/dictionary grounding instead of PointLtl2's zone geometry.
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

from tlrl_benchmarks.letter_env import capture_official_step  # noqa: E402


UPSTREAM_COMMIT = "3157200b5910cd7f5c493ffba8913994f2207eaf"
LETTERS = tuple("abcdefghijkl")


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
        default=ROOT / "evidence" / "letter_env" / "validation" / "letter_env_environment_smoke.json",
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
    import numpy as np  # noqa: E402
    import envs.letter_world  # noqa: F401,E402  (registers LetterEnv-v0)

    env = gymnasium.make("LetterEnv-v0")
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
            captured = capture_official_step(env, info, action=int(action), reward=reward)
            labels[",".join(sorted(captured.geometry.grounded_propositions)) or "<empty>"] += 1
            checked += 1
            if terminated or truncated:
                episodes += 1
                _observation, _info = env.reset()
        episodes += 1

        # Development-only instrumentation probes exercise each positive label.
        # They are not controller episodes and can never enter usefulness data.
        # LetterEnv is a discrete torus grid, so a probe is: teleport the agent
        # to the cell one step away from a known letter cell, then take the
        # single action that lands exactly on it.
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for letter in LETTERS:
            _observation, _info = env.reset(seed=args.seed)
            unwrapped = env.unwrapped
            target_cell = next(cell for cell, value in unwrapped.map.items() if value == letter)
            di, dj = actions[0]  # move "up" (-1, 0) into the target
            source_cell = (
                (target_cell[0] - di) % unwrapped.grid_size,
                (target_cell[1] - dj) % unwrapped.grid_size,
            )
            unwrapped.agent = source_cell
            _observation, reward, terminated, truncated, info = env.step(0)
            captured = capture_official_step(env, info, action=0, reward=reward)
            observed = sorted(captured.geometry.grounded_propositions)
            if observed != [letter]:
                raise AssertionError(
                    f"positive contact probe for {letter!r} produced {observed}; "
                    f"target_cell={target_cell}, source_cell={source_cell}, "
                    f"agent_after_step={unwrapped.agent}, info={info}"
                )
            contact_probes[letter] = observed
    finally:
        env.close()

    result = {
        "schema_version": "tlrl-letter-env-environment-smoke/1",
        "status": "passed",
        "scope": "development_only_schema_and_grounding_no_policy_no_utility",
        "upstream_commit": commit,
        "environment_id": "LetterEnv-v0",
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
