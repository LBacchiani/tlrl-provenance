#!/usr/bin/env python
"""Run a development-only real-environment grounding smoke check.

This script deliberately bypasses the LDBA and policy. It checks only the
FlatWorld-v0 simulator schema and proposition reconstruction, so it neither
requires Rabinizer nor observes a methodology-usefulness outcome.

Mirrors ``smoke_letter_env_environment.py`` exactly, adapted for FlatWorld's
continuous circle-containment grounding instead of LetterEnv's discrete grid
lookup: probes teleport the agent to a point independently verified (not
assumed) to sit inside exactly one circle, take the discrete "stay" action
(index 8, zero displacement), and check the reported proposition set.
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

from tlrl_benchmarks.flatworld import capture_official_step  # noqa: E402


UPSTREAM_COMMIT = "3157200b5910cd7f5c493ffba8913994f2207eaf"
# Independently verified (by direct circle-containment search, not by
# inspection -- see tests/test_flatworld_benchmark.py) to ground to exactly
# the single named color under the frozen geometry.
PURE_COLOR_POSITIONS = {
    "aqua": (0.9702791534208636, 0.08198763564735649),
    "red": (-1.4187224366781153, 0.616705631564025),
    "magenta": (-0.6918871148046649, 1.1046868558173903),
    "yellow": (-1.1308972933601777, -1.0465174775056656),
    "orange": (-1.454243842127788, -0.659675941528038),
    "blue": (-0.2037638890890678, 0.36773079721620583),
    "green": (1.0988382879679934, 0.8839839319154412),
}
STAY_ACTION = 8


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
        default=ROOT / "evidence" / "flatworld" / "validation" / "flatworld_environment_smoke.json",
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
    import envs.flatworld  # noqa: F401,E402  (registers FlatWorld-v0)

    env = gymnasium.make("FlatWorld-v0")
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

        # Development-only instrumentation probes exercise each positive
        # label. They are not controller episodes and can never enter
        # usefulness data. FlatWorld is continuous, so a probe teleports the
        # agent directly to a position independently verified to sit inside
        # exactly one circle, then takes the discrete "stay" action so the
        # position does not move before the check.
        for color, position in PURE_COLOR_POSITIONS.items():
            _observation, _info = env.reset(seed=args.seed)
            unwrapped = env.unwrapped
            unwrapped.agent_pos = np.array(position, dtype=float)
            _observation, reward, terminated, truncated, info = env.step(STAY_ACTION)
            captured = capture_official_step(env, info, action=STAY_ACTION, reward=reward)
            observed = sorted(captured.geometry.grounded_propositions)
            if observed != [color]:
                raise AssertionError(
                    f"positive contact probe for {color!r} produced {observed}; "
                    f"position={position}, agent_after_step={unwrapped.agent_pos.tolist()}, info={info}"
                )
            contact_probes[color] = observed
    finally:
        env.close()

    result = {
        "schema_version": "tlrl-flatworld-environment-smoke/1",
        "status": "passed",
        "scope": "development_only_schema_and_grounding_no_policy_no_utility",
        "upstream_commit": commit,
        "environment_id": "FlatWorld-v0",
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
