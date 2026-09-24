#!/usr/bin/env python
"""Exercise DeepLTL's official FlatWorld LDBA wrapper end to end.

This is a development-only compatibility and instrumentation check. It runs
no learned policy and inspects no methodology-usefulness outcome. Mirrors
``smoke_letter_env_official_ldba.py`` exactly, adapted for FlatWorld's
continuous position instead of LetterEnv's discrete grid: probes teleport
the agent directly to a position independently verified to sit inside
exactly one circle, then take the discrete "stay" action (index 8, zero
displacement) so the position does not move before the check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tlrl_benchmarks.flatworld import (  # noqa: E402
    capture_official_step,
    preserve_termination_and_truncation,
)


UPSTREAM_COMMIT = "3157200b5910cd7f5c493ffba8913994f2207eaf"
RABINIZER_SHA256 = "46efc53d769d1016c75b6b76f7cb2db259d4426053591993a62aa874f3ef486d"
FORMULA = "F (magenta & F red)"
# Independently verified (see tests/test_flatworld_benchmark.py and
# smoke_flatworld_environment.py) to ground to exactly one color each.
MAGENTA_POSITION = (-0.6918871148046649, 1.1046868558173903)
RED_POSITION = (-1.4187224366781153, 0.616705631564025)
STAY_ACTION = 8


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    parser.add_argument("--jar", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evidence" / "flatworld" / "validation" / "flatworld_official_ldba_env_smoke.json",
    )
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    upstream = args.upstream.resolve()
    jar = args.jar.resolve()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=upstream, text=True
    ).strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"upstream commit mismatch: {commit}")
    jar_digest = sha256(jar)
    if jar_digest != RABINIZER_SHA256:
        raise RuntimeError(f"Rabinizer jar SHA-256 mismatch: {jar_digest}")

    os.environ["RABINIZER_JAR"] = str(jar)
    sys.path.insert(0, str(upstream / "src"))

    import gymnasium  # noqa: E402
    import numpy as np  # noqa: E402
    from envs import make_env  # noqa: E402
    from ltl import FixedSampler  # noqa: E402

    official_env = make_env(
        "FlatWorld-v0",
        FixedSampler.partial(FORMULA),
        render_mode=None,
    )
    # The published training API collapses truncation into done. Auditable
    # collection removes only that outer compatibility wrapper.
    env = preserve_termination_and_truncation(official_env)
    checked_steps = 0
    state_changes = 0
    accepting_steps = 0
    terminations = 0
    truncations = 0
    observed_labels: set[str] = set()
    forced_sequence: list[dict[str, object]] = []
    try:
        observation, info = env.reset(seed=args.seed)
        env.action_space.seed(args.seed)
        ldba = observation["ldba"]
        if not ldba.check_valid():
            raise AssertionError("official wrapper constructed an invalid LDBA")
        if observation["goal"] != FORMULA:
            raise AssertionError("official wrapper changed the fixed task formula")
        if observation["ldba_state"] != ldba.initial_state:
            raise AssertionError("official wrapper did not expose the initial LDBA state")

        for _ in range(args.steps):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            captured = capture_official_step(env, info, action=int(action), reward=reward)
            observed_labels.update(captured.geometry.grounded_propositions)
            checked_steps += 1
            state_changes += int(bool(info.get("ldba_state_changed", False)))
            accepting_steps += int(bool(info.get("accepting", False)))
            terminations += int(terminated)
            truncations += int(truncated)
            if observation["propositions"] != info["propositions"]:
                raise AssertionError("observation and info proposition labels disagree")
            if observation["ldba_state"] != env.env.ldba_state:
                raise AssertionError("observation and wrapper LDBA states disagree")
            if terminated or truncated:
                observation, info = env.reset()

        # A deterministic instrumentation probe makes the automaton traverse
        # the ordered-eventuality task "F (magenta & F red)": visit
        # 'magenta', then 'red'. FlatWorld is continuous, so the probe
        # teleports the agent directly to a position verified to sit inside
        # exactly one circle, then takes the discrete "stay" action.
        # Mirroring LetterEnv's and PointLtl2's probes, the final target may
        # need visiting twice: reaching the accepting sink state is not
        # itself flagged accepting until the Büchi self-loop at the sink is
        # traversed once more -- checked empirically below, not assumed.
        observation, info = env.reset(seed=args.seed)
        for color, position in (("magenta", MAGENTA_POSITION), ("red", RED_POSITION), ("red", RED_POSITION)):
            unwrapped = env.unwrapped
            unwrapped.agent_pos = np.array(position, dtype=float)
            observation, reward, terminated, truncated, info = env.step(np.int64(STAY_ACTION))
            captured = capture_official_step(env, info, action=STAY_ACTION, reward=reward)
            actual_labels = sorted(captured.geometry.grounded_propositions)
            actual_state = int(observation["ldba_state"])
            if actual_labels != [color]:
                raise AssertionError(f"forced {color!r} contact produced labels {actual_labels}")
            if truncated:
                raise AssertionError("forced LDBA traversal unexpectedly truncated")
            forced_sequence.append(
                {
                    "color": color,
                    "ldba_state": actual_state,
                    "accepting": bool(info["accepting"]),
                    "terminated": bool(terminated),
                }
            )
            if terminated:
                break
        if not forced_sequence[-1]["accepting"] or not forced_sequence[-1]["terminated"]:
            raise AssertionError("accepting LDBA transition did not terminate the finite task")
    finally:
        env.close()

    result = {
        "schema_version": "tlrl-flatworld-official-ldba-env-smoke/1",
        "status": "passed",
        "scope": "development_only_official_wrapper_no_learned_policy_no_utility",
        "upstream_commit": commit,
        "rabinizer_jar_sha256": jar_digest,
        "environment_id": "FlatWorld-v0",
        "formula": FORMULA,
        "seed": args.seed,
        "checked_steps": checked_steps,
        "geometry_info_mismatches": 0,
        "ldba_valid": True,
        "ldba_states": ldba.num_states,
        "ldba_transitions": ldba.num_transitions,
        "ldba_state_changes": state_changes,
        "accepting_steps": accepting_steps,
        "environment_terminations": terminations,
        "time_limit_truncations": truncations,
        "observed_positive_labels": sorted(observed_labels),
        "forced_ordered_eventuality_probe": forced_sequence,
        "termination_truncation_preserved": True,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "gymnasium": gymnasium.__version__,
        },
    }
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write(args.output, payload)
    atomic_write(
        args.output.with_suffix(args.output.suffix + ".sha256"),
        f"{hashlib.sha256(payload).hexdigest()}  {args.output.name}\n".encode("ascii"),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
