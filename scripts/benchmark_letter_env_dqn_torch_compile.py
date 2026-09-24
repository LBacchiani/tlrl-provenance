#!/usr/bin/env python
"""Does torch.compile(mode="reduce-overhead") actually help this model?

Real risk being tested, not assumed: "reduce-overhead" uses CUDA graphs,
which require fixed tensor shapes across replays. This model's
next_obs_batch shape depends on how many sampled transitions happen to be
nonterminal -- that varies randomly every update in real training. This
script simulates that exact pattern (a fixed-size obs_batch call plus a
randomly-varying-size next_obs_batch call, repeated many times) rather than
a single best-case fixed shape, and reports whether torch.compile is
actually faster once compilation/recompilation cost is included -- not the
theoretical steady-state number after warmup alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEEPLTL = ROOT / ".runtime" / "deep-ltl" / "src"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(DEEPLTL))
sys.path.insert(0, str(ROOT))

from scripts.train_letter_env_dqn import make_env  # noqa: E402
from tlrl_benchmarks.letter_env.dqn import (  # noqa: E402
    DQNConfig,
    build_q_network,
    collate_cached_observations,
    encode_observation,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--seed", type=int, default=930000)
    return parser.parse_args()


def synchronize(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    import torch

    device = resolve_device(args.device)
    env, _curriculum, _ = make_env(args.seed)
    propositions = set(env.unwrapped.get_propositions())
    config = DQNConfig(device=str(device), seed=args.seed, batch_size=args.batch_size)
    torch.manual_seed(args.seed)

    pool_observations = [env.reset(seed=args.seed * 100 + i) for i in range(args.batch_size * 2)]
    pool_encoded = [encode_observation(o, propositions) for o in pool_observations]
    rng = random.Random(args.seed)

    def random_next_batch(size: int):
        picks = rng.sample(pool_encoded, size)
        return collate_cached_observations(picks, device=device)

    fixed_obs_batch = collate_cached_observations(pool_encoded[: args.batch_size], device=device)

    def run_model_pattern(model: Any) -> float:
        """One 'update': a fixed-size forward, then a randomly-sized one --
        exactly the obs_batch / next_obs_batch shape pattern real training
        produces every update."""

        with torch.no_grad():
            _ = model(fixed_obs_batch)
            live_count = rng.randint(args.batch_size - 20, args.batch_size)
            next_batch = random_next_batch(live_count)
            _ = model(next_batch)
        return live_count

    result: dict[str, Any] = {"batch_size": args.batch_size, "device": str(device), "iterations": args.iterations}

    torch.manual_seed(args.seed)
    eager_model = build_q_network(env, config).to(device).eval()
    eager_timings = []
    for i in range(args.iterations):
        synchronize(torch, device)
        started = time.perf_counter()
        run_model_pattern(eager_model)
        synchronize(torch, device)
        eager_timings.append(time.perf_counter() - started)
    result["eager"] = {
        "first_iteration_ms": 1000.0 * eager_timings[0],
        "median_ms_all": 1000.0 * statistics.median(eager_timings),
        "median_ms_last_20": 1000.0 * statistics.median(eager_timings[-20:]),
        "total_seconds": sum(eager_timings),
    }

    compile_error = None
    try:
        torch.manual_seed(args.seed)
        compiled_model = build_q_network(env, config).to(device).eval()
        compiled_model = torch.compile(compiled_model, mode="reduce-overhead")
        compiled_timings = []
        for i in range(args.iterations):
            synchronize(torch, device)
            started = time.perf_counter()
            run_model_pattern(compiled_model)
            synchronize(torch, device)
            compiled_timings.append(time.perf_counter() - started)
        result["compiled"] = {
            "first_iteration_ms": 1000.0 * compiled_timings[0],
            "median_ms_all": 1000.0 * statistics.median(compiled_timings),
            "median_ms_last_20": 1000.0 * statistics.median(compiled_timings[-20:]),
            "total_seconds": sum(compiled_timings),
            "per_iteration_ms": [round(1000.0 * t, 2) for t in compiled_timings],
        }
        result["speedup_total_wallclock"] = result["eager"]["total_seconds"] / result["compiled"]["total_seconds"]
        result["speedup_steady_state_median"] = (
            result["eager"]["median_ms_last_20"] / result["compiled"]["median_ms_last_20"]
        )
    except Exception as exc:  # noqa: BLE001 -- want to report any failure mode, not just known ones
        compile_error = f"{type(exc).__name__}: {exc}"

    result["compile_error"] = compile_error
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
