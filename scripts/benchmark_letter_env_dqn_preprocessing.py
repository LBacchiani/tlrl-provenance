#!/usr/bin/env python
"""Verify and time cached LetterEnv DQN preprocessing against DeepLTL.

CUDA work is explicitly synchronized at every timing boundary. Without that,
asynchronous kernels are commonly charged to the following measured region,
which makes per-component profiles misleading.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
DEEPLTL = ROOT / ".runtime" / "deep-ltl" / "src"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(DEEPLTL))
sys.path.insert(0, str(ROOT))

from scripts import train_letter_env_dqn as trainer  # noqa: E402
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
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=930000)
    parser.add_argument(
        "--training-steps",
        type=int,
        default=0,
        help="Also compare two real learner runs in temporary directories.",
    )
    return parser.parse_args()


def synchronize(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(
    operation: Callable[[], Any],
    *,
    torch: Any,
    device: Any,
    repeats: int,
    warmup: int,
) -> tuple[float, Any]:
    result = None
    for _ in range(warmup):
        result = operation()
    synchronize(torch, device)
    timings = []
    for _ in range(repeats):
        synchronize(torch, device)
        started = time.perf_counter()
        result = operation()
        synchronize(torch, device)
        timings.append(time.perf_counter() - started)
    return statistics.median(timings), result


def assert_equivalent(stock: Any, cached: Any, torch: Any) -> None:
    if not torch.equal(stock.features, cached.features):
        raise AssertionError("feature tensors differ")
    if not torch.equal(stock.epsilon_mask, cached.epsilon_mask):
        raise AssertionError("epsilon masks differ")
    stock_reach, stock_avoid = stock.seq.all()
    cached_reach, cached_avoid = cached.seq
    for label, stock_part, cached_part in (
        ("reach", stock_reach, cached_reach),
        ("avoid", stock_avoid, cached_avoid),
    ):
        if not torch.equal(stock_part[0], cached_part[0]):
            raise AssertionError(f"{label} lengths differ")
        if not torch.equal(stock_part[1], cached_part[1]):
            raise AssertionError(f"{label} token tensors differ")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.repeats <= 0 or args.warmup < 0:
        raise ValueError("batch-size/repeats must be positive and warmup nonnegative")

    import preprocessing
    import torch

    device = resolve_device(args.device)
    env, _curriculum, _assignments = make_env(args.seed)
    propositions = set(env.unwrapped.get_propositions())
    observations = [env.reset(seed=args.seed * 100 + index) for index in range(args.batch_size)]

    encode_started = time.perf_counter()
    encoded = [encode_observation(observation, propositions) for observation in observations]
    encode_seconds = time.perf_counter() - encode_started

    stock = preprocessing.preprocess_obss(observations, propositions, device=device)
    cached = collate_cached_observations(encoded, device=device)
    synchronize(torch, device)
    assert_equivalent(stock, cached, torch)

    config = DQNConfig(device=str(device), batch_size=args.batch_size)
    online = build_q_network(env, config).to(device).eval()
    target = build_q_network(env, config).to(device).eval()
    target.load_state_dict(online.state_dict())
    with torch.no_grad():
        stock_q = online(stock)
        cached_q = online(cached)
    synchronize(torch, device)
    if not torch.allclose(stock_q, cached_q, rtol=0.0, atol=0.0):
        raise AssertionError("model outputs differ after cached collation")

    stock_preprocess, _ = measure(
        lambda: preprocessing.preprocess_obss(observations, propositions, device=device),
        torch=torch,
        device=device,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    cached_collate, _ = measure(
        lambda: collate_cached_observations(encoded, device=device),
        torch=torch,
        device=device,
        repeats=args.repeats,
        warmup=args.warmup,
    )

    def stock_update_input_path():
        batch = preprocessing.preprocess_obss(observations, propositions, device=device)
        with torch.no_grad():
            actions = online(batch).argmax(dim=1)
            return target(batch).gather(1, actions.unsqueeze(1)).squeeze(1)

    def cached_update_input_path():
        batch = collate_cached_observations(encoded, device=device)
        with torch.no_grad():
            actions = online(batch).argmax(dim=1)
            return target(batch).gather(1, actions.unsqueeze(1)).squeeze(1)

    stock_target_path, _ = measure(
        stock_update_input_path,
        torch=torch,
        device=device,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    cached_target_path, _ = measure(
        cached_update_input_path,
        torch=torch,
        device=device,
        repeats=args.repeats,
        warmup=args.warmup,
    )

    result = {
                "status": "equivalent",
                "device": str(device),
                "batch_size": args.batch_size,
                "repeats": args.repeats,
                "one_time_encode_ms_per_observation": 1000.0
                * encode_seconds
                / args.batch_size,
                "median_ms": {
                    "stock_preprocess": 1000.0 * stock_preprocess,
                    "cached_collate": 1000.0 * cached_collate,
                    "stock_double_dqn_target_path": 1000.0 * stock_target_path,
                    "cached_double_dqn_target_path": 1000.0 * cached_target_path,
                },
                "speedup": {
                    "preprocess_only": stock_preprocess / cached_collate,
                    "double_dqn_target_path": stock_target_path / cached_target_path,
                },
                "timing_note": "CUDA synchronized at every measurement boundary",
            }

    if args.training_steps:
        if args.training_steps < args.batch_size:
            raise ValueError("training-steps must be at least batch-size")
        training_runs = []
        final_model_states = []
        with tempfile.TemporaryDirectory(prefix="letter-dqn-benchmark-") as temporary:
            original_experiments = trainer.EXPERIMENTS
            trainer.EXPERIMENTS = Path(temporary)
            try:
                for mode in ("upstream", "cached"):
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    config = DQNConfig(
                        total_steps=args.training_steps,
                        num_envs=8,
                        buffer_size=max(5_000, args.training_steps),
                        learning_starts=args.batch_size,
                        batch_size=args.batch_size,
                        train_frequency=4,
                        target_update_frequency=1_000,
                        checkpoint_interval=args.training_steps,
                        log_interval=args.training_steps,
                        seed=939999,
                        name=f"benchmark_{mode}",
                        device=str(device),
                        num_threads=8,
                        replay_preprocessing=mode,
                    )
                    synchronize(torch, device)
                    started = time.perf_counter()
                    run = trainer.train_one_seed(config)
                    synchronize(torch, device)
                    elapsed = time.perf_counter() - started
                    training_runs.append(
                        {
                            "mode": mode,
                            "seconds": elapsed,
                            "steps_per_second": args.training_steps / elapsed,
                            "updates": run["num_updates"],
                            "recent": run["recent"],
                        }
                    )
                    checkpoint = torch.load(
                        Path(run["output_dir"]) / "eval" / f"{args.training_steps}.pth",
                        map_location="cpu",
                    )
                    final_model_states.append(checkpoint["model_state"])
            finally:
                trainer.EXPERIMENTS = original_experiments
        result["real_training_comparison"] = training_runs
        result["real_training_speedup"] = (
            training_runs[0]["seconds"] / training_runs[1]["seconds"]
        )
        result["final_model_bitwise_equal"] = (
            final_model_states[0].keys() == final_model_states[1].keys()
            and all(
                torch.equal(final_model_states[0][key], final_model_states[1][key])
                for key in final_model_states[0]
            )
        )
        result["final_model_max_abs_difference"] = max(
            float(
                (final_model_states[0][key] - final_model_states[1][key])
                .abs()
                .max()
                .item()
            )
            for key in final_model_states[0]
        )
        result["training_equivalence_note"] = (
            "Inputs and pre-update outputs are bit-identical. Separate CUDA GRU "
            "training runs may diverge numerically because deterministic kernels "
            "are not forced by the published training protocol."
        )

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
