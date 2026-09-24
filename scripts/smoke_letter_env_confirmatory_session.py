#!/usr/bin/env python
"""Persistent regression smoke for the LetterEnv DQN confirmatory session.

This exercises exactly what tests/test_letter_env_confirmatory.py cannot
cover without torch/DeepLTL installed: sequence refresh via SequenceSearch,
DQN value estimation (max_a Q(s,a)) for LDBA candidate-sequence ranking,
generated-map recording, and a complete episode through
capture_official_step -> build_letter_trace -> assess_letter_episode.

Requires a trained DQN checkpoint (any checkpoint works, including a short
--smoke-steps one from train_letter_env_dqn.py) and the DeepLTL-compatible
interpreter, e.g.:

    python scripts/smoke_letter_env_confirmatory_session.py --seed 930001 --checkpoint-steps 250000 \\
        --experiment letter_env_dqn_v2_cached_curriculum_recency_fix_evidence_harmonized

This is a development-only compatibility and correctness check. It does not
freeze or execute any confirmatory campaign.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tlrl_benchmarks.letter_env.confirmatory import LetterEnvDQNConfirmatoryAdapter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint-steps", type=int, required=True)
    parser.add_argument("--formulas", nargs="*", default=["letter_env_000", "letter_env_018", "letter_env_002"])
    args = parser.parse_args()

    config = {
        "upstream": ".runtime/deep-ltl",
        "rabinizer_jar": "rabinizer-4/lib/rabinizer.jar",
        "subjects": [
            {
                "subject_id": f"dqn_seed_{args.seed}",
                "training_seed": args.seed,
                "experiment": args.experiment,
                "checkpoint_steps": args.checkpoint_steps,
            }
        ],
    }
    adapter = LetterEnvDQNConfirmatoryAdapter(config=config, root=ROOT)
    subject = adapter.subjects()[0]
    by_id = {t.task_id: t for t in adapter.tasks()}

    results = []
    for task_id in args.formulas:
        task = by_id[task_id]
        session = adapter.open_session(subject, task)
        try:
            episodes = [
                session.run_episode(evaluation_seed=100 + i, scenario_seed=9000 + i, deterministic=True, verification_mode="full")
                for i in range(2)
            ]
        finally:
            session.close()
        maps = {episode["generated_map_canonical_sha256"] for episode in episodes}
        if len(maps) != len(episodes):
            raise AssertionError(f"{task_id}: distinct scenario_seeds produced identical maps: {maps}")
        results.append(
            {
                "task_id": task_id,
                "formula": task.formula,
                "episodes": [
                    {
                        "scenario_seed": e["scenario_seed"],
                        "steps": e["steps"],
                        "disposition": e["disposition"],
                        "independent_verdict": e["independent_verdict"],
                        "official_independent_mismatch": e["official_independent_mismatch"],
                        "necessary_evidence_count": e["necessary_evidence_count"],
                        "possible_evidence_count": e["possible_evidence_count"],
                        "generated_map_canonical_sha256": e["generated_map_canonical_sha256"],
                    }
                    for e in episodes
                ],
            }
        )

    # Cross-formula pairing check: same scenario_seed must reproduce the same
    # map even across different sessions/formulas (each opens a fresh env).
    by_scenario: dict[int, set[str]] = {}
    for entry in results:
        for episode in entry["episodes"]:
            by_scenario.setdefault(episode["scenario_seed"], set()).add(episode["generated_map_canonical_sha256"])
    for scenario_seed, digests in by_scenario.items():
        if len(digests) != 1:
            raise AssertionError(f"scenario_seed {scenario_seed} produced different maps across formulas: {digests}")

    print(json.dumps({"status": "passed", "results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
