"""Stress test: how sensitive is the audited verdict to synthetic label
noise, simulating a noisy or learned proposition grounding instead of this
project's exact-simulator grounding?

"Noisy or learned groundings" are a declared, disclosed scope boundary (see
paper \\S Threats to Validity): every benchmark in this project grounds
propositions from exact retained simulator state, cross-checked against the
official monitor. This script does not remove that boundary -- it does not
claim the engine detects sensor noise, since nothing here gives it an
independent second observation to detect against. It answers a narrower,
honest question: reusing 500 already-frozen, real confirmatory PointLTL
episodes (no new rollouts, no retraining), how often does independently
perturbing the recorded per-step proposition labels at a given noise rate
change the audited verdict relative to the true, unperturbed grounding? This
is a sensitivity measurement, not a robustness guarantee or a noise-recovery
claim.

Pure post-hoc analysis of already-frozen data; runs under the ordinary
project environment (no torch/DeepLTL needed):
    python scripts/stress_test_grounding_noise.py
"""

from __future__ import annotations

import os
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tlrl_provenance import evaluate_with_provenance, parse_formula  # noqa: E402
from tlrl_provenance.prefix import assess_prefix  # noqa: E402
from tlrl_provenance.trace import LabeledTrace  # noqa: E402

RECORDS_PATH = (
    ROOT
    / "evidence"
    / "deepltl_point"
    / "confirmatory"
    / "semantic"
    / "deepltl_point_full_confirmatory_semantic_records.jsonl"
)
OUTPUT_PATH = (
    ROOT
    / "evidence"
    / "deepltl_point"
    / "confirmatory"
    / "semantic"
    / "deepltl_point_grounding_noise_stress_test.json"
)
if os.environ.get("TLRL_OUTPUT_DIR"):  # redirect generated evidence; frozen files stay untouched
    OUTPUT_PATH = Path(os.environ["TLRL_OUTPUT_DIR"]) / OUTPUT_PATH.name

SAMPLE_SIZE = 500
NOISE_RATES = (0.01, 0.02, 0.05, 0.10)
TRIALS_PER_RATE = 3
COLORS = ("blue", "green", "magenta", "yellow")
SEED = 0


def load_sample(path: Path, n: int, seed: int) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        all_records = [json.loads(line) for line in handle]
    rng = random.Random(seed)
    rng.shuffle(all_records)
    return all_records[:n]


def corrupt_labels(true_labels: list[list[str]], rate: float, rng: random.Random) -> list[list[str]]:
    """Independently, per step, with probability `rate`, replace the
    reported label with a uniformly random *different* one from the
    zero-or-one-color alphabet (empty, or exactly one of the four colors),
    simulating a noisy per-step classifier rather than a targeted attack."""

    alphabet: list[list[str]] = [[]] + [[c] for c in COLORS]
    corrupted = []
    for labels in true_labels:
        current = sorted(labels)
        if rng.random() < rate:
            choices = [alt for alt in alphabet if sorted(alt) != current]
            corrupted.append(rng.choice(choices))
        else:
            corrupted.append(labels)
    return corrupted


def independent_verdict(formula_text: str, labels: list[list[str]], terminated: bool, truncated: bool) -> dict[str, Any]:
    formula = parse_formula(formula_text)
    trace = LabeledTrace(labels, terminated=terminated, truncated=truncated)
    certificate = evaluate_with_provenance(formula, trace, require_grounding=False, verify_reference="direct")
    prefix = assess_prefix(formula, trace)
    return {"satisfied": certificate.satisfied, "prefix_status": prefix.status.value}


def main() -> None:
    records = load_sample(RECORDS_PATH, SAMPLE_SIZE, SEED)
    rng = random.Random(SEED)

    baseline_mismatches = 0
    per_rate: dict[str, dict[str, Any]] = {}
    for rate in NOISE_RATES:
        flips_satisfied = 0
        flips_prefix_status = 0
        total_trials = 0
        for record in records:
            true_labels = record["retained_trace"]["reported_labels"]
            terminated = bool(record["terminated"])
            truncated = bool(record["truncated"])
            formula_text = record["formula"]

            baseline = independent_verdict(formula_text, true_labels, terminated, truncated)
            # Sanity: baseline recomputation must agree with the frozen
            # record's own independent verdict; a disagreement here would
            # mean this script itself has a bug, not a noise effect.
            expected_satisfied = record["independent_verdict"] == "satisfied"
            if baseline["satisfied"] != expected_satisfied:
                baseline_mismatches += 1

            for _trial in range(TRIALS_PER_RATE):
                noisy_labels = corrupt_labels(true_labels, rate, rng)
                noisy = independent_verdict(formula_text, noisy_labels, terminated, truncated)
                total_trials += 1
                if noisy["satisfied"] != baseline["satisfied"]:
                    flips_satisfied += 1
                if noisy["prefix_status"] != baseline["prefix_status"]:
                    flips_prefix_status += 1

        per_rate[str(rate)] = {
            "noise_rate": rate,
            "total_trials": total_trials,
            "verdict_flips": flips_satisfied,
            "verdict_flip_fraction": flips_satisfied / total_trials,
            "prefix_status_flips": flips_prefix_status,
            "prefix_status_flip_fraction": flips_prefix_status / total_trials,
        }

    result = {
        "schema_version": "deepltl-point-grounding-noise-stress-test/1",
        "purpose": (
            "Measures how often synthetic per-step label noise changes the "
            "audited verdict, using 500 real frozen PointLTL episodes and "
            "no new rollouts. A sensitivity measurement, not a noise-"
            "detection or noise-recovery claim; exact-simulator grounding "
            "remains this project's only evaluated grounding source."
        ),
        "source_records": RECORDS_PATH.name,
        "source_records_sha256": hashlib.sha256(RECORDS_PATH.read_bytes()).hexdigest(),
        "sample_size": SAMPLE_SIZE,
        "trials_per_rate": TRIALS_PER_RATE,
        "seed": SEED,
        "baseline_recomputation_mismatches": baseline_mismatches,
        "results_by_noise_rate": per_rate,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(OUTPUT_PATH.read_bytes()).hexdigest()
    (OUTPUT_PATH.parent / f"{OUTPUT_PATH.name}.sha256").write_text(
        f"{digest}  {OUTPUT_PATH.name}\n", encoding="utf-8"
    )

    print(f"wrote {OUTPUT_PATH} ({digest})")
    print(f"sample size: {SAMPLE_SIZE}, baseline recomputation mismatches: {baseline_mismatches}")
    for rate_key, row in per_rate.items():
        print(f"  noise_rate={row['noise_rate']}: verdict flip fraction={row['verdict_flip_fraction']:.4f} "
              f"({row['verdict_flips']}/{row['total_trials']})")


if __name__ == "__main__":
    main()
