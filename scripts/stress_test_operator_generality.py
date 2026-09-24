"""Stress test: exercise the audit engine on formulas using
operators the confirmatory corpus never exercises (G, R, W, X, Xw -- the
confirmatory corpus is restricted to nested F and U, a declared, disclosed
scope boundary; see paper \\S Threats to Validity), against real RL-generated
traces rather than only the synthetic exhaustive-suite traces already
covering these operators (see evidence/prefix_exact_completeness_verification.json)?

This checks implementation coverage; it does not establish empirical
generalization of the confirmatory findings beyond the F/U corpus. It does
not change the confirmatory corpus, does not require the trained
policy to pursue these formulas (that would need Rabinizer/LDBA support for
whatever concrete syntax DeepLTL's environment wrapper expects, a separate
and unrelated question), and produces no P_sat claim. It collects real
grounded traces from an already-trained PointLtl2 PPO checkpoint navigating
toward the *existing* frozen corpus formulas (so trajectories are realistic,
not synthetic), then independently audits each collected trace against five
hand-authored formulas, one per non-F/U operator, using this project's own
engine (`tlrl_provenance.parse_formula` / `evaluate_with_provenance` /
`assess_prefix`) exactly as it is used everywhere else in this project.

Must be run under the DeepLTL-compatible interpreter:
    python scripts/stress_test_operator_generality.py
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
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
    build_point_trace,
    capture_official_step,
    load_tasks,
    point_zone_rules,
)
from tlrl_provenance import evaluate_with_provenance, parse_formula  # noqa: E402
from tlrl_provenance.prefix import assess_prefix  # noqa: E402
from tlrl_provenance.verification import verify_certificate  # noqa: E402
from tlrl_provenance.grounding import verify_grounding_reconstruction  # noqa: E402

EXPERIMENT = "v3_confirm_ppo_5m_seed_920001"
TRAINING_SEED = 920001
CHECKPOINT_STEPS = 4_980_736
NUM_STEERING_FORMULAS = 5
EPISODES_PER_STEERING_FORMULA = 3
EVAL_SEED_BASE = 8_990_000
OUTPUT_PATH = (
    ROOT
    / "evidence"
    / "deepltl_point"
    / "confirmatory"
    / "semantic"
    / "deepltl_point_operator_generality_stress_test.json"
)
if os.environ.get("TLRL_OUTPUT_DIR"):  # redirect generated evidence; frozen files stay untouched
    OUTPUT_PATH = Path(os.environ["TLRL_OUTPUT_DIR"]) / OUTPUT_PATH.name

# One hand-authored probe formula per operator the confirmatory corpus never
# exercises, grounded in PointLtl2's real four colors.
PROBE_FORMULAS = {
    "globally": "G (! magenta)",
    "release": "(! magenta) R blue",
    "weak_until": "(! magenta) W blue",
    "weak_next_boundary": "G (Xw (! magenta))",
    "strong_next_boundary": "G (X (! magenta))",
}


def run_steering_episode(env: Any, agent: Any, task: Any, seed: int, policy_id: str) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs, info = env.reset(seed=seed)
    agent.reset()
    steps = []
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = agent.get_action(obs, info, deterministic=True).flatten()
        if action.shape == (1,):
            action = action[0]
        obs, reward, terminated, truncated, info = env.step(action)
        steps.append(capture_official_step(env, info, action=action.tolist(), reward=float(reward)))
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
    return trace


def audit_probe(formula_text: str, trace: Any) -> dict[str, Any]:
    formula = parse_formula(formula_text)
    rules = point_zone_rules()
    grounding_report = verify_grounding_reconstruction(trace, rules)
    if not grounding_report.valid:
        raise AssertionError(f"grounding replay failed: {grounding_report.errors}")
    certificate = evaluate_with_provenance(formula, trace, require_grounding=True, verify_reference="direct")
    verification = verify_certificate(certificate)
    if not verification.valid:
        raise AssertionError(f"certificate verification failed: {verification.errors}")
    prefix = assess_prefix(formula, trace)
    if not prefix.decision_complete:
        raise AssertionError(
            "continuation-aware status was not decided "
            f"(origin={prefix.decision_origin.value}, "
            f"states_explored={prefix.states_explored}, "
            f"alphabet_size={prefix.alphabet_size})"
        )
    return {
        "satisfied": certificate.satisfied,
        "prefix_status": prefix.status.value,
        "prefix_decision_origin": prefix.decision_origin.value,
        "prefix_decision_complete": prefix.decision_complete,
        "prefix_states_explored": prefix.states_explored,
        "prefix_alphabet_size": prefix.alphabet_size,
        "necessary_evidence_count": len(certificate.necessary_evidence),
        "possible_evidence_count": len(certificate.possible_evidence),
        "certificate_verified": verification.valid,
        "grounding_verified": grounding_report.valid,
    }


def main() -> None:
    os.chdir(UPSTREAM)
    checkpoint = checkpoint_path(EXPERIMENT, TRAINING_SEED, CHECKPOINT_STEPS)
    steering_tasks = load_tasks()[:NUM_STEERING_FORMULAS]

    traces: list[Any] = []
    trace_meta: list[dict[str, Any]] = []
    for task in steering_tasks:
        env, agent = load_agent(EXPERIMENT, TRAINING_SEED, checkpoint, task.source)
        try:
            for episode_index in range(EPISODES_PER_STEERING_FORMULA):
                seed = EVAL_SEED_BASE + episode_index
                trace = run_steering_episode(env, agent, task, seed, f"ppo_seed_{TRAINING_SEED}_operator_probe")
                traces.append(trace)
                trace_meta.append({"steering_task_id": task.task_id, "seed": seed, "steps": len(trace.labels)})
        finally:
            env.close()

    per_probe: dict[str, list[dict[str, Any]]] = {name: [] for name in PROBE_FORMULAS}
    failures: list[dict[str, Any]] = []
    for trace, meta in zip(traces, trace_meta):
        for probe_name, formula_text in PROBE_FORMULAS.items():
            try:
                outcome = audit_probe(formula_text, trace)
                outcome.update(meta)
                per_probe[probe_name].append(outcome)
            except AssertionError as exc:
                failures.append({"probe": probe_name, "meta": meta, "error": str(exc)})

    summary = {}
    for probe_name, rows in per_probe.items():
        satisfied_count = sum(1 for r in rows if r["satisfied"])
        summary[probe_name] = {
            "formula": PROBE_FORMULAS[probe_name],
            "n": len(rows),
            "satisfied_count": satisfied_count,
            "prefix_status_counts": {
                status: sum(1 for r in rows if r["prefix_status"] == status)
                for status in sorted({r["prefix_status"] for r in rows})
            },
            "prefix_decision_origin_counts": {
                origin: sum(1 for r in rows if r["prefix_decision_origin"] == origin)
                for origin in sorted({r["prefix_decision_origin"] for r in rows})
            },
            "incomplete_prefix_decisions": sum(
                1 for r in rows if not r["prefix_decision_complete"]
            ),
        }

    result = {
        "schema_version": "deepltl-point-operator-generality-stress-test/2",
        "purpose": (
            "Exercises the audit engine on G, R, W, X, and Xw formulas -- "
            "operators the confirmatory corpus never uses -- against real "
            "RL-generated PointLtl2 traces, in addition to synthetic "
            "exhaustive-suite traces. This checks implementation coverage, "
            "not empirical generalization, a P_sat claim, or an extension "
            "of the confirmatory corpus."
        ),
        "checkpoint": {
            "experiment": EXPERIMENT,
            "training_seed": TRAINING_SEED,
            "checkpoint_steps": CHECKPOINT_STEPS,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
        "num_real_traces_audited": len(traces),
        "probe_formulas": PROBE_FORMULAS,
        "num_audit_failures": len(failures),
        "failures": failures,
        "summary_by_probe": summary,
        "per_probe_per_trace": per_probe,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_bytes((json.dumps(result, indent=2) + "\n").encode("utf-8"))
    digest = hashlib.sha256(OUTPUT_PATH.read_bytes()).hexdigest()
    (OUTPUT_PATH.parent / f"{OUTPUT_PATH.name}.sha256").write_bytes(
        f"{digest}  {OUTPUT_PATH.name}\n".encode("utf-8")
    )

    print(f"wrote {OUTPUT_PATH} ({digest})")
    print(f"real traces audited: {len(traces)}")
    print(f"audit failures: {len(failures)}")
    for probe_name, row in summary.items():
        print(f"  {probe_name} ({row['formula']}): satisfied {row['satisfied_count']}/{row['n']}, "
              f"prefix status counts={row['prefix_status_counts']}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
