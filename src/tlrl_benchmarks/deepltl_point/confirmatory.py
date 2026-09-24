"""DeepLTL PointWorld adapter for the generic confirmatory campaign runner.

Evaluates each of the 50 published formulas against a `scenario_seed`, shared
across all five subjects for a given (formula, episode index) via
`tlrl_provenance.campaign`'s opt-in `scenario_namespace`. This makes
cross-policy comparisons for a fixed formula a controlled comparison: every
policy sees the identical scenario, not just the identical formula.
`evaluation_seed` remains each record's unique identity.

This matters because PointLtl2's zone geometry is randomized per reset.
Without scenario pairing, "the same formula, the same episode index" across
different policies would still mean different random zone layouts per
policy, confounding any cross-policy comparison with scenario difficulty.
Verified empirically (not assumed): PointLtl2's zone geometry is a pure
function of the reset seed, with no cross-reset state leakage, so pairing
via a shared seed is sufficient with no additional per-episode state
restoration.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

from tlrl_provenance import EvaluationLimits
from tlrl_provenance.campaign import AuditSubject, AuditTask
from .adapter import (
    ENVIRONMENT_ID,
    assess_point_episode,
    build_point_trace,
    capture_official_step,
    load_tasks,
    preserve_termination_and_truncation,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _formula_family(formula: str) -> str:
    source = formula.strip()
    if source.startswith("F "):
        return "eventually_chain"
    if " U " in source:
        return "until_chain"
    return "other"


def _retained_trace_payload(trace: Any) -> tuple[dict[str, Any], str]:
    """Compact, replayable semantic input retained with every audit record.

    Zone geometry is static within an episode, so store it once; agent
    positions, official labels, actions, and rewards remain per step. This is
    sufficient to reconstruct grounding and temporal semantics later without
    relying on MuJoCo trajectory reproducibility.
    """

    if not trace.states:
        raise ValueError("cannot retain an empty PointWorld trace")
    first_geometry = trace.states[0].geometry
    zone_layout = [
        {
            "color": zone.color,
            "centers_xy": [list(center) for center in zone.centers_xy],
            "radius": zone.radius,
        }
        for zone in first_geometry.zones
    ]
    for state in trace.states[1:]:
        if state.geometry.zones != first_geometry.zones:
            raise AssertionError("PointWorld zone geometry changed within one episode")
    payload = {
        "zone_layout": zone_layout,
        "agent_xy": [list(state.geometry.agent_xy) for state in trace.states],
        "reported_labels": [sorted(labels) for labels in trace.labels],
        "actions": [state.action for state in trace.states],
        "rewards": [state.reward for state in trace.states],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return payload, hashlib.sha256(encoded).hexdigest()


def compare_official_verdict(
    *,
    official_success: bool,
    official_violation: bool,
    independent_verdict: str,
    disposition: str,
) -> tuple[str, bool]:
    """Compare DeepLTL's *partial* terminal verdict with the v3 assessment.

    DeepLTL exposes two asserted terminal flags, not a total Boolean verdict.
    When both are false, the official monitor is undecided: this includes
    right-censored timeouts and ordinary environment terminations.  A finite
    trace certificate is still Boolean in those cases, but comparing that
    Boolean value to an absent official decision would manufacture a mismatch.
    """

    if official_success and official_violation:
        raise ValueError("official success and violation cannot both be asserted")
    if independent_verdict not in {"satisfied", "violated"}:
        raise ValueError(f"unsupported independent verdict: {independent_verdict!r}")
    decided_dispositions = {"decided_satisfaction", "decided_violation"}
    if official_success:
        return (
            "satisfied",
            independent_verdict != "satisfied" or disposition != "decided_satisfaction",
        )
    if official_violation:
        return (
            "violated",
            independent_verdict != "violated" or disposition != "decided_violation",
        )
    return "undecided", disposition in decided_dispositions


class DeepLTLPointConfirmatoryAdapter:
    """Resolve trained PPO subjects and evaluate the frozen official task set,
    with cross-policy scenario pairing (see module docstring).
    """

    def __init__(self, *, config: dict[str, Any], root: Path):
        self.config = config
        self.root = root.resolve()
        self.upstream = (self.root / config.get("upstream", ".runtime/deep-ltl")).resolve()
        self.upstream_src = self.upstream / "src"
        if not self.upstream_src.is_dir():
            raise FileNotFoundError(self.upstream_src)
        if str(self.upstream_src) not in sys.path:
            sys.path.insert(0, str(self.upstream_src))
        rabinizer = (self.root / self.config.get("rabinizer_jar", "rabinizer-4/lib/rabinizer.jar")).resolve()
        if not rabinizer.is_file():
            raise FileNotFoundError(f"RABINIZER_JAR not found: {rabinizer}")
        os.environ["RABINIZER_JAR"] = str(rabinizer)
        self._subjects = self._resolve_subjects()
        self._tasks = tuple(
            AuditTask(
                task_id=task.task_id,
                formula=task.source,
                family=_formula_family(task.source),
                metadata={"formula_sha256": hashlib.sha256(task.source.encode("utf-8")).hexdigest()},
            )
            for task in load_tasks()
        )

    def _resolve_subjects(self) -> tuple[AuditSubject, ...]:
        result = []
        for item in self.config["subjects"]:
            seed = int(item["training_seed"])
            experiment = str(item["experiment"])
            checkpoint_steps = int(item["checkpoint_steps"])
            run = self.upstream / "experiments" / "ppo" / ENVIRONMENT_ID / experiment / str(seed)
            checkpoint = run / "eval" / f"{checkpoint_steps}.pth"
            vocab = run / "vocab.pkl"
            if not checkpoint.is_file() or not vocab.is_file():
                raise FileNotFoundError(checkpoint if not checkpoint.is_file() else vocab)
            subject_id = str(item.get("subject_id", f"ppo_seed_{seed}"))
            result.append(
                AuditSubject(
                    subject_id=subject_id,
                    metadata={
                        "algorithm": "formula_conditioned_recurrent_ppo",
                        "training_seed": seed,
                        "experiment": experiment,
                        "checkpoint_steps": checkpoint_steps,
                        "checkpoint_path": str(checkpoint),
                        "checkpoint_sha256": _sha256(checkpoint),
                        "vocab_path": str(vocab),
                        "vocab_sha256": _sha256(vocab),
                    },
                )
            )
        return tuple(result)

    def subjects(self) -> tuple[AuditSubject, ...]:
        return self._subjects

    def tasks(self) -> tuple[AuditTask, ...]:
        return self._tasks

    def provenance(self) -> Mapping[str, Any]:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=self.upstream, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        task_payload = "\n".join(task.formula for task in self._tasks) + "\n"
        rabinizer = (self.root / self.config.get("rabinizer_jar", "rabinizer-4/lib/rabinizer.jar")).resolve()
        package_names = ("numpy", "torch", "gymnasium", "mujoco", "scipy")
        return {
            "adapter": "deepltl_point_confirmatory/2",
            "environment_id": ENVIRONMENT_ID,
            "upstream_path": str(self.upstream),
            "upstream_commit": commit,
            "official_task_count": len(self._tasks),
            "official_task_corpus_sha256": hashlib.sha256(task_payload.encode("utf-8")).hexdigest(),
            "label_timing": "post_action_before_ldba_transition",
            "python": sys.version,
            "dependency_versions": {
                name: importlib.metadata.version(name) for name in package_names
            },
            "rabinizer_jar_path": str(rabinizer),
            "rabinizer_jar_sha256": _sha256(rabinizer),
            "scenario_pairing": "cross_policy_comparisons_for_a_fixed_formula_use_a_shared_scenario_seed",
            "semantic_adapter_source_sha256": _sha256(Path(__file__).with_name("adapter.py")),
            "retained_trace_schema": "pointworld_compact_grounding_replay/1",
        }

    def open_session(self, subject: AuditSubject, task: AuditTask) -> "DeepLTLPointSession":
        return DeepLTLPointSession(self, subject, task)


class DeepLTLPointSession:
    def __init__(self, adapter: DeepLTLPointConfirmatoryAdapter, subject: AuditSubject, task: AuditTask):
        self.adapter = adapter
        self.subject = subject
        self.task = task
        self._task = {item.task_id: item for item in load_tasks()}[task.task_id]

        import torch
        from config import model_configs
        from envs import make_env
        from ltl import FixedSampler
        from ltl.automata import LDBASequence
        from model.agent import Agent
        from model.model import build_model
        from preprocessing import VOCAB, reset_vocab
        from sequence.search import SequenceSearch

        self.torch = torch
        self.LDBASequence = LDBASequence
        sampler = FixedSampler.partial(task.formula)
        self.env = preserve_termination_and_truncation(make_env(ENVIRONMENT_ID, sampler, render_mode=None))
        reset_vocab()
        with Path(str(subject.metadata["vocab_path"])).open("rb") as handle:
            VOCAB.update(pickle.load(handle))
        status = torch.load(str(subject.metadata["checkpoint_path"]), map_location="cpu")
        model = build_model(self.env, status, model_configs[ENVIRONMENT_ID])
        model.eval()
        props = set(self.env.get_propositions())
        search = _RobustFiniteSearch(model, props, SequenceSearch, LDBASequence)
        self.agent = Agent(model, search=search, propositions=props, verbose=False)

    def run_episode(
        self,
        *,
        evaluation_seed: int,
        deterministic: bool,
        verification_mode: str,
        scenario_seed: int | None = None,
    ) -> Mapping[str, Any]:
        # scenario_seed (shared across all subjects for a given formula and
        # episode index) determines the zone layout, so cross-policy
        # comparisons for a fixed formula are not confounded by different
        # subjects drawing different random scenarios. evaluation_seed
        # remains the record's unique identity/RNG seed. Falling back to it
        # when scenario_seed is absent keeps this session usable outside a
        # scenario-paired campaign (e.g. standalone smoke runs).
        map_seed = evaluation_seed if scenario_seed is None else scenario_seed
        random.seed(evaluation_seed)
        np.random.seed(evaluation_seed)
        self.torch.manual_seed(evaluation_seed)
        obs, info = self.env.reset(seed=map_seed)
        self.agent.reset()
        steps = []
        terminated = truncated = official_success = official_violation = False
        total_reward = 0.0
        while not (terminated or truncated):
            action = self.agent.get_action(obs, info, deterministic=deterministic).flatten()
            if action.shape == (1,):
                action = action[0]
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            action_value = action.tolist() if hasattr(action, "tolist") else action
            steps.append(capture_official_step(self.env, info, action=action_value, reward=float(reward)))
            official_success = official_success or bool(info.get("success", False))
            official_violation = official_violation or bool(info.get("violation", False))
            if len(steps) > 1000:
                raise RuntimeError("official PointLtl2 TimeLimit failed to truncate at 1000 steps")

        trace = build_point_trace(
            steps, policy_id=self.subject.subject_id, seed=evaluation_seed,
            task_id=self.task.task_id, terminated=terminated, truncated=truncated,
        )
        full = verification_mode == "full"
        # Official success/violation is already a completed monitor verdict
        # and is cross-checked against the independent finite-trace evaluator.
        # When neither flag is asserted, however, a real prefix assessment is
        # required to distinguish an open obligation from a decisive outcome
        # missed by the official monitor.  Never manufacture censoring merely
        # from the environment's truncated/terminated bit.
        compute_prefix = full or not (official_success or official_violation)
        assessment = assess_point_episode(
            self._task, trace, official_success=official_success, official_violation=official_violation,
            evaluation_limits=EvaluationLimits(collect_decisions=full),
            verify_structural_certificate=full, compute_prefix=compute_prefix,
        )
        certificate = assessment.certificate
        retained_trace, retained_trace_sha256 = _retained_trace_payload(trace)
        atom_counts = Counter(atom for labels in trace.labels for atom in labels)
        official_verdict, mismatch = compare_official_verdict(
            official_success=bool(official_success),
            official_violation=bool(official_violation),
            independent_verdict=certificate.verdict.value,
            disposition=assessment.disposition.value,
        )
        if certificate.verdict.value == "satisfied":
            failure_class = "decided_satisfaction"
        elif assessment.disposition.value == "decided_violation":
            failure_class = "forbidden_or_temporal_decided_violation"
        elif assessment.disposition.value == "censored":
            failure_class = "censored"
        elif assessment.disposition.value == "environment_termination" and atom_counts:
            failure_class = "termination_after_relevant_progress"
        elif assessment.disposition.value == "environment_termination":
            failure_class = "termination_without_relevant_progress"
        else:
            failure_class = "other_unresolved"
        return {
            "steps": len(steps), "terminated": bool(terminated), "truncated": bool(truncated),
            "official_success": bool(official_success), "official_violation": bool(official_violation),
            "official_verdict": official_verdict,
            "independent_verdict": certificate.verdict.value,
            "prefix_status": assessment.prefix.status.value,
            "disposition": assessment.disposition.value, "return_sum": total_reward,
            "possible_evidence_count": len(certificate.possible_evidence),
            "necessary_evidence_count": len(certificate.necessary_evidence),
            "atom_counts": dict(sorted(atom_counts.items())), "failure_class": failure_class,
            "official_independent_mismatch": mismatch,
            "scenario_seed": map_seed,
            "retained_trace": retained_trace,
            "retained_trace_sha256": retained_trace_sha256,
        }

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()


class _RobustFiniteSearch:
    """Small proxy implementing DeepLTL's search API without importing it at plan time."""

    def __init__(self, model: Any, propositions: set[str], base: type, sequence_type: type):
        self._delegate = _SearchImplementation(model, propositions, base, sequence_type)

    def __call__(self, ldba: Any, ldba_state: int, obs: dict[str, Any]) -> Any:
        return self._delegate(ldba, ldba_state, obs)


def _SearchImplementation(model: Any, propositions: set[str], base: type, sequence_type: type) -> Any:
    class Search(base):
        def __call__(self, ldba: Any, ldba_state: int, obs: dict[str, Any]) -> Any:
            candidates = self._finite_sequences(ldba, ldba_state)
            if not candidates:
                raise RuntimeError(f"no accepting LDBA sequence from state {ldba_state}")
            return max(candidates, key=lambda sequence: self.get_value(sequence, obs))

        def _finite_sequences(self, ldba: Any, state: int) -> list[Any]:
            scc = ldba.state_to_scc[state]
            if scc.accepting:
                for transition in ldba.state_to_transitions[state]:
                    if transition.target == state and transition.accepting:
                        reach = sequence_type.EPSILON if transition.is_epsilon() else frozenset(transition.valid_assignments)
                        return [sequence_type(((reach, frozenset()),))]
            queue = [(state, (), frozenset())]
            shortest = None
            results = []
            while queue:
                current, sequence, visited = queue.pop(0)
                if shortest is not None and len(sequence) >= shortest:
                    continue
                avoid_sets = [
                    transition.valid_assignments
                    for transition in ldba.state_to_transitions[current]
                    if transition.source != transition.target
                    and ((ldba.state_to_scc[transition.target].bottom and not ldba.state_to_scc[transition.target].accepting)
                         or transition.target in visited)
                ]
                avoid = frozenset().union(*(frozenset(items) for items in avoid_sets)) if avoid_sets else frozenset()
                for transition in ldba.state_to_transitions[current]:
                    if transition.target in visited:
                        continue
                    target_scc = ldba.state_to_scc[transition.target]
                    if target_scc.bottom and not target_scc.accepting:
                        continue
                    reach = sequence_type.EPSILON if transition.is_epsilon() else frozenset(transition.valid_assignments)
                    next_sequence = sequence + ((reach, avoid),)
                    if transition.accepting or target_scc.accepting:
                        shortest = len(next_sequence)
                        results.append(sequence_type(next_sequence))
                    else:
                        queue.append((transition.target, next_sequence, visited | {current}))
            return results

    return Search(model, propositions)
