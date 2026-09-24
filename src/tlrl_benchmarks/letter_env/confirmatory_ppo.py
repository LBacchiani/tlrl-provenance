"""DeepLTL LetterEnv-v0 (PPO) adapter for the generic confirmatory campaign runner.

Mirrors ``tlrl_benchmarks.letter_env.confirmatory`` (the DQN confirmatory
adapter) exactly in structure and evaluation design -- same scenario-paired,
official-random-map evaluation distribution, same disposition/failure-class
logic, same fail-closed official/independent comparison. Copy-pasted rather
than shared, per this project's rule that each benchmark/algorithm pillar's
confirmatory session is a self-contained artifact. The only substantive
difference is the policy: this session loads DeepLTL's own formula-
conditioned recurrent PPO actor-critic checkpoint (via ``model.agent.Agent``)
instead of the Q-network, and its ``SequenceSearch`` proxy uses the critic's
own value head instead of a ``max_a Q(s,a)`` estimate -- PPO is exactly what
DeepLTL's Agent/SequenceSearch machinery was built for, so no override is
needed there (unlike the DQN adapter's ``get_value`` override).
"""

from __future__ import annotations

from collections import Counter
import hashlib
import importlib.metadata
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

from tlrl_provenance.campaign import AuditSubject, AuditTask
from .adapter import (
    ENVIRONMENT_ID,
    assess_letter_episode,
    build_letter_trace,
    canonical_world_digest,
    capture_official_step,
    load_tasks,
    preserve_termination_and_truncation,
    validate_world_layout,
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


class LetterEnvPPOConfirmatoryAdapter:
    """Resolve trained PPO subjects and evaluate the frozen official formula set."""

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
                metadata={
                    "formula_sha256": hashlib.sha256(task.source.encode("utf-8")).hexdigest(),
                    "official_canonical_world_index": task.world_index,
                },
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
        corpus_manifest = self.root / "src" / "tlrl_benchmarks" / "letter_env" / "corpus_manifest.json"
        package_names = ("numpy", "torch", "gymnasium")
        return {
            "adapter": "letter_env_ppo_confirmatory/1",
            "environment_id": ENVIRONMENT_ID,
            "upstream_path": str(self.upstream),
            "upstream_commit": commit,
            "official_task_count": len(self._tasks),
            "official_task_corpus_sha256": hashlib.sha256(task_payload.encode("utf-8")).hexdigest(),
            "official_canonical_world_corpus_manifest_sha256": _sha256(corpus_manifest),
            "evaluation_scenario_distribution": (
                "official_letter_env_random_map_reset; the official_canonical_world_corpus "
                "is provenance and a smoke/compatibility check only, not the scenario pool"
            ),
            "label_timing": "post_action_before_ldba_transition",
            "python": sys.version,
            "dependency_versions": {
                name: importlib.metadata.version(name) for name in package_names
            },
            "rabinizer_jar_path": str(rabinizer),
            "rabinizer_jar_sha256": _sha256(rabinizer),
        }

    def open_session(self, subject: AuditSubject, task: AuditTask) -> "LetterEnvPPOConfirmatorySession":
        return LetterEnvPPOConfirmatorySession(self, subject, task)


class LetterEnvPPOConfirmatorySession:
    def __init__(
        self,
        adapter: LetterEnvPPOConfirmatoryAdapter,
        subject: AuditSubject,
        task: AuditTask,
    ):
        self.adapter = adapter
        self.subject = subject
        self.task = task
        self._task = {item.task_id: item for item in load_tasks()}[task.task_id]

        import torch
        from envs import make_env
        from ltl import FixedSampler
        from ltl.automata import LDBASequence
        from model.agent import Agent
        from model.model import build_model
        from preprocessing import VOCAB, reset_vocab
        from sequence.search import SequenceSearch
        import config as deepltl_config

        self.torch = torch
        self.deepltl_config = deepltl_config

        sampler = FixedSampler.partial(task.formula)
        self.env = preserve_termination_and_truncation(make_env(ENVIRONMENT_ID, sampler, render_mode=None))
        self.propositions = set(self.env.get_propositions())

        reset_vocab()
        with Path(str(subject.metadata["vocab_path"])).open("rb") as handle:
            VOCAB.update(pickle.load(handle))
        status = torch.load(str(subject.metadata["checkpoint_path"]), map_location="cpu")
        model = build_model(self.env, status, deepltl_config.model_configs[ENVIRONMENT_ID])
        model.eval()
        search = _RobustFiniteSearch(model, self.propositions, SequenceSearch, LDBASequence)
        self.agent = Agent(model, search=search, propositions=self.propositions, verbose=False)

    def run_episode(
        self,
        *,
        evaluation_seed: int,
        deterministic: bool,
        verification_mode: str,
        scenario_seed: int | None = None,
    ) -> Mapping[str, Any]:
        if not deterministic:
            raise ValueError("the confirmatory PPO audit is defined as a deterministic evaluation")
        map_seed = evaluation_seed if scenario_seed is None else scenario_seed
        random.seed(evaluation_seed)
        np.random.seed(evaluation_seed)
        self.torch.manual_seed(evaluation_seed)
        self.agent.reset()

        # Mirrors LetterEnvDQNConfirmatorySession: LetterEnv.reset() shuffles
        # self.locations in place and never restores it, so the map a given
        # seed produces otherwise depends on this env instance's entire reset
        # history. Restoring canonical construction order before every reset
        # makes the generated map a pure function of map_seed alone.
        unwrapped = self.env.unwrapped
        grid = unwrapped.grid_size
        unwrapped.locations = [
            (row, column) for row in range(grid) for column in range(grid) if (row, column) != (0, 0)
        ]
        obs, info = self.env.reset(seed=map_seed)
        generated_map = validate_world_layout(dict(self.env.unwrapped.map))
        generated_map_digest = canonical_world_digest(generated_map)

        steps = []
        terminated = truncated = official_success = official_violation = False
        total_reward = 0.0
        while not (terminated or truncated):
            action = self.agent.get_action(obs, info, deterministic=deterministic).flatten()
            if action.shape == (1,):
                action = action[0]
            action_int = int(action)
            obs, reward, terminated, truncated, info = self.env.step(np.int64(action_int))
            total_reward += float(reward)
            steps.append(capture_official_step(self.env, info, action=action_int, reward=float(reward)))
            official_success = official_success or bool(info.get("success", False))
            official_violation = official_violation or bool(info.get("violation", False))
            if len(steps) > 75:
                raise RuntimeError("official LetterEnv TimeLimit failed to truncate at 75 steps")

        trace = build_letter_trace(
            steps, policy_id=self.subject.subject_id, seed=evaluation_seed,
            task_id=self.task.task_id, terminated=terminated, truncated=truncated,
        )
        full = verification_mode == "full"
        assessment = assess_letter_episode(
            self._task, trace, official_success=official_success, official_violation=official_violation,
            verify_structural_certificate=full, compute_prefix=True,
        )
        certificate = assessment.certificate
        atom_counts = Counter(atom for labels in trace.labels for atom in labels)
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
            "independent_verdict": certificate.verdict.value,
            "disposition": assessment.disposition.value, "return_sum": total_reward,
            "possible_evidence_count": len(certificate.possible_evidence),
            "necessary_evidence_count": len(certificate.necessary_evidence),
            "atom_counts": dict(sorted(atom_counts.items())), "failure_class": failure_class,
            "official_independent_mismatch": assessment.official_independent_mismatch,
            "scenario_seed": map_seed,
            "start_position": [0, 0],
            "generated_map": [[row, column, letter] for (row, column), letter in generated_map.items()],
            "generated_map_canonical_sha256": generated_map_digest,
        }

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()


# Copy-pasted from tlrl_benchmarks.letter_env.confirmatory / deepltl_point.confirmatory,
# not imported: each benchmark/algorithm session is a self-contained artifact.
class _RobustFiniteSearch:
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
