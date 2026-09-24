"""Fail-closed adapter for the official DeepLTL LetterEnv-v0 benchmark.

LetterEnv is a torus grid: the agent occupies exactly one cell, and each cell
holds at most one letter.  The proposition at any step is therefore a plain
dictionary lookup, not a geometric computation, and the environment already
guarantees the zero-or-one-proposition contract by construction
(``get_possible_assignments`` in the upstream ``letter_env.py`` explicitly
declares ``Assignment.zero_or_one_propositions``).

Each of the 50 official evaluation formulas is paired with a specific,
pre-generated world layout (``worlds/world_info_{index}.pkl``), matched by
list position to the corresponding line of ``tasks.txt``.  This pairing was
confirmed by reading the upstream dataset-generation script
(``create_eval_dataset.py``, which writes a world and its formula in the same
loop iteration under the same index) and the upstream evaluation harness
(``eval_sync_env.py``, which loads both by the same index).

That pairing is official corpus *provenance* and a canonical smoke/
compatibility check (see ``load_world_layout``/``world_path`` below) -- it is
not the confirmatory-audit scenario pool.  With a deterministic (epsilon=0)
evaluation policy, ``LetterEnv.reset()`` always sets ``self.agent = (0, 0)``,
so pinning both the map and the policy would make every audit episode for a
given (subject, formula) pair identical.  The confirmatory campaign
(``tlrl_benchmarks.letter_env.confirmatory``) therefore evaluates each formula
against genuinely random maps drawn from the environment's own official reset
distribution (``use_fixed_map=False``), paired across subjects via a shared
``scenario_seed`` so that cross-policy comparisons for "the same formula,
the same episode index" are not confounded by different subjects drawing
different random maps.

This module is intentionally self-contained (no shared code with the
PointLtl2 adapter): PointLtl2's adapter is a frozen, hash-anchored artifact
and must not be modified or depended upon by other benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import partial
import hashlib
import inspect
import json
import math
from pathlib import Path
import pickle
from typing import Iterable, Mapping, Sequence

from tlrl_provenance import (
    DecisionOrigin,
    Formula,
    EvaluationLimits,
    LabeledTrace,
    PrefixAssessment,
    PrefixStatus,
    PropositionRule,
    SemanticCertificate,
    assess_prefix,
    evaluate_with_provenance,
    ground_raw_steps,
    parse_formula,
    verify_certificate,
    verify_grounding_reconstruction,
)


ENVIRONMENT_ID = "LetterEnv-v0"
ATOMS = frozenset("abcdefghijkl")
GRID_SIZE = 7
EXPECTED_TASK_COUNT = 50
EXPECTED_MAX_STEPS = 75
EXPECTED_LETTER_COPIES = 2
EXPECTED_OCCUPIED_CELLS = len(ATOMS) * EXPECTED_LETTER_COPIES
EXPECTED_UPSTREAM_TASKS_SHA256 = "423eb4bbd5737b6f7dee14f2cef0feab40b78a55c7ad415632b0817c49e7f4f4"
# Same CRLF -> LF freezing convention as the PointLtl2 adapter.
EXPECTED_TASKS_SHA256 = "c91a1e4ec51b3181c08cb68460c47d48f9daf8dcbe242aa22bab26993c0b6268"
RULE_VERSION = "letter-env-grid-cell/1"


@dataclass(frozen=True, slots=True)
class LetterGridStep:
    """Raw grid state sufficient to reconstruct every LetterEnv proposition."""

    agent_pos: tuple[int, int]
    letter_map: Mapping[tuple[int, int], str]
    reported_propositions: frozenset[str] | None = None

    def __post_init__(self) -> None:
        agent_pos = _cell(self.agent_pos, "agent position")
        object.__setattr__(self, "agent_pos", agent_pos)
        letter_map: dict[tuple[int, int], str] = {}
        for raw_cell, letter in self.letter_map.items():
            cell = _cell(raw_cell, "letter cell")
            if not isinstance(letter, str) or letter not in ATOMS:
                raise ValueError(f"unsupported LetterEnv letter: {letter!r}")
            if cell in letter_map:
                raise ValueError(f"duplicate letter assignment at cell {cell}")
            letter_map[cell] = letter
        _require_official_layout(letter_map)
        object.__setattr__(self, "letter_map", letter_map)
        if self.reported_propositions is not None:
            reported = frozenset(self.reported_propositions)
            if any(not isinstance(item, str) for item in reported):
                raise TypeError("reported propositions must be strings")
            unknown = reported - ATOMS
            if unknown:
                raise ValueError(f"unknown reported propositions: {sorted(unknown)}")
            if len(reported) > 1:
                raise ValueError(
                    "LetterEnv guarantees zero-or-one active propositions, "
                    f"found {sorted(reported)}"
                )
            object.__setattr__(self, "reported_propositions", reported)

    def contact(self, letter: str) -> bool:
        return self.letter_map.get(self.agent_pos) == letter

    @property
    def grounded_propositions(self) -> frozenset[str]:
        letter = self.letter_map.get(self.agent_pos)
        return frozenset({letter}) if letter is not None else frozenset()

    def require_reported_agreement(self) -> None:
        if self.reported_propositions is None:
            raise ValueError("official proposition output was not retained")
        grounded = self.grounded_propositions
        if grounded != self.reported_propositions:
            raise AssertionError(
                "official proposition output disagrees with independent grid lookup: "
                f"reported={sorted(self.reported_propositions)}, grounded={sorted(grounded)}"
            )


@dataclass(frozen=True, slots=True)
class LetterStepRecord:
    """One step's grid state with the transition payload retained."""

    geometry: LetterGridStep
    action: object | None = None
    reward: float | None = None

    def __post_init__(self) -> None:
        if self.reward is not None and not math.isfinite(float(self.reward)):
            raise ValueError("reward must be finite")
        if self.reward is not None:
            object.__setattr__(self, "reward", float(self.reward))


@dataclass(frozen=True, slots=True)
class LetterTask:
    task_id: str
    source: str
    formula: Formula
    world_index: int


class EpisodeDisposition(str, Enum):
    DECIDED_SATISFACTION = "decided_satisfaction"
    DECIDED_VIOLATION = "decided_violation"
    CENSORED = "censored"
    ENVIRONMENT_TERMINATION = "environment_termination"
    OPEN_PREFIX = "open_prefix"


@dataclass(frozen=True, slots=True)
class LetterEpisodeAssessment:
    disposition: EpisodeDisposition
    certificate: SemanticCertificate
    prefix: PrefixAssessment
    official_independent_mismatch: bool = False

    @property
    def decided(self) -> bool:
        return self.disposition in {
            EpisodeDisposition.DECIDED_SATISFACTION,
            EpisodeDisposition.DECIDED_VIOLATION,
        }


def _cell(value: Sequence[object], name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must have exactly two coordinates")
    row, col = int(value[0]), int(value[1])
    if float(value[0]) != row or float(value[1]) != col:
        raise ValueError(f"{name} coordinates must be integral")
    if not (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE):
        raise ValueError(f"{name} {(row, col)} is outside the {GRID_SIZE}x{GRID_SIZE} grid")
    return row, col


def _require_official_layout(letter_map: Mapping[tuple[int, int], str]) -> None:
    """Enforce the published ``LetterEnv-v0`` two-copies-per-letter map."""

    if len(letter_map) != EXPECTED_OCCUPIED_CELLS:
        raise ValueError(
            "official LetterEnv-v0 layouts must contain exactly "
            f"{EXPECTED_OCCUPIED_CELLS} occupied cells, found {len(letter_map)}"
        )
    counts = {letter: 0 for letter in ATOMS}
    for letter in letter_map.values():
        if letter not in counts:
            raise ValueError(f"unsupported LetterEnv letter: {letter!r}")
        counts[letter] += 1
    wrong = {
        letter: count
        for letter, count in sorted(counts.items())
        if count != EXPECTED_LETTER_COPIES
    }
    if wrong:
        raise ValueError(
            "official LetterEnv-v0 layouts require exactly "
            f"{EXPECTED_LETTER_COPIES} copies of every letter; found {wrong}"
        )


def validate_world_layout(raw_layout: object) -> dict[tuple[int, int], str]:
    """Validate and canonically order one trusted upstream world payload."""

    if not isinstance(raw_layout, Mapping):
        raise TypeError("LetterEnv world payload must be a mapping")
    normalized: dict[tuple[int, int], str] = {}
    for raw_cell, raw_letter in raw_layout.items():
        if isinstance(raw_cell, (str, bytes)) or not isinstance(raw_cell, Sequence):
            raise TypeError("LetterEnv world cells must be coordinate sequences")
        cell = _cell(raw_cell, "letter cell")
        if cell in normalized:
            raise ValueError(f"duplicate letter assignment at cell {cell}")
        if not isinstance(raw_letter, str) or raw_letter not in ATOMS:
            raise ValueError(f"unsupported LetterEnv letter: {raw_letter!r}")
        normalized[cell] = raw_letter
    _require_official_layout(normalized)
    return dict(sorted(normalized.items()))


def canonical_world_digest(layout: Mapping[tuple[int, int], str]) -> str:
    """Hash layout meaning independently of pickle byte representation."""

    normalized = validate_world_layout(layout)
    payload = json.dumps(
        [[row, column, letter] for (row, column), letter in normalized.items()],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _letter_contact(record: LetterStepRecord, _time: int, *, letter: str) -> bool:
    return record.geometry.contact(letter)


def _rule_digest(letter: str) -> str:
    payload = inspect.getsource(_letter_contact) + f"\nletter={letter}\n{RULE_VERSION}\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def letter_grid_rules() -> tuple[PropositionRule, ...]:
    """Return the frozen, independently replayable grid-lookup proposition rules."""

    return tuple(
        PropositionRule(
            proposition=letter,
            rule_id=f"letter_env.grid_cell.{letter}",
            rule_version=RULE_VERSION,
            evaluator=partial(_letter_contact, letter=letter),
            raw_fields=("geometry.agent_pos", "geometry.letter_map"),
            source_digest=_rule_digest(letter),
        )
        for letter in sorted(ATOMS)
    )


def tasks_path() -> Path:
    return Path(__file__).with_name("tasks.txt")


def worlds_dir(root: Path) -> Path:
    return (root / ".runtime" / "deep-ltl" / "eval_datasets" / "LetterEnv-v0" / "worlds").resolve()


def world_path(root: Path, world_index: int) -> Path:
    if isinstance(world_index, bool) or not isinstance(world_index, int):
        raise TypeError("world index must be an integer")
    if not 0 <= world_index < EXPECTED_TASK_COUNT:
        raise ValueError(
            f"world index must be in [0, {EXPECTED_TASK_COUNT - 1}], found {world_index}"
        )
    return worlds_dir(root) / f"world_info_{world_index}.pkl"


def load_world_layout(root: Path, world_index: int) -> dict[tuple[int, int], str]:
    """Load one pinned upstream pickle and validate its complete semantic payload.

    Pickle is used only because it is the official DeepLTL dataset format. The
    caller must point ``root`` at the pinned, locally verified upstream checkout;
    arbitrary or downloaded untrusted pickle files must not be passed here.
    """

    path = world_path(root, world_index)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return validate_world_layout(payload)


def load_tasks(path: Path | str | None = None) -> tuple[LetterTask, ...]:
    """Load and verify the byte-frozen official 50-task evaluation corpus.

    Task index N is paired with world file ``world_info_N.pkl`` by upstream
    construction (see module docstring) -- this function preserves that
    pairing as ``LetterTask.world_index``.
    """

    source_path = tasks_path() if path is None else Path(path)
    raw_payload = source_path.read_bytes()
    payload = raw_payload.replace(b"\r\n", b"\n")
    if b"\r" in payload:
        raise ValueError("task corpus contains unsupported carriage returns")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_TASKS_SHA256:
        raise ValueError(f"task corpus SHA-256 mismatch: {digest}")
    lines = tuple(line.strip() for line in payload.decode("utf-8").splitlines() if line.strip())
    if len(lines) != EXPECTED_TASK_COUNT:
        raise ValueError(f"expected {EXPECTED_TASK_COUNT} tasks, found {len(lines)}")
    tasks = tuple(
        LetterTask(f"letter_env_{index:03d}", source, parse_formula(source), world_index=index)
        for index, source in enumerate(lines)
    )
    atom_union = frozenset(atom for task in tasks for atom in task.formula.atoms())
    if not atom_union.issubset(ATOMS):
        raise ValueError(f"task atoms exceed frozen environment atoms: {sorted(atom_union - ATOMS)}")
    return tasks


def capture_official_step(
    env: object,
    info: Mapping[str, object],
    *,
    action: object | None = None,
    reward: float | None = None,
) -> LetterStepRecord:
    """Capture one step from DeepLTL's LetterEnv without importing its packages.

    Depends only on the raw ``LetterEnv`` instance's public ``agent``/``map``
    attributes and validates the official ``info['propositions']`` against an
    independent grid lookup.
    """

    unwrapped = getattr(env, "unwrapped", env)
    agent = getattr(unwrapped, "agent", None)
    letter_map = getattr(unwrapped, "map", None)
    if agent is None:
        raise TypeError("environment does not expose unwrapped.agent")
    if letter_map is None:
        raise TypeError("environment does not expose unwrapped.map")
    if "propositions" not in info:
        raise KeyError("official step info is missing 'propositions'")
    reported_raw = info["propositions"]
    if isinstance(reported_raw, str) or not isinstance(reported_raw, Iterable):
        raise TypeError("info['propositions'] must be a non-string iterable")
    geometry = LetterGridStep(
        agent_pos=tuple(agent),
        letter_map=dict(letter_map),
        reported_propositions=frozenset(reported_raw),
    )
    geometry.require_reported_agreement()
    return LetterStepRecord(geometry, action, reward)


def preserve_termination_and_truncation(env: object) -> object:
    """Return DeepLTL's inner TimeLimit wrapper with the five-value step API.

    The upstream ``RemoveTruncWrapper`` returns ``done = terminated or
    truncated`` and discards which condition occurred. Auditable collection
    must bypass only that outer wrapper; guessing from ``done`` after the fact
    is forbidden.  LetterEnv's official horizon is 75 steps (``make_env``'s
    own default for every ``Letter*`` environment name, confirmed against
    ``src/envs/env_utils.py`` and the one existing evaluation driver,
    ``simulate_letter.py``, which passes the same value explicitly).
    """

    if type(env).__name__ != "RemoveTruncWrapper":
        raise TypeError("expected the official outer RemoveTruncWrapper")
    inner = getattr(env, "env", None)
    if inner is None or type(inner).__name__ != "TimeLimit":
        raise TypeError("RemoveTruncWrapper must directly contain Gymnasium TimeLimit")
    maximum = getattr(inner, "_max_episode_steps", None)
    if maximum != EXPECTED_MAX_STEPS:
        raise ValueError(f"official LetterEnv TimeLimit must be {EXPECTED_MAX_STEPS}, found {maximum}")
    return inner


def build_letter_trace(
    steps: Sequence[LetterStepRecord],
    *,
    policy_id: str | None = None,
    seed: int | None = None,
    task_id: str | None = None,
    terminated: bool = True,
    truncated: bool = False,
) -> LabeledTrace:
    """Ground a sequence of step records under the frozen label timing."""

    for step in steps:
        step.geometry.require_reported_agreement()
    return ground_raw_steps(
        steps,
        letter_grid_rules(),
        raw_kind="state",
        policy_id=policy_id,
        environment_id=ENVIRONMENT_ID,
        seed=seed,
        terminated=terminated,
        truncated=truncated,
        metadata={
            "task_id": task_id,
            "label_timing": "post_action_before_ldba_transition",
            "transition_payload": "LetterStepRecord.action/reward",
        },
    )


def assess_letter_episode(
    task: LetterTask,
    trace: LabeledTrace,
    *,
    official_success: bool = False,
    official_violation: bool = False,
    evaluation_limits: EvaluationLimits = EvaluationLimits(),
    verify_structural_certificate: bool = True,
    verification_kwargs: Mapping[str, object] | None = None,
    compute_prefix: bool = True,
) -> LetterEpisodeAssessment:
    """Cross-check official terminal flags without misclassifying truncations.

    When ``compute_prefix`` is false, the returned ``prefix`` is a sentinel
    placeholder used only to preserve the result shape for high-throughput
    completed-episode audits.  It must not be interpreted as a real
    continuation assessment.
    """

    if official_success and official_violation:
        raise ValueError("an episode cannot be both an official success and violation")
    if trace.truncated and (official_success or official_violation):
        raise ValueError("a truncated episode cannot carry a decided official TL outcome")
    if (official_success or official_violation) and not trace.terminated:
        raise ValueError("official TL outcomes require termination")
    if not compute_prefix and not (official_success or official_violation):
        raise ValueError(
            "compute_prefix=False is permitted only for an asserted official terminal outcome"
        )

    rules = letter_grid_rules()
    grounding_report = verify_grounding_reconstruction(trace, rules)
    if not grounding_report.valid:
        raise AssertionError(f"grounding replay failed: {grounding_report.errors}")
    certificate = evaluate_with_provenance(
        task.formula,
        trace,
        require_grounding=True,
        verify_reference="direct",
        limits=evaluation_limits,
    )
    if verify_structural_certificate:
        verification = verify_certificate(certificate, **(verification_kwargs or {}))
        if not verification.valid:
            raise AssertionError(f"certificate verification failed: {verification.errors}")
    prefix = (
        assess_prefix(task.formula, trace)
        if compute_prefix
        else PrefixAssessment(
            PrefixStatus.OPEN,
            certificate.satisfied,
            task.formula,
            decision_origin=DecisionOrigin.NOT_COMPUTED,
        )
    )
    if compute_prefix and not prefix.decision_complete:
        raise AssertionError(
            "continuation-aware status was not decided (origin="
            f"{prefix.decision_origin.value}, states_explored={prefix.states_explored}, "
            f"alphabet_size={prefix.alphabet_size}); refusing to silently treat this as a "
            "proven OPEN"
        )

    if official_success:
        if not certificate.satisfied or (
            compute_prefix and prefix.status is not PrefixStatus.DEFINITELY_SATISFIED
        ):
            raise AssertionError("official success disagrees with independent temporal semantics")
    elif official_violation:
        if certificate.satisfied or (
            compute_prefix and prefix.status is not PrefixStatus.DEFINITELY_VIOLATED
        ):
            raise AssertionError("official violation disagrees with independent temporal semantics")

    if compute_prefix:
        independent_success = prefix.status is PrefixStatus.DEFINITELY_SATISFIED
        independent_violation = prefix.status is PrefixStatus.DEFINITELY_VIOLATED
    else:
        # Fast mode is restricted above to completed outcomes asserted by the
        # official monitor and already checked against the finite-trace result.
        independent_success = official_success
        independent_violation = official_violation

    if independent_success:
        disposition = EpisodeDisposition.DECIDED_SATISFACTION
    elif independent_violation:
        disposition = EpisodeDisposition.DECIDED_VIOLATION
    elif trace.truncated:
        disposition = EpisodeDisposition.CENSORED
    elif trace.terminated:
        disposition = EpisodeDisposition.ENVIRONMENT_TERMINATION
    else:
        disposition = EpisodeDisposition.OPEN_PREFIX

    completed = trace.terminated or trace.truncated
    mismatch = completed and (
        bool(official_success) != independent_success
        or bool(official_violation) != independent_violation
    )
    return LetterEpisodeAssessment(disposition, certificate, prefix, mismatch)
