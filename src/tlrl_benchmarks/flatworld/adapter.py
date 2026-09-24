"""Fail-closed adapter for the official DeepLTL FlatWorld-v0 benchmark.

FlatWorld is a continuous 2D navigation task: the agent occupies a point in
``[-2, 2]^2`` and nine fixed, colored circles ("zones") are scattered across
that square. Unlike PointLtl2's zones, FlatWorld's circles are not mutually
exclusive by color -- several circles overlap by upstream construction (for
example the ``blue``, ``green``, and ``aqua`` circles all cover a common
region), so more than one proposition can be simultaneously true. Unlike
LetterEnv's per-episode randomized letter map, FlatWorld's nine circles are a
fixed class attribute (``FlatWorld.CIRCLES``) baked into the upstream source:
only the agent's starting position is randomized per episode, by the reset
seed alone. There is therefore no analogue of LetterEnv's paired
``world_info_N.pkl`` files or PointLtl2's disjoint-zone invariant -- this
adapter freezes the circle geometry itself as a manifest-hashed constant and
validates every recorded position against it.

The official environment registers ``FlatWorld-v0`` with
``continuous_actions=False`` (see ``envs/flatworld/__init__.py``), giving a
native ``Discrete(9)`` action space (eight compass directions plus "stay").
This differs from PointLtl2's continuous-only ``Box`` action space.

This module is intentionally self-contained (no shared code with the
PointLtl2 or LetterEnv adapters): each benchmark adapter is an independent,
hash-anchored artifact and must not be modified by or depended upon by
another benchmark's code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import partial
import hashlib
import inspect
import math
from pathlib import Path
from typing import Mapping, Sequence

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


ENVIRONMENT_ID = "FlatWorld-v0"
ATOMS = frozenset({"aqua", "blue", "green", "magenta", "orange", "red", "yellow"})
EXPECTED_TASK_COUNT = 49
EXPECTED_MAX_STEPS = 500
EXPECTED_UPSTREAM_TASKS_SHA256 = "1934a774097b917687660d1358694be4ce52b1d22fc4aa87e844ea0da4a00c2f"
# Same CRLF -> LF freezing convention as the PointLtl2 and LetterEnv adapters.
EXPECTED_TASKS_SHA256 = "a1756f3b8318307258bc3a7922d100116cd9534301eabb10f1c7455703af65e0"
RULE_VERSION = "flatworld-circle-contact/1"
# The domain FlatWorld.reset() samples from and clips step displacement into;
# a recorded position is invalid corpus provenance if it ever leaves this
# generously-bounded square (the environment terminates once |pos| > 2 in
# either coordinate, so a well-formed trace should never exceed this bound
# by more than one step's displacement).
POSITION_BOUND = 2.5

# Frozen, independently-replayable copy of upstream FlatWorld.CIRCLES
# (envs/flatworld/flatworld.py). Nine circles, exact literal center/radius
# values transcribed from the upstream class attribute. Two colors (red,
# green) have two circles each; different-color circles are allowed to
# overlap by upstream design (unlike PointLtl2's disjoint-zone invariant).
# Drift between this tuple and the live upstream source is caught by the
# corpus manifest's source-file hash, not re-derived at runtime, so that this
# adapter never has to import the environment module to ground a trace.
CIRCLES: tuple[tuple[str, float, float, float], ...] = (
    ("red", -1.4, 0.55, 0.4),
    ("magenta", -1.1, 1.1, 0.5),
    ("yellow", -1.0, -1.2, 0.3),
    ("orange", -1.53, -0.5, 0.32),
    ("blue", 0.1, 0.0, 0.8),
    ("red", 0.5, -1.3, 0.35),
    ("green", 0.7, 0.7, 0.5),
    ("green", 1.5, -0.75, 0.4),
    ("aqua", 0.8, 0.2, 0.3),
)

_CIRCLE_COLORS = frozenset(color for color, _, _, _ in CIRCLES)
if _CIRCLE_COLORS != ATOMS:
    raise AssertionError(
        f"frozen FlatWorld circle colors {sorted(_CIRCLE_COLORS)} disagree with ATOMS {sorted(ATOMS)}"
    )


@dataclass(frozen=True, slots=True)
class FlatworldPositionStep:
    """One recorded agent position, validated against the frozen domain."""

    position: tuple[float, float]

    def __post_init__(self) -> None:
        position = _xy(self.position, "agent position")
        object.__setattr__(self, "position", position)

    def contact(self, color: str) -> bool:
        if color not in ATOMS:
            raise ValueError(f"unsupported FlatWorld color: {color!r}")
        x, y = self.position
        return any(
            circle_color == color and (x - cx) ** 2 + (y - cy) ** 2 < radius * radius
            for circle_color, cx, cy, radius in CIRCLES
        )

    @property
    def grounded_propositions(self) -> frozenset[str]:
        x, y = self.position
        return frozenset(
            color
            for color, cx, cy, radius in CIRCLES
            if (x - cx) ** 2 + (y - cy) ** 2 < radius * radius
        )


@dataclass(frozen=True, slots=True)
class FlatworldStepRecord:
    """One step's position with the transition payload retained."""

    geometry: FlatworldPositionStep
    action: object | None = None
    reward: float | None = None
    reported_propositions: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if self.reward is not None and not math.isfinite(float(self.reward)):
            raise ValueError("reward must be finite")
        if self.reward is not None:
            object.__setattr__(self, "reward", float(self.reward))
        if self.reported_propositions is not None:
            reported = frozenset(self.reported_propositions)
            unknown = reported - ATOMS
            if unknown:
                raise ValueError(f"unknown reported propositions: {sorted(unknown)}")
            object.__setattr__(self, "reported_propositions", reported)

    def require_reported_agreement(self) -> None:
        if self.reported_propositions is None:
            raise ValueError("official proposition output was not retained")
        grounded = self.geometry.grounded_propositions
        if grounded != self.reported_propositions:
            raise AssertionError(
                "official proposition output disagrees with independent circle-containment "
                f"check: reported={sorted(self.reported_propositions)}, grounded={sorted(grounded)}"
            )


@dataclass(frozen=True, slots=True)
class FlatworldTask:
    task_id: str
    source: str
    formula: Formula


class EpisodeDisposition(str, Enum):
    DECIDED_SATISFACTION = "decided_satisfaction"
    DECIDED_VIOLATION = "decided_violation"
    CENSORED = "censored"
    ENVIRONMENT_TERMINATION = "environment_termination"
    OPEN_PREFIX = "open_prefix"


@dataclass(frozen=True, slots=True)
class FlatworldEpisodeAssessment:
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


def _xy(value: Sequence[object], name: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must have exactly two coordinates")
    x, y = float(value[0]), float(value[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError(f"{name} must be finite")
    if abs(x) > POSITION_BOUND or abs(y) > POSITION_BOUND:
        raise ValueError(f"{name} {(x, y)} exceeds the frozen FlatWorld domain bound {POSITION_BOUND}")
    return x, y


def tasks_path() -> Path:
    return Path(__file__).with_name("tasks.txt")


def load_tasks(path: Path | str | None = None) -> tuple[FlatworldTask, ...]:
    """Load and verify the byte-frozen official 49-task FlatWorld-v0 corpus."""

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
        FlatworldTask(f"flatworld_{index:03d}", source, parse_formula(source))
        for index, source in enumerate(lines)
    )
    atom_union = frozenset(atom for task in tasks for atom in task.formula.atoms())
    if not atom_union.issubset(ATOMS):
        raise ValueError(f"task atoms exceed frozen environment atoms: {sorted(atom_union - ATOMS)}")
    return tasks


def _circle_contact(record: FlatworldStepRecord, _time: int, *, color: str) -> bool:
    return record.geometry.contact(color)


def _rule_digest(color: str) -> str:
    payload = inspect.getsource(_circle_contact) + f"\ncolor={color}\n{RULE_VERSION}\n{CIRCLES}\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def flatworld_zone_rules() -> tuple[PropositionRule, ...]:
    """Return the frozen, independently replayable circle-containment rules."""

    return tuple(
        PropositionRule(
            proposition=color,
            rule_id=f"flatworld.circle_contact.{color}",
            rule_version=RULE_VERSION,
            evaluator=partial(_circle_contact, color=color),
            raw_fields=("geometry.position",),
            source_digest=_rule_digest(color),
        )
        for color in sorted(ATOMS)
    )


def capture_official_step(
    env: object,
    info: Mapping[str, object],
    *,
    action: object | None = None,
    reward: float | None = None,
) -> FlatworldStepRecord:
    """Capture one step from DeepLTL's FlatWorld without importing its packages.

    Depends only on the raw ``FlatWorld`` instance's public ``agent_pos``
    attribute and validates the official ``info['propositions']`` against an
    independent circle-containment lookup.
    """

    unwrapped = getattr(env, "unwrapped", env)
    agent_pos = getattr(unwrapped, "agent_pos", None)
    if agent_pos is None:
        raise TypeError("environment does not expose unwrapped.agent_pos")
    if "propositions" not in info:
        raise KeyError("official step info is missing 'propositions'")
    reported_raw = info["propositions"]
    if isinstance(reported_raw, str) or not isinstance(reported_raw, (set, frozenset, list, tuple)):
        raise TypeError("info['propositions'] must be a non-string set/sequence")
    geometry = FlatworldPositionStep(position=(float(agent_pos[0]), float(agent_pos[1])))
    record = FlatworldStepRecord(
        geometry=geometry,
        action=action,
        reward=reward,
        reported_propositions=frozenset(reported_raw),
    )
    record.require_reported_agreement()
    return record


def preserve_termination_and_truncation(env: object) -> object:
    """Return DeepLTL's inner TimeLimit wrapper with the five-value step API.

    The upstream ``RemoveTruncWrapper`` returns ``done = terminated or
    truncated`` and discards which condition occurred. Auditable collection
    must bypass only that outer wrapper; guessing from ``done`` after the fact
    is forbidden. FlatWorld's official horizon is 500 steps (``make_env``'s
    own default for every ``FlatWorld*`` environment name, confirmed against
    ``src/envs/env_utils.py``).
    """

    if type(env).__name__ != "RemoveTruncWrapper":
        raise TypeError("expected the official outer RemoveTruncWrapper")
    inner = getattr(env, "env", None)
    if inner is None or type(inner).__name__ != "TimeLimit":
        raise TypeError("RemoveTruncWrapper must directly contain Gymnasium TimeLimit")
    maximum = getattr(inner, "_max_episode_steps", None)
    if maximum != EXPECTED_MAX_STEPS:
        raise ValueError(f"official FlatWorld TimeLimit must be {EXPECTED_MAX_STEPS}, found {maximum}")
    return inner


def build_flatworld_trace(
    steps: Sequence[FlatworldStepRecord],
    *,
    policy_id: str | None = None,
    seed: int | None = None,
    task_id: str | None = None,
    terminated: bool = True,
    truncated: bool = False,
) -> LabeledTrace:
    """Ground a sequence of step records under the frozen label timing."""

    for step in steps:
        step.require_reported_agreement()
    return ground_raw_steps(
        steps,
        flatworld_zone_rules(),
        raw_kind="state",
        policy_id=policy_id,
        environment_id=ENVIRONMENT_ID,
        seed=seed,
        terminated=terminated,
        truncated=truncated,
        metadata={
            "task_id": task_id,
            "label_timing": "post_action_before_ldba_transition",
            "transition_payload": "FlatworldStepRecord.action/reward",
        },
    )


def assess_flatworld_episode(
    task: FlatworldTask,
    trace: LabeledTrace,
    *,
    official_success: bool = False,
    official_violation: bool = False,
    evaluation_limits: EvaluationLimits = EvaluationLimits(),
    verify_structural_certificate: bool = True,
    verification_kwargs: Mapping[str, object] | None = None,
    compute_prefix: bool = True,
) -> FlatworldEpisodeAssessment:
    """Cross-check official terminal flags without misclassifying truncations.

    When ``compute_prefix`` is false, the returned ``prefix`` is a sentinel
    placeholder used only to preserve the result shape for high-throughput
    completed-episode audits. It must not be interpreted as a real
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

    rules = flatworld_zone_rules()
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
    return FlatworldEpisodeAssessment(disposition, certificate, prefix, mismatch)
