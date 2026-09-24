"""Fail-closed adapter for the official DeepLTL PointLtl2-v0 benchmark.

The official LDBA is advanced using propositions emitted *after* each physical
action.  Accordingly, each semantic trace position below is one post-action
state.  Actions and rewards are retained inside ``PointStepRecord`` instead of
being shifted into ``LabeledTrace.actions``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import partial
import hashlib
import inspect
import math
from pathlib import Path
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


ENVIRONMENT_ID = "PointLtl2-v0"
ATOMS = frozenset({"blue", "green", "magenta", "yellow"})
ZONE_RADIUS = 0.4
ZONES_PER_COLOR = 2
EXPECTED_TASK_COUNT = 50
EXPECTED_UPSTREAM_TASKS_SHA256 = "0eb487d76bbbcae2e6698abefa1f6f6e03b0cca68d02d5cebdd3ba652af902bb"
# Git may normalize the upstream CRLF file to LF.  The semantic payload is
# frozen after canonical CRLF -> LF conversion, while the original byte hash
# remains recorded separately above.
EXPECTED_TASKS_SHA256 = "20c87b5ba134443e91e06db892a54ff330551e480d04643f3a3eeb3d311144ea"
RULE_VERSION = "deepltl-point-zone-contact/1"


@dataclass(frozen=True, slots=True)
class ZoneGeometry:
    """All circular zones for one proposition."""

    color: str
    centers_xy: tuple[tuple[float, float], ...]
    radius: float = ZONE_RADIUS

    def __post_init__(self) -> None:
        if self.color not in ATOMS:
            raise ValueError(f"unsupported PointLtl2 zone color: {self.color!r}")
        if not math.isfinite(float(self.radius)) or float(self.radius) <= 0:
            raise ValueError("zone radius must be finite and positive")
        centers = tuple(sorted(_xy(center, "zone center") for center in self.centers_xy))
        if len(centers) != ZONES_PER_COLOR:
            raise ValueError(
                f"{self.color} must have exactly {ZONES_PER_COLOR} zones, found {len(centers)}"
            )
        object.__setattr__(self, "centers_xy", centers)
        object.__setattr__(self, "radius", float(self.radius))


@dataclass(frozen=True, slots=True)
class PointZoneStep:
    """Raw geometry sufficient to reconstruct every PointLtl2 proposition."""

    agent_xy: tuple[float, float]
    zones: tuple[ZoneGeometry, ...]
    reported_propositions: frozenset[str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_xy", _xy(self.agent_xy, "agent position"))
        zones = tuple(sorted(self.zones, key=lambda item: item.color))
        colors = [item.color for item in zones]
        if len(set(colors)) != len(colors):
            raise ValueError("zone colors must be unique")
        if frozenset(colors) != ATOMS:
            raise ValueError(f"zone colors must be exactly {sorted(ATOMS)}, found {colors}")
        for zone in zones:
            if not math.isclose(zone.radius, ZONE_RADIUS, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"official PointLtl2 zone radius must be {ZONE_RADIUS}, found {zone.radius}"
                )
        object.__setattr__(self, "zones", zones)
        for left_index, left in enumerate(zones):
            for right in zones[left_index + 1 :]:
                for lx, ly in left.centers_xy:
                    for rx, ry in right.centers_xy:
                        if (lx - rx) ** 2 + (ly - ry) ** 2 <= (left.radius + right.radius) ** 2:
                            raise ValueError(
                                "different-color zones overlap or touch, violating DeepLTL's "
                                "zero-or-one-proposition assignment contract"
                            )
        if self.reported_propositions is not None:
            reported = frozenset(self.reported_propositions)
            unknown = reported - ATOMS
            if unknown:
                raise ValueError(f"unknown reported propositions: {sorted(unknown)}")
            object.__setattr__(self, "reported_propositions", reported)

    def contact(self, color: str) -> bool:
        zone = next((item for item in self.zones if item.color == color), None)
        if zone is None:
            raise KeyError(color)
        x, y = self.agent_xy
        radius_sq = zone.radius * zone.radius
        return any((x - zx) ** 2 + (y - zy) ** 2 <= radius_sq for zx, zy in zone.centers_xy)

    @property
    def grounded_propositions(self) -> frozenset[str]:
        return frozenset(color for color in ATOMS if self.contact(color))

    def require_reported_agreement(self) -> None:
        if self.reported_propositions is None:
            raise ValueError("official proposition output was not retained")
        grounded = self.grounded_propositions
        if grounded != self.reported_propositions:
            raise AssertionError(
                "official proposition output disagrees with independent geometry: "
                f"reported={sorted(self.reported_propositions)}, grounded={sorted(grounded)}"
            )


@dataclass(frozen=True, slots=True)
class PointStepRecord:
    """One post-action semantic position with the transition payload retained."""

    geometry: PointZoneStep
    action: object | None = None
    reward: float | None = None

    def __post_init__(self) -> None:
        if self.reward is not None and not math.isfinite(float(self.reward)):
            raise ValueError("reward must be finite")
        if self.reward is not None:
            object.__setattr__(self, "reward", float(self.reward))


@dataclass(frozen=True, slots=True)
class PointTask:
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
class PointEpisodeAssessment:
    disposition: EpisodeDisposition
    certificate: SemanticCertificate
    prefix: PrefixAssessment

    @property
    def decided(self) -> bool:
        return self.disposition in {
            EpisodeDisposition.DECIDED_SATISFACTION,
            EpisodeDisposition.DECIDED_VIOLATION,
        }


def _xy(value: Sequence[object], name: str) -> tuple[float, float]:
    if len(value) < 2:
        raise ValueError(f"{name} requires at least two coordinates")
    result = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} coordinates must be finite")
    return result


def _zone_contact(record: PointStepRecord, _time: int, *, color: str) -> bool:
    return record.geometry.contact(color)


def _rule_digest(color: str) -> str:
    payload = inspect.getsource(_zone_contact) + f"\ncolor={color}\n{RULE_VERSION}\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def point_zone_rules() -> tuple[PropositionRule, ...]:
    """Return the frozen, independently replayable geometric proposition rules."""

    return tuple(
        PropositionRule(
            proposition=color,
            rule_id=f"deepltl_point.zone_contact.{color}",
            rule_version=RULE_VERSION,
            evaluator=partial(_zone_contact, color=color),
            raw_fields=(
                "geometry.agent_xy",
                f"geometry.zones[{color}].centers_xy",
                f"geometry.zones[{color}].radius",
            ),
            source_digest=_rule_digest(color),
        )
        for color in sorted(ATOMS)
    )


def tasks_path() -> Path:
    return Path(__file__).with_name("tasks.txt")


def load_tasks(path: Path | str | None = None) -> tuple[PointTask, ...]:
    """Load and verify the byte-frozen official 50-task evaluation corpus."""

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
        PointTask(f"deepltl_point_{index:03d}", source, parse_formula(source))
        for index, source in enumerate(lines)
    )
    atom_union = frozenset(atom for task in tasks for atom in task.formula.atoms())
    if atom_union != ATOMS:
        raise ValueError(f"task atoms differ from frozen environment atoms: {sorted(atom_union)}")
    return tasks


def capture_official_step(
    env: object,
    info: Mapping[str, object],
    *,
    action: object | None = None,
    reward: float | None = None,
) -> PointStepRecord:
    """Capture one post-action step from DeepLTL without importing its packages.

    This intentionally depends only on the public/unwrapped task object shape
    and validates the official ``info['propositions']`` against raw geometry.
    """

    unwrapped = getattr(env, "unwrapped", env)
    task = getattr(unwrapped, "task", None)
    if task is None:
        raise TypeError("environment does not expose unwrapped.task")
    agent = getattr(task, "agent", None)
    if agent is None or not hasattr(agent, "pos"):
        raise TypeError("PointLtl2 task does not expose agent.pos")
    geoms = getattr(task, "_geoms", None)
    if geoms is None:
        raise TypeError("PointLtl2 task does not expose _geoms")
    values: Iterable[object] = geoms.values() if isinstance(geoms, Mapping) else geoms
    zones: list[ZoneGeometry] = []
    for geom in values:
        color = getattr(geom, "color_name", None)
        if color not in ATOMS:
            continue
        centers = tuple(_xy(position, f"{color} zone center") for position in getattr(geom, "pos"))
        zones.append(ZoneGeometry(str(color), centers, float(getattr(geom, "size"))))
    if "propositions" not in info:
        raise KeyError("official step info is missing 'propositions'")
    reported_raw = info["propositions"]
    if isinstance(reported_raw, str) or not isinstance(reported_raw, Iterable):
        raise TypeError("info['propositions'] must be a non-string iterable")
    geometry = PointZoneStep(
        agent_xy=_xy(getattr(agent, "pos"), "agent position"),
        zones=tuple(zones),
        reported_propositions=frozenset(str(item) for item in reported_raw),
    )
    geometry.require_reported_agreement()
    return PointStepRecord(geometry, action, reward)


def preserve_termination_and_truncation(env: object) -> object:
    """Return DeepLTL's inner TimeLimit wrapper with the five-value step API.

    The upstream ``RemoveTruncWrapper`` returns ``done = terminated or
    truncated`` and discards which condition occurred. Auditable collection
    must bypass only that outer wrapper; guessing from ``done`` after the fact
    is forbidden.
    """

    if type(env).__name__ != "RemoveTruncWrapper":
        raise TypeError("expected the official outer RemoveTruncWrapper")
    inner = getattr(env, "env", None)
    if inner is None or type(inner).__name__ != "TimeLimit":
        raise TypeError("RemoveTruncWrapper must directly contain Gymnasium TimeLimit")
    maximum = getattr(inner, "_max_episode_steps", None)
    if maximum != 1000:
        raise ValueError(f"official PointLtl2 TimeLimit must be 1000, found {maximum}")
    return inner


def build_point_trace(
    steps: Sequence[PointStepRecord],
    *,
    policy_id: str | None = None,
    seed: int | None = None,
    task_id: str | None = None,
    terminated: bool = True,
    truncated: bool = False,
) -> LabeledTrace:
    """Ground a sequence of post-action records under the frozen label timing."""

    for step in steps:
        step.geometry.require_reported_agreement()
    return ground_raw_steps(
        steps,
        point_zone_rules(),
        raw_kind="state",
        policy_id=policy_id,
        environment_id=ENVIRONMENT_ID,
        seed=seed,
        terminated=terminated,
        truncated=truncated,
        metadata={
            "task_id": task_id,
            "label_timing": "post_action_before_ldba_transition",
            "transition_payload": "PointStepRecord.action/reward",
        },
    )


def assess_point_episode(
    task: PointTask,
    trace: LabeledTrace,
    *,
    official_success: bool = False,
    official_violation: bool = False,
    evaluation_limits: EvaluationLimits = EvaluationLimits(),
    verify_structural_certificate: bool = True,
    verification_kwargs: Mapping[str, object] | None = None,
    compute_prefix: bool = True,
) -> PointEpisodeAssessment:
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

    rules = point_zone_rules()
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
        # Fast mode still compares the completed trace against the independent
        # temporal evaluator through ``certificate``.  This placeholder is only
        # for callers that do not consume prefix-continuation semantics; its
        # NOT_COMPUTED origin marks it as never decided, not as a proven OPEN.
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
        # official monitor and already checked against the independent finite-
        # trace evaluator.
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
    return PointEpisodeAssessment(disposition, certificate, prefix)
