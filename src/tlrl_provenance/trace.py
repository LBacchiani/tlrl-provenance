"""Labeled finite traces and auditable proposition grounding."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import re
from typing import Iterable, Mapping, Sequence


class FactKind(str, Enum):
    ATOM = "atom"
    END_OF_TRACE = "end_of_trace"


@dataclass(frozen=True, slots=True)
class EvidenceFact:
    kind: FactKind
    time: int
    value: bool = True
    proposition: str | None = None

    def __post_init__(self) -> None:
        if self.time < 0:
            raise ValueError("evidence time must be non-negative")
        if self.kind is FactKind.ATOM and not self.proposition:
            raise ValueError("atomic evidence requires a proposition")
        if self.kind is FactKind.ATOM and not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", self.proposition or ""
        ):
            raise ValueError(f"invalid evidence proposition: {self.proposition!r}")
        if self.kind is FactKind.END_OF_TRACE:
            if self.proposition is not None or self.value is not True:
                raise ValueError("end-of-trace evidence has no proposition and is true")

    def stable_key(self) -> tuple[str, int, str, bool]:
        return (self.kind.value, self.time, self.proposition or "", self.value)

    def to_source(self) -> str:
        if self.kind is FactKind.END_OF_TRACE:
            return f"END@{self.time}"
        prefix = "" if self.value else "not "
        return f"{prefix}{self.proposition}@{self.time}"


@dataclass(frozen=True, slots=True)
class GroundingRecord:
    proposition: str
    time: int
    value: bool
    rule_id: str
    rule_version: str
    raw_fields: tuple[str, ...] = ()
    source_digest: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.proposition or not self.rule_id or not self.rule_version:
            raise ValueError("grounding proposition, rule_id, and rule_version are required")
        if self.time < 0:
            raise ValueError("grounding time must be non-negative")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.proposition):
            raise ValueError(f"invalid grounding proposition: {self.proposition!r}")
        if any(not isinstance(item, str) or not item for item in self.raw_fields):
            raise ValueError("raw_fields must contain nonempty strings")
        if self.source_digest is not None and not re.fullmatch(
            r"[0-9a-fA-F]{64}", self.source_digest
        ):
            raise ValueError("source_digest must be a SHA-256 hexadecimal digest")
        object.__setattr__(self, "raw_fields", tuple(self.raw_fields))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class LabeledTrace:
    labels: Sequence[Iterable[str]]
    groundings: Sequence[GroundingRecord] = ()
    states: Sequence[object] = ()
    observations: Sequence[object] = ()
    actions: Sequence[object] = ()
    rewards: Sequence[float] = ()
    policy_id: str | None = None
    environment_id: str | None = None
    seed: int | None = None
    terminated: bool = True
    truncated: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        labels = tuple(frozenset(step) for step in self.labels)
        if not labels:
            raise ValueError("a finite trace requires at least one position")
        if self.terminated and self.truncated:
            raise ValueError("a trace cannot be both terminated and truncated")
        for time, step in enumerate(labels):
            for proposition in step:
                if not isinstance(proposition, str) or not re.fullmatch(
                    r"[A-Za-z_][A-Za-z0-9_]*", proposition
                ):
                    raise ValueError(f"invalid proposition at time {time}: {proposition!r}")
        for name, sequence in (("states", self.states), ("observations", self.observations)):
            if sequence and len(sequence) != len(labels):
                raise ValueError(f"{name} must be empty or match label length")
        for name, sequence in (("actions", self.actions), ("rewards", self.rewards)):
            if len(sequence) > len(labels) - 1:
                raise ValueError(f"{name} cannot exceed the transition horizon")
        if self.actions and self.rewards and len(self.actions) != len(self.rewards):
            raise ValueError("actions and rewards must have equal lengths when both are present")
        if any(not math.isfinite(float(reward)) for reward in self.rewards):
            raise ValueError("rewards must be finite")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "groundings", tuple(self.groundings))
        object.__setattr__(self, "states", tuple(self.states))
        object.__setattr__(self, "observations", tuple(self.observations))
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "rewards", tuple(float(x) for x in self.rewards))
        object.__setattr__(self, "metadata", dict(self.metadata))
        self._validate_groundings()

    @property
    def horizon(self) -> int:
        return len(self.labels) - 1

    def atom_value(self, proposition: str, time: int) -> bool:
        return proposition in self.labels[time]

    def atom_fact(self, proposition: str, time: int) -> EvidenceFact:
        return EvidenceFact(
            FactKind.ATOM,
            time=time,
            proposition=proposition,
            value=self.atom_value(proposition, time),
        )

    def grounding_coverage(self, propositions: Iterable[str]) -> tuple[int, int]:
        expected = {(time, proposition) for time in range(len(self.labels)) for proposition in propositions}
        present = {(item.time, item.proposition) for item in self.groundings}
        return len(expected & present), len(expected)

    def require_complete_grounding(self, propositions: Iterable[str]) -> None:
        propositions = frozenset(propositions)
        present = {(item.time, item.proposition) for item in self.groundings}
        missing = [
            (time, proposition)
            for time in range(len(self.labels))
            for proposition in sorted(propositions)
            if (time, proposition) not in present
        ]
        if missing:
            preview = ", ".join(f"{p}@{t}" for t, p in missing[:8])
            raise ValueError(f"missing {len(missing)} proposition groundings: {preview}")

    def _validate_groundings(self) -> None:
        seen: set[tuple[int, str]] = set()
        for item in self.groundings:
            key = (item.time, item.proposition)
            if key in seen:
                raise ValueError(f"duplicate grounding for {item.proposition}@{item.time}")
            seen.add(key)
            if item.time > self.horizon:
                raise ValueError(f"grounding time {item.time} exceeds trace horizon")
            observed = item.proposition in self.labels[item.time]
            if observed != item.value:
                raise ValueError(
                    f"grounding for {item.proposition}@{item.time} says {item.value}, labels say {observed}"
                )
