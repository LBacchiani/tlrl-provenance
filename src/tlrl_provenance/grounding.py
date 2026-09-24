"""Reusable proposition-rule registry and independent grounding replay."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from .trace import GroundingRecord, LabeledTrace


@dataclass(frozen=True, slots=True)
class PropositionRule:
    proposition: str
    rule_id: str
    rule_version: str
    evaluator: Callable[[object, int], bool]
    raw_fields: tuple[str, ...] = ()
    source_digest: str | None = None

    def __post_init__(self) -> None:
        if not callable(self.evaluator):
            raise ValueError("proposition evaluator must be callable")
        # GroundingRecord performs the common identifier/digest validation.
        GroundingRecord(
            self.proposition,
            0,
            False,
            self.rule_id,
            self.rule_version,
            self.raw_fields,
            self.resolved_source_digest,
        )

    @property
    def resolved_source_digest(self) -> str:
        return self.source_digest or callable_source_digest(self.evaluator)


@dataclass(frozen=True, slots=True)
class GroundingVerificationReport:
    valid: bool
    checked_records: int
    errors: tuple[str, ...]


def ground_raw_steps(
    raw_steps: Sequence[object],
    rules: Iterable[PropositionRule],
    *,
    raw_kind: str = "state",
    actions: Sequence[object] = (),
    rewards: Sequence[float] = (),
    policy_id: str | None = None,
    environment_id: str | None = None,
    seed: int | None = None,
    terminated: bool = True,
    truncated: bool = False,
    metadata: Mapping[str, object] | None = None,
) -> LabeledTrace:
    """Apply each frozen rule to each raw step and construct a strict trace."""

    raw_steps = tuple(raw_steps)
    if not raw_steps:
        raise ValueError("raw_steps cannot be empty")
    rules = _validated_rules(rules)
    if raw_kind not in {"state", "observation"}:
        raise ValueError("raw_kind must be 'state' or 'observation'")
    labels: list[set[str]] = []
    records: list[GroundingRecord] = []
    for time, raw in enumerate(raw_steps):
        step_labels: set[str] = set()
        for rule in rules:
            value = rule.evaluator(raw, time)
            if not isinstance(value, bool):
                raise TypeError(
                    f"grounding rule {rule.rule_id!r} returned {type(value).__name__}, expected bool"
                )
            if value:
                step_labels.add(rule.proposition)
            records.append(
                GroundingRecord(
                    proposition=rule.proposition,
                    time=time,
                    value=value,
                    rule_id=rule.rule_id,
                    rule_version=rule.rule_version,
                    raw_fields=rule.raw_fields,
                    source_digest=rule.resolved_source_digest,
                )
            )
        labels.append(step_labels)
    storage = {"states": raw_steps} if raw_kind == "state" else {"observations": raw_steps}
    return LabeledTrace(
        labels=labels,
        groundings=records,
        actions=actions,
        rewards=rewards,
        policy_id=policy_id,
        environment_id=environment_id,
        seed=seed,
        terminated=terminated,
        truncated=truncated,
        metadata={} if metadata is None else metadata,
        **storage,
    )


def verify_grounding_reconstruction(
    trace: LabeledTrace,
    rules: Iterable[PropositionRule],
) -> GroundingVerificationReport:
    """Replay frozen proposition code without calling the semantic evaluator."""

    rules = _validated_rules(rules)
    by_proposition = {rule.proposition: rule for rule in rules}
    raw_steps = trace.states or trace.observations
    errors: list[str] = []
    if not raw_steps:
        return GroundingVerificationReport(False, 0, ("trace retains no raw states or observations",))
    records = {(item.time, item.proposition): item for item in trace.groundings}
    checked = 0
    for time, raw in enumerate(raw_steps):
        for proposition, rule in by_proposition.items():
            key = (time, proposition)
            record = records.get(key)
            if record is None:
                errors.append(f"missing record for {proposition}@{time}")
                continue
            checked += 1
            if (
                record.rule_id != rule.rule_id
                or record.rule_version != rule.rule_version
                or record.source_digest != rule.resolved_source_digest
            ):
                errors.append(f"rule provenance mismatch for {proposition}@{time}")
            value = rule.evaluator(raw, time)
            if not isinstance(value, bool):
                errors.append(f"rule {rule.rule_id} returned non-Boolean during replay")
            elif value != record.value or value != trace.atom_value(proposition, time):
                errors.append(f"reconstruction mismatch for {proposition}@{time}")
    expected = len(raw_steps) * len(rules)
    if len(records) != expected:
        errors.append(f"record count {len(records)} does not match expected {expected}")
    return GroundingVerificationReport(not errors, checked, tuple(errors))


def callable_source_digest(function: Callable[..., object]) -> str:
    """Hash callable source with a stable module/qualname prefix."""

    identity = f"{getattr(function, '__module__', '')}:{getattr(function, '__qualname__', '')}\n"
    try:
        payload = inspect.getsource(function)
    except (OSError, TypeError):
        module = inspect.getmodule(function)
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            raise ValueError("cannot establish source provenance for callable")
        payload = Path(module_file).read_text(encoding="utf-8")
    return hashlib.sha256((identity + payload).encode("utf-8")).hexdigest()


def _validated_rules(rules: Iterable[PropositionRule]) -> tuple[PropositionRule, ...]:
    rules = tuple(rules)
    if not rules:
        raise ValueError("at least one proposition rule is required")
    propositions = [item.proposition for item in rules]
    rule_ids = [item.rule_id for item in rules]
    if len(set(propositions)) != len(propositions):
        raise ValueError("proposition rules must have unique proposition names")
    if len(set(rule_ids)) != len(rule_ids):
        raise ValueError("proposition rules must have unique rule IDs")
    return tuple(sorted(rules, key=lambda item: item.proposition))
