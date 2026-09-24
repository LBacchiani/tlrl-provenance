"""Compatibility and convenience queries over occurrence-level semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .logic import Formula, IndexedFormula, Op, Top, Bottom
from .reference import evaluate_reference
from .trace import LabeledTrace


class Polarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    MIXED = "mixed"

    def flip(self) -> "Polarity":
        if self is Polarity.POSITIVE:
            return Polarity.NEGATIVE
        if self is Polarity.NEGATIVE:
            return Polarity.POSITIVE
        return self


@dataclass(frozen=True, slots=True)
class OccurrenceMutationQuery:
    occurrence_id: str
    polarity: Polarity
    original_verdict: bool
    strengthened_verdict: bool
    weakened_verdict: bool
    pass_necessary: bool | None
    failure_repairable: bool | None


def occurrence_polarities(formula: Formula) -> dict[str, Polarity]:
    indexed = IndexedFormula.build(formula)
    result: dict[str, Polarity] = {}

    def visit(occurrence_id: str, polarity: Polarity) -> None:
        occurrence = indexed.by_id()[occurrence_id]
        result[occurrence_id] = polarity
        if occurrence.formula.op is Op.NOT:
            visit(occurrence.child_ids[0], polarity.flip())
        elif occurrence.formula.op is Op.IMPLIES:
            visit(occurrence.child_ids[0], polarity.flip())
            visit(occurrence.child_ids[1], polarity)
        elif occurrence.formula.op is Op.IFF:
            for child_id in occurrence.child_ids:
                visit(child_id, Polarity.MIXED)
        else:
            for child_id in occurrence.child_ids:
                visit(child_id, polarity)

    visit(indexed.root.id, Polarity.POSITIVE)
    return result


def occurrence_mutation_query(
    formula: Formula,
    trace: LabeledTrace,
    occurrence_id: str,
) -> OccurrenceMutationQuery:
    indexed = IndexedFormula.build(formula)
    occurrences = indexed.by_id()
    if occurrence_id not in occurrences:
        raise KeyError(occurrence_id)
    polarity = occurrence_polarities(formula)[occurrence_id]
    if polarity is Polarity.MIXED:
        raise ValueError("polarity-based strengthening is undefined for a mixed-polarity occurrence")
    strengthen_with = Bottom() if polarity is Polarity.POSITIVE else Top()
    weaken_with = Top() if polarity is Polarity.POSITIVE else Bottom()
    path = occurrences[occurrence_id].path
    strengthened = _replace(formula, path, strengthen_with)
    weakened = _replace(formula, path, weaken_with)
    original_verdict = evaluate_reference(formula, trace.labels)
    strengthened_verdict = evaluate_reference(strengthened, trace.labels)
    weakened_verdict = evaluate_reference(weakened, trace.labels)
    if original_verdict and not weakened_verdict:
        raise AssertionError("weakening changed a satisfying trace to failure")
    if not original_verdict and strengthened_verdict:
        raise AssertionError("strengthening changed a failing trace to success")
    return OccurrenceMutationQuery(
        occurrence_id=occurrence_id,
        polarity=polarity,
        original_verdict=original_verdict,
        strengthened_verdict=strengthened_verdict,
        weakened_verdict=weakened_verdict,
        pass_necessary=(not strengthened_verdict) if original_verdict else None,
        failure_repairable=weakened_verdict if not original_verdict else None,
    )


def _replace(formula: Formula, path: tuple[int, ...], replacement: Formula) -> Formula:
    if not path:
        return replacement
    index = path[0]
    children = list(formula.args)
    children[index] = _replace(children[index], path[1:], replacement)
    return Formula(formula.op, tuple(children), formula.name)

