"""Proof-producing dynamic-programming semantics for finite LTL traces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .circuit import (
    DEFAULT_SUPPORT_WORK_LIMIT,
    CircuitBuilder,
    ProvenanceCircuit,
    SupportEnumeration,
    SupportEnumerationMode,
)
from .logic import Formula, IndexedFormula, Occurrence, Op
from .reference import evaluate_reference, evaluate_reference_dp
from .trace import EvidenceFact, FactKind, LabeledTrace


class Verdict(str, Enum):
    SATISFIED = "satisfied"
    VIOLATED = "violated"


@dataclass(frozen=True, slots=True)
class EvaluationLimits:
    max_trace_positions: int = 1_000_000
    max_formula_occurrences: int = 100_000
    max_formula_depth: int = 512
    max_cells: int = 5_000_000
    max_circuit_nodes: int = 10_000_000
    collect_decisions: bool = True

    def __post_init__(self) -> None:
        if min(
            self.max_trace_positions,
            self.max_formula_occurrences,
            self.max_formula_depth,
            self.max_cells,
            self.max_circuit_nodes,
        ) < 1:
            raise ValueError("all evaluation limits must be positive")


@dataclass(frozen=True, slots=True)
class EvaluationCell:
    occurrence_id: str
    time: int
    truth: bool
    proof_node: int


@dataclass(frozen=True, slots=True)
class SemanticDecision:
    occurrence_id: str
    time: int
    operator: str
    truth: bool
    kind: str
    alternatives: tuple[str, ...]

    @property
    def signature(self) -> str:
        joined = ",".join(self.alternatives)
        return f"{self.occurrence_id}@{self.time}:{self.kind}:{joined}"


@dataclass(frozen=True, slots=True)
class SemanticCertificate:
    formula: Formula
    indexed_formula: IndexedFormula
    trace: LabeledTrace
    verdict: Verdict
    circuit: ProvenanceCircuit
    cells: tuple[EvaluationCell, ...]
    decisions: tuple[SemanticDecision, ...]
    grounded_count: int
    grounding_expected_count: int

    @property
    def satisfied(self) -> bool:
        return self.verdict is Verdict.SATISFIED

    @property
    def grounding_complete(self) -> bool:
        return self.grounded_count == self.grounding_expected_count

    def cell(self, occurrence_id: str, time: int) -> EvaluationCell:
        for cell in self.cells:
            if cell.occurrence_id == occurrence_id and cell.time == time:
                return cell
        raise KeyError((occurrence_id, time))

    def minimal_supports(
        self,
        limit: int = 10_000,
        *,
        work_limit: int = DEFAULT_SUPPORT_WORK_LIMIT,
        mode: SupportEnumerationMode | str = SupportEnumerationMode.EXACT,
        intermediate_limit: int | None = None,
    ) -> SupportEnumeration:
        return self.circuit.minimal_supports(
            limit,
            work_limit=work_limit,
            mode=mode,
            intermediate_limit=intermediate_limit,
        )

    @property
    def necessary_evidence(self) -> frozenset[EvidenceFact]:
        return self.circuit.necessary_evidence()

    @property
    def possible_evidence(self) -> frozenset[EvidenceFact]:
        return self.circuit.possible_evidence()


def evaluate_with_provenance(
    formula: Formula,
    trace: LabeledTrace,
    *,
    require_grounding: bool = False,
    verify_reference: bool | str = True,
    limits: EvaluationLimits = EvaluationLimits(),
) -> SemanticCertificate:
    """Evaluate `formula` and retain all viable operational derivations.

    The returned monotone circuit is a compact representation of alternatives;
    it is not expanded into exponentially many support sets unless explicitly
    requested with a hard limit.
    """

    indexed = IndexedFormula.build(formula)
    if len(trace.labels) > limits.max_trace_positions:
        raise RuntimeError(f"trace exceeds position limit {limits.max_trace_positions}")
    if len(indexed.occurrences) > limits.max_formula_occurrences:
        raise RuntimeError(
            f"formula exceeds occurrence limit {limits.max_formula_occurrences}"
        )
    formula_depth = max(len(item.path) + 1 for item in indexed.occurrences)
    if formula_depth > limits.max_formula_depth:
        raise RuntimeError(f"formula depth {formula_depth} exceeds limit {limits.max_formula_depth}")
    cell_count = len(trace.labels) * len(indexed.occurrences)
    if cell_count > limits.max_cells:
        raise RuntimeError(f"evaluation requires {cell_count} cells, limit is {limits.max_cells}")
    if require_grounding:
        trace.require_complete_grounding(formula.atoms())
    grounded_count, grounding_expected = trace.grounding_coverage(formula.atoms())
    builder = CircuitBuilder(max_nodes=limits.max_circuit_nodes)
    truth: dict[tuple[str, int], bool] = {}
    proof: dict[tuple[str, int], int] = {}
    horizon = trace.horizon
    occurrence_by_id = indexed.by_id()

    def child(occurrence: Occurrence, index: int) -> Occurrence:
        return occurrence_by_id[occurrence.child_ids[index]]

    def record(occurrence: Occurrence, time: int, result: bool, proof_node: int) -> None:
        truth[(occurrence.id, time)] = result
        proof[(occurrence.id, time)] = proof_node

    for occurrence in indexed.postorder():
        op = occurrence.formula.op
        for time in range(horizon, -1, -1):
            if op is Op.ATOM:
                assert occurrence.formula.name is not None
                fact = trace.atom_fact(occurrence.formula.name, time)
                record(occurrence, time, fact.value, builder.evidence(fact))
                continue
            if op is Op.TOP:
                record(occurrence, time, True, builder.true)
                continue
            if op is Op.BOTTOM:
                record(occurrence, time, False, builder.true)
                continue

            children = [child(occurrence, i) for i in range(len(occurrence.child_ids))]
            child_truth = [truth[(item.id, time)] for item in children]
            child_proof = [proof[(item.id, time)] for item in children]

            if op is Op.NOT:
                record(occurrence, time, not child_truth[0], child_proof[0])
            elif op is Op.AND:
                result = child_truth[0] and child_truth[1]
                node = (
                    builder.all_of(child_proof)
                    if result
                    else builder.any_of(child_proof[i] for i in range(2) if not child_truth[i])
                )
                record(occurrence, time, result, node)
            elif op is Op.OR:
                result = child_truth[0] or child_truth[1]
                node = (
                    builder.any_of(child_proof[i] for i in range(2) if child_truth[i])
                    if result
                    else builder.all_of(child_proof)
                )
                record(occurrence, time, result, node)
            elif op is Op.IMPLIES:
                result = (not child_truth[0]) or child_truth[1]
                if result:
                    options = []
                    if not child_truth[0]:
                        options.append(child_proof[0])
                    if child_truth[1]:
                        options.append(child_proof[1])
                    node = builder.any_of(options)
                else:
                    node = builder.all_of(child_proof)
                record(occurrence, time, result, node)
            elif op is Op.IFF:
                result = child_truth[0] == child_truth[1]
                record(occurrence, time, result, builder.all_of(child_proof))
            elif op in {Op.NEXT, Op.WEAK_NEXT}:
                if time == horizon:
                    result = op is Op.WEAK_NEXT
                    boundary = EvidenceFact(FactKind.END_OF_TRACE, time=time)
                    record(occurrence, time, result, builder.evidence(boundary))
                else:
                    target = children[0]
                    record(
                        occurrence,
                        time,
                        truth[(target.id, time + 1)],
                        proof[(target.id, time + 1)],
                    )
            elif op in {Op.EVENTUALLY, Op.GLOBALLY}:
                if time == horizon:
                    record(occurrence, time, child_truth[0], child_proof[0])
                    continue
                future_truth = truth[(occurrence.id, time + 1)]
                future_proof = proof[(occurrence.id, time + 1)]
                if op is Op.EVENTUALLY:
                    result = child_truth[0] or future_truth
                    node = (
                        builder.any_of(
                            node
                            for enabled, node in (
                                (child_truth[0], child_proof[0]),
                                (future_truth, future_proof),
                            )
                            if enabled
                        )
                        if result
                        else builder.all_of((child_proof[0], future_proof))
                    )
                else:
                    result = child_truth[0] and future_truth
                    node = (
                        builder.all_of((child_proof[0], future_proof))
                        if result
                        else builder.any_of(
                            node
                            for enabled, node in (
                                (not child_truth[0], child_proof[0]),
                                (not future_truth, future_proof),
                            )
                            if enabled
                        )
                    )
                record(occurrence, time, result, node)
            elif op in {Op.UNTIL, Op.WEAK_UNTIL}:
                left_truth, right_truth = child_truth
                left_proof, right_proof = child_proof
                if time == horizon:
                    if op is Op.UNTIL:
                        record(occurrence, time, right_truth, right_proof)
                    else:
                        result = right_truth or left_truth
                        node = (
                            builder.any_of(
                                node
                                for enabled, node in (
                                    (right_truth, right_proof),
                                    (left_truth, left_proof),
                                )
                                if enabled
                            )
                            if result
                            else builder.all_of((right_proof, left_proof))
                        )
                        record(occurrence, time, result, node)
                    continue
                future_truth = truth[(occurrence.id, time + 1)]
                future_proof = proof[(occurrence.id, time + 1)]
                continuation_truth = left_truth and future_truth
                continuation_proof = (
                    builder.all_of((left_proof, future_proof))
                    if continuation_truth
                    else builder.any_of(
                        node
                        for enabled, node in (
                            (not left_truth, left_proof),
                            (not future_truth, future_proof),
                        )
                        if enabled
                    )
                )
                result = right_truth or continuation_truth
                node = (
                    builder.any_of(
                        candidate
                        for enabled, candidate in (
                            (right_truth, right_proof),
                            (continuation_truth, continuation_proof),
                        )
                        if enabled
                    )
                    if result
                    else builder.all_of((right_proof, continuation_proof))
                )
                record(occurrence, time, result, node)
            elif op is Op.RELEASE:
                left_truth, right_truth = child_truth
                left_proof, right_proof = child_proof
                if time == horizon:
                    record(occurrence, time, right_truth, right_proof)
                    continue
                future_truth = truth[(occurrence.id, time + 1)]
                future_proof = proof[(occurrence.id, time + 1)]
                inner_truth = left_truth or future_truth
                inner_proof = (
                    builder.any_of(
                        node
                        for enabled, node in (
                            (left_truth, left_proof),
                            (future_truth, future_proof),
                        )
                        if enabled
                    )
                    if inner_truth
                    else builder.all_of((left_proof, future_proof))
                )
                result = right_truth and inner_truth
                node = (
                    builder.all_of((right_proof, inner_proof))
                    if result
                    else builder.any_of(
                        candidate
                        for enabled, candidate in (
                            (not right_truth, right_proof),
                            (not inner_truth, inner_proof),
                        )
                        if enabled
                    )
                )
                record(occurrence, time, result, node)
            else:
                raise AssertionError(op)

    root_key = (indexed.root.id, 0)
    root_truth = truth[root_key]
    if verify_reference:
        if verify_reference not in {True, "linear", "direct"}:
            raise ValueError("verify_reference must be False, True, 'linear', or 'direct'")
        reference = (
            evaluate_reference(formula, trace.labels)
            if verify_reference == "direct"
            else evaluate_reference_dp(formula, trace.labels)
        )
        if reference != root_truth:
            raise AssertionError(
                f"provenance and reference semantics disagree: {root_truth} != {reference}"
            )
    cells = tuple(
        EvaluationCell(occurrence.id, time, truth[(occurrence.id, time)], proof[(occurrence.id, time)])
        for occurrence in indexed.occurrences
        for time in range(horizon + 1)
    )
    decisions = _semantic_decisions(indexed, truth, horizon) if limits.collect_decisions else ()
    return SemanticCertificate(
        formula=formula,
        indexed_formula=indexed,
        trace=trace,
        verdict=Verdict.SATISFIED if root_truth else Verdict.VIOLATED,
        circuit=builder.freeze(proof[root_key]),
        cells=cells,
        decisions=decisions,
        grounded_count=grounded_count,
        grounding_expected_count=grounding_expected,
    )


def _semantic_decisions(
    indexed: IndexedFormula,
    truth: dict[tuple[str, int], bool],
    horizon: int,
) -> tuple[SemanticDecision, ...]:
    decisions: list[SemanticDecision] = []

    def values(occurrence: Occurrence, time: int) -> list[bool]:
        return [truth[(child_id, time)] for child_id in occurrence.child_ids]

    for occurrence in indexed.occurrences:
        op = occurrence.formula.op
        if op in {Op.ATOM, Op.TOP, Op.BOTTOM}:
            continue
        for time in range(horizon + 1):
            result = truth[(occurrence.id, time)]
            children = values(occurrence, time)
            kind: str
            alternatives: tuple[str, ...]
            if op is Op.NOT:
                kind, alternatives = "negation", ("child_false" if result else "child_true",)
            elif op is Op.AND:
                kind = "conjunction_support" if result else "failing_conjunct"
                alternatives = (
                    ("left", "right")
                    if result
                    else tuple(side for side, value in zip(("left", "right"), children) if not value)
                )
            elif op is Op.OR:
                kind = "satisfying_disjunct" if result else "all_disjuncts_false"
                alternatives = (
                    tuple(side for side, value in zip(("left", "right"), children) if value)
                    if result
                    else ("left", "right")
                )
            elif op is Op.IMPLIES:
                kind = "implication_support" if result else "implication_violation"
                alternatives = tuple(
                    label
                    for enabled, label in (
                        (not children[0], "antecedent_false"),
                        (children[1], "consequent_true"),
                    )
                    if enabled
                ) or ("antecedent_true_and_consequent_false",)
            elif op is Op.IFF:
                kind = "equivalence_match" if result else "equivalence_mismatch"
                alternatives = (f"left_{str(children[0]).lower()}_right_{str(children[1]).lower()}",)
            elif op in {Op.NEXT, Op.WEAK_NEXT}:
                kind = "trace_boundary" if time == horizon else "next_position"
                alternatives = (("end_of_trace",) if time == horizon else (f"time@{time + 1}",))
            elif op is Op.EVENTUALLY:
                kind = "eventuality_progress" if result else "eventuality_unfulfilled"
                alternatives = (
                    tuple(
                        label
                        for enabled, label in (
                            (children[0], "witness_now"),
                            (
                                time < horizon and truth.get((occurrence.id, time + 1), False),
                                "witness_in_future",
                            ),
                        )
                        if enabled
                    )
                    if result
                    else ("none",)
                )
            elif op is Op.GLOBALLY:
                kind = "global_progress" if result else "global_counterexample"
                alternatives = (
                    ("current_and_remaining",)
                    if result
                    else tuple(
                        label
                        for enabled, label in (
                            (not children[0], "failure_now"),
                            (
                                time < horizon and not truth.get((occurrence.id, time + 1), True),
                                "failure_in_future",
                            ),
                        )
                        if enabled
                    )
                )
            elif op in {Op.UNTIL, Op.WEAK_UNTIL}:
                left_id, right_id = occurrence.child_ids
                if result:
                    kind = "until_progress" if op is Op.UNTIL else "weak_until_progress"
                    alternatives = tuple(
                        label
                        for enabled, label in (
                            (truth[(right_id, time)], "right_now"),
                            (
                                time < horizon
                                and truth[(left_id, time)]
                                and truth.get((occurrence.id, time + 1), False),
                                "continue",
                            ),
                            (
                                op is Op.WEAK_UNTIL
                                and time == horizon
                                and truth[(left_id, time)],
                                "left_at_end",
                            ),
                        )
                        if enabled
                    )
                else:
                    kind = (
                        "until_unfulfilled"
                        if op is Op.UNTIL
                        else "weak_until_unfulfilled"
                    )
                    alternatives = ("none",)
            elif op is Op.RELEASE:
                left_id, right_id = occurrence.child_ids
                if result:
                    kind = "release_progress"
                    alternatives = (
                        ("right_at_end",)
                        if time == horizon
                        else tuple(
                            label
                            for enabled, label in (
                                (
                                    truth[(right_id, time)] and truth[(left_id, time)],
                                    "right_and_release_now",
                                ),
                                (
                                    truth[(right_id, time)]
                                    and truth.get((occurrence.id, time + 1), False),
                                    "right_and_continue",
                                ),
                            )
                            if enabled
                        )
                    )
                else:
                    alternatives = tuple(
                        label
                        for enabled, label in (
                            (not truth[(right_id, time)], "right_false_now"),
                            (
                                time < horizon
                                and not truth[(left_id, time)]
                                and not truth.get((occurrence.id, time + 1), True),
                                "no_release_continuation",
                            ),
                        )
                        if enabled
                    )
                    kind = "release_violation"
            else:
                raise AssertionError(op)
            decisions.append(
                SemanticDecision(
                    occurrence_id=occurrence.id,
                    time=time,
                    operator=op.value,
                    truth=result,
                    kind=kind,
                    alternatives=alternatives,
                )
            )
    return tuple(decisions)


def temporal_landmarks(
    certificate: SemanticCertificate,
    occurrence_id: str,
    time: int = 0,
) -> SemanticDecision:
    """Expand absolute temporal witnesses/counterexamples on demand."""

    occurrences = certificate.indexed_formula.by_id()
    if occurrence_id not in occurrences:
        raise KeyError(occurrence_id)
    occurrence = occurrences[occurrence_id]
    if time < 0 or time > certificate.trace.horizon:
        raise IndexError(time)
    truth = {(cell.occurrence_id, cell.time): cell.truth for cell in certificate.cells}
    horizon = certificate.trace.horizon
    result = truth[(occurrence_id, time)]
    op = occurrence.formula.op
    if op is Op.EVENTUALLY:
        alternatives = tuple(
            f"time@{j}"
            for j in range(time, horizon + 1)
            if truth[(occurrence.child_ids[0], j)]
        )
        kind = "eventuality_witness" if result else "eventuality_unfulfilled"
    elif op is Op.GLOBALLY:
        alternatives = tuple(
            f"time@{j}"
            for j in range(time, horizon + 1)
            if not truth[(occurrence.child_ids[0], j)]
        )
        kind = "global_invariant" if result else "global_counterexample"
        if result:
            alternatives = ("all_positions",)
    elif op in {Op.UNTIL, Op.WEAK_UNTIL}:
        left_id, right_id = occurrence.child_ids
        alternatives = tuple(
            f"right@{j}"
            for j in range(time, horizon + 1)
            if truth[(right_id, j)]
            and all(truth[(left_id, k)] for k in range(time, j))
        )
        if op is Op.WEAK_UNTIL and all(
            truth[(left_id, k)] for k in range(time, horizon + 1)
        ):
            alternatives += ("left_to_end",)
        kind = (
            "until_witness"
            if result and op is Op.UNTIL
            else "weak_until_witness"
            if result
            else "until_unfulfilled"
            if op is Op.UNTIL
            else "weak_until_unfulfilled"
        )
    elif op is Op.RELEASE:
        left_id, right_id = occurrence.child_ids
        if result:
            alternatives = tuple(
                f"left@{j}"
                for j in range(time, horizon + 1)
                if truth[(left_id, j)]
                and all(truth[(right_id, k)] for k in range(time, j + 1))
            )
            if all(truth[(right_id, k)] for k in range(time, horizon + 1)):
                alternatives += ("right_to_end",)
            kind = "release_witness"
        else:
            alternatives = tuple(
                f"right_failure@{j}"
                for j in range(time, horizon + 1)
                if not truth[(right_id, j)]
                and not any(truth[(left_id, k)] for k in range(time, j))
            )
            kind = "release_violation"
    else:
        raise ValueError(f"{op.value} has no temporal landmark query")
    return SemanticDecision(occurrence_id, time, op.value, result, kind, alternatives or ("none",))


def verdict_relevant_cells(
    certificate: SemanticCertificate,
) -> frozenset[tuple[str, int]]:
    """Cells participating in at least one operational derivation of the root verdict."""

    occurrences = certificate.indexed_formula.by_id()
    truth = {(cell.occurrence_id, cell.time): cell.truth for cell in certificate.cells}
    horizon = certificate.trace.horizon
    reachable: set[tuple[str, int]] = set()
    stack = [(certificate.indexed_formula.root.id, 0)]
    while stack:
        key = stack.pop()
        if key in reachable:
            continue
        reachable.add(key)
        occurrence_id, time = key
        occurrence = occurrences[occurrence_id]
        op = occurrence.formula.op
        result = truth[key]
        children = occurrence.child_ids
        if op in {Op.ATOM, Op.TOP, Op.BOTTOM}:
            continue
        if op is Op.NOT:
            stack.append((children[0], time))
        elif op is Op.AND:
            stack.extend(
                (child_id, time)
                for child_id in children
                if result or not truth[(child_id, time)]
            )
        elif op is Op.OR:
            stack.extend(
                (child_id, time)
                for child_id in children
                if not result or truth[(child_id, time)]
            )
        elif op is Op.IMPLIES:
            if result:
                if not truth[(children[0], time)]:
                    stack.append((children[0], time))
                if truth[(children[1], time)]:
                    stack.append((children[1], time))
            else:
                stack.extend(((children[0], time), (children[1], time)))
        elif op is Op.IFF:
            stack.extend(((children[0], time), (children[1], time)))
        elif op in {Op.NEXT, Op.WEAK_NEXT}:
            if time < horizon:
                stack.append((children[0], time + 1))
        elif op in {Op.EVENTUALLY, Op.GLOBALLY}:
            child_key = (children[0], time)
            future_key = (occurrence_id, time + 1)
            if time == horizon:
                stack.append(child_key)
            elif op is Op.EVENTUALLY:
                if result:
                    if truth[child_key]:
                        stack.append(child_key)
                    if truth[future_key]:
                        stack.append(future_key)
                else:
                    stack.extend((child_key, future_key))
            else:
                if result:
                    stack.extend((child_key, future_key))
                else:
                    if not truth[child_key]:
                        stack.append(child_key)
                    if not truth[future_key]:
                        stack.append(future_key)
        elif op in {Op.UNTIL, Op.WEAK_UNTIL}:
            left_key, right_key = (children[0], time), (children[1], time)
            if time == horizon:
                if op is Op.UNTIL:
                    stack.append(right_key)
                elif result:
                    if truth[right_key]:
                        stack.append(right_key)
                    if truth[left_key]:
                        stack.append(left_key)
                else:
                    stack.extend((left_key, right_key))
                continue
            future_key = (occurrence_id, time + 1)
            continuation = truth[left_key] and truth[future_key]
            if result:
                if truth[right_key]:
                    stack.append(right_key)
                if continuation:
                    stack.extend((left_key, future_key))
            else:
                stack.append(right_key)
                if not truth[left_key]:
                    stack.append(left_key)
                if not truth[future_key]:
                    stack.append(future_key)
        elif op is Op.RELEASE:
            right_key = (children[1], time)
            if time == horizon:
                stack.append(right_key)
                continue
            left_key = (children[0], time)
            future_key = (occurrence_id, time + 1)
            inner = truth[left_key] or truth[future_key]
            if result:
                stack.append(right_key)
                if truth[left_key]:
                    stack.append(left_key)
                if truth[future_key]:
                    stack.append(future_key)
            else:
                if not truth[right_key]:
                    stack.append(right_key)
                if not inner:
                    stack.extend((left_key, future_key))
        else:
            raise AssertionError(op)
    return frozenset(reachable)
