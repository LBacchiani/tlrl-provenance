"""Independent and structural certificate verification."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from .circuit import DEFAULT_SUPPORT_WORK_LIMIT, NodeKind, SupportEnumerationMode
from .evaluator import SemanticCertificate
from .reference import evaluate_reference, reference_value_table
from .trace import EvidenceFact, FactKind


@dataclass(frozen=True, slots=True)
class VerificationReport:
    valid: bool
    checked_cells: int
    checked_supports: int
    support_enumeration_complete: bool
    support_enumeration_mode: SupportEnumerationMode
    support_work_done: int
    support_work_limit: int
    support_intermediate_limit: int | None
    support_truncation_reasons: tuple[str, ...]
    support_truncated_node_ids: tuple[int, ...]
    exhaustive_support_checks: bool
    errors: tuple[str, ...]


def verify_certificate(
    certificate: SemanticCertificate,
    *,
    support_limit: int = 10_000,
    support_work_limit: int = DEFAULT_SUPPORT_WORK_LIMIT,
    support_enumeration_mode: SupportEnumerationMode | str = (
        SupportEnumerationMode.BOUNDED_PARTIAL
    ),
    support_intermediate_limit: int | None = None,
    exhaustive_atom_time_limit: int = 14,
) -> VerificationReport:
    errors: list[str] = []
    table = reference_value_table(certificate.formula, certificate.trace.labels)
    root_reference = table[certificate.formula][0]
    if root_reference != certificate.satisfied:
        errors.append("root verdict disagrees with independent reference evaluator")

    occurrences = certificate.indexed_formula.by_id()
    for cell in certificate.cells:
        expected = table[occurrences[cell.occurrence_id].formula][cell.time]
        if expected != cell.truth:
            errors.append(f"cell disagreement at {cell.occurrence_id}@{cell.time}")

    for node in certificate.circuit.nodes:
        if node.kind is NodeKind.EVIDENCE:
            assert node.evidence is not None
            if not _fact_matches_trace(node.evidence, certificate):
                errors.append(f"false evidence node {node.id}: {node.evidence.to_source()}")

    enumeration = certificate.minimal_supports(
        support_limit,
        work_limit=support_work_limit,
        mode=support_enumeration_mode,
        intermediate_limit=support_intermediate_limit,
    )
    for support in enumeration.supports:
        if not all(_fact_matches_trace(fact, certificate) for fact in support):
            errors.append("support contains evidence contradicted by the trace")

    universe_size = len(certificate.formula.atoms()) * len(certificate.trace.labels)
    exhaustive = universe_size <= exhaustive_atom_time_limit
    if exhaustive:
        for support in enumeration.supports:
            if not support_entails_verdict(certificate, support):
                errors.append("a derivational support does not entail the reported verdict")

    return VerificationReport(
        valid=not errors,
        checked_cells=len(certificate.cells),
        checked_supports=len(enumeration.supports),
        support_enumeration_complete=enumeration.complete,
        support_enumeration_mode=enumeration.mode,
        support_work_done=enumeration.work_done,
        support_work_limit=enumeration.work_limit,
        support_intermediate_limit=enumeration.intermediate_limit,
        support_truncation_reasons=enumeration.truncation_reasons,
        support_truncated_node_ids=enumeration.truncated_node_ids,
        exhaustive_support_checks=exhaustive,
        errors=tuple(errors),
    )


def support_entails_verdict(
    certificate: SemanticCertificate,
    support: frozenset[EvidenceFact],
) -> bool:
    """Exhaustively check a support against all unspecified AP valuations."""

    atoms = sorted(certificate.formula.atoms())
    fixed = {
        (fact.time, fact.proposition): fact.value
        for fact in support
        if fact.kind is FactKind.ATOM
    }
    free = [
        (time, atom)
        for time in range(len(certificate.trace.labels))
        for atom in atoms
        if (time, atom) not in fixed
    ]
    for assignment in product((False, True), repeat=len(free)):
        values = dict(fixed)
        values.update(zip(free, assignment))
        labels = tuple(
            frozenset(atom for atom in atoms if values[(time, atom)])
            for time in range(len(certificate.trace.labels))
        )
        if evaluate_reference(certificate.formula, labels) != certificate.satisfied:
            return False
    return True


def _fact_matches_trace(fact: EvidenceFact, certificate: SemanticCertificate) -> bool:
    if fact.kind is FactKind.END_OF_TRACE:
        return fact.time == certificate.trace.horizon
    assert fact.proposition is not None
    return certificate.trace.atom_value(fact.proposition, fact.time) == fact.value
