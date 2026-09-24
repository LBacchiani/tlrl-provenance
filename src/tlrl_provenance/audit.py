"""Fail-closed practitioner entry point for a complete semantic audit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .circuit import DEFAULT_SUPPORT_WORK_LIMIT, SupportEnumerationMode
from .evaluator import SemanticCertificate, evaluate_with_provenance
from .logic import Formula
from .grounding import (
    GroundingVerificationReport,
    PropositionRule,
    verify_grounding_reconstruction,
)
from .statistics import SemanticProfile, aggregate_certificates
from .trace import LabeledTrace
from .verification import VerificationReport, verify_certificate


@dataclass(frozen=True, slots=True)
class AuditConfig:
    require_grounding: bool = True
    confidence: float = 0.95
    support_limit: int = 10_000
    support_work_limit: int = DEFAULT_SUPPORT_WORK_LIMIT
    support_enumeration_mode: SupportEnumerationMode | str = (
        SupportEnumerationMode.BOUNDED_PARTIAL
    )
    support_intermediate_limit: int | None = None
    exhaustive_atom_time_limit: int = 14
    require_complete_support_enumeration: bool = False


@dataclass(frozen=True, slots=True)
class SemanticAudit:
    formula: Formula
    certificates: tuple[SemanticCertificate, ...]
    verification: tuple[VerificationReport, ...]
    grounding_verification: tuple[GroundingVerificationReport, ...]
    profile: SemanticProfile


def audit_traces(
    formula: Formula,
    traces: Iterable[LabeledTrace],
    *,
    config: AuditConfig = AuditConfig(),
    grounding_rules: Iterable[PropositionRule] | None = None,
) -> SemanticAudit:
    traces = tuple(traces)
    if not traces:
        raise ValueError("an audit requires at least one trace")
    certificates: list[SemanticCertificate] = []
    reports: list[VerificationReport] = []
    grounding_reports: list[GroundingVerificationReport] = []
    frozen_rules = None if grounding_rules is None else tuple(grounding_rules)
    for index, trace in enumerate(traces):
        if frozen_rules is not None:
            grounding_report = verify_grounding_reconstruction(trace, frozen_rules)
            if not grounding_report.valid:
                raise AssertionError(
                    f"trace {index} failed grounding replay: {grounding_report.errors}"
                )
            grounding_reports.append(grounding_report)
        certificate = evaluate_with_provenance(
            formula,
            trace,
            require_grounding=config.require_grounding,
            verify_reference=True,
        )
        report = verify_certificate(
            certificate,
            support_limit=config.support_limit,
            support_work_limit=config.support_work_limit,
            support_enumeration_mode=config.support_enumeration_mode,
            support_intermediate_limit=config.support_intermediate_limit,
            exhaustive_atom_time_limit=config.exhaustive_atom_time_limit,
        )
        if not report.valid:
            raise AssertionError(f"certificate {index} failed verification: {report.errors}")
        if config.require_complete_support_enumeration:
            enumeration = certificate.minimal_supports(
                config.support_limit,
                work_limit=config.support_work_limit,
                mode=config.support_enumeration_mode,
                intermediate_limit=config.support_intermediate_limit,
            )
            if not enumeration.complete:
                raise RuntimeError(
                    f"certificate {index} support enumeration incomplete: "
                    f"mode={enumeration.mode.value}, "
                    f"reasons={enumeration.truncation_reasons}, "
                    f"nodes={enumeration.truncated_node_ids}, "
                    f"output_limit={enumeration.limit}, "
                    f"work={enumeration.work_done}/{enumeration.work_limit}, "
                    f"intermediate_limit={enumeration.intermediate_limit}"
                )
        certificates.append(certificate)
        reports.append(report)
    frozen = tuple(certificates)
    return SemanticAudit(
        formula=formula,
        certificates=frozen,
        verification=tuple(reports),
        grounding_verification=tuple(grounding_reports),
        profile=aggregate_certificates(frozen, confidence=config.confidence),
    )
