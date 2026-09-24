"""Statistical semantic provenance for temporal-logic-guided RL."""

from .circuit import ProvenanceCircuit, SupportEnumeration, SupportEnumerationMode
from .campaign import (
    AuditSession,
    AuditSubject,
    AuditTask,
    CampaignAdapter,
    build_plan,
    build_summary,
    freeze_plan,
    render_markdown,
)
from .audit import AuditConfig, SemanticAudit, audit_traces
from .artifacts import replay_certificate_artifact, write_certificate_artifact
from .compat import from_legacy_formula, from_v2_episode
from .evaluator import (
    EvaluationCell,
    EvaluationLimits,
    SemanticCertificate,
    SemanticDecision,
    Verdict,
    evaluate_with_provenance,
    temporal_landmarks,
    verdict_relevant_cells,
)
from .logic import Formula, IndexedFormula, Op, parse_formula
from .grounding import (
    GroundingVerificationReport,
    PropositionRule,
    callable_source_digest,
    ground_raw_steps,
    verify_grounding_reconstruction,
)
from .prefix import DecisionOrigin, PrefixAssessment, PrefixStatus, assess_prefix
from .provenance import ENGINE_VERSION, SEMANTICS_VERSION, engine_source_digest
from .queries import (
    OccurrenceMutationQuery,
    Polarity,
    occurrence_mutation_query,
    occurrence_polarities,
)
from .serialization import canonical_json, certificate_digest, certificate_to_dict
from .statistics import (
    EventKey,
    PermutationTestResult,
    SemanticProfile,
    aggregate_certificates,
    bonferroni_alpha,
    fisher_exact_two_sided,
    permutation_test_group_range,
    permutation_test_median_range,
    permutation_test_paired_group_range,
    permutation_test_paired_median_range,
    policy_cluster_bootstrap_event,
    policy_cluster_bootstrap_satisfaction,
)
from .trace import EvidenceFact, FactKind, GroundingRecord, LabeledTrace
from .verification import VerificationReport, verify_certificate

__all__ = [
    "EvaluationCell",
    "EvaluationLimits",
    "ENGINE_VERSION",
    "AuditConfig",
    "AuditSession",
    "AuditSubject",
    "AuditTask",
    "EvidenceFact",
    "EventKey",
    "FactKind",
    "Formula",
    "CampaignAdapter",
    "GroundingRecord",
    "GroundingVerificationReport",
    "IndexedFormula",
    "LabeledTrace",
    "Op",
    "OccurrenceMutationQuery",
    "Polarity",
    "PermutationTestResult",
    "DecisionOrigin",
    "PrefixAssessment",
    "PrefixStatus",
    "PropositionRule",
    "ProvenanceCircuit",
    "SemanticCertificate",
    "SemanticAudit",
    "SemanticDecision",
    "SemanticProfile",
    "SEMANTICS_VERSION",
    "SupportEnumeration",
    "SupportEnumerationMode",
    "Verdict",
    "VerificationReport",
    "aggregate_certificates",
    "audit_traces",
    "bonferroni_alpha",
    "build_plan",
    "build_summary",
    "callable_source_digest",
    "assess_prefix",
    "canonical_json",
    "certificate_digest",
    "certificate_to_dict",
    "fisher_exact_two_sided",
    "evaluate_with_provenance",
    "engine_source_digest",
    "temporal_landmarks",
    "verdict_relevant_cells",
    "from_legacy_formula",
    "from_v2_episode",
    "freeze_plan",
    "ground_raw_steps",
    "parse_formula",
    "occurrence_mutation_query",
    "occurrence_polarities",
    "permutation_test_group_range",
    "permutation_test_median_range",
    "permutation_test_paired_group_range",
    "permutation_test_paired_median_range",
    "policy_cluster_bootstrap_satisfaction",
    "policy_cluster_bootstrap_event",
    "replay_certificate_artifact",
    "render_markdown",
    "verify_certificate",
    "verify_grounding_reconstruction",
    "write_certificate_artifact",
]
