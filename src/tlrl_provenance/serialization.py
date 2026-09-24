"""Canonical, deterministic serialization for semantic certificates."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .circuit import CircuitNode, NodeKind
from .evaluator import SemanticCertificate
from .provenance import ENGINE_VERSION, SEMANTICS_VERSION, engine_source_digest
from .trace import EvidenceFact


SCHEMA_VERSION = "tlrl-semantic-certificate/1"


def certificate_to_dict(certificate: SemanticCertificate) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "engine": {
            "version": ENGINE_VERSION,
            "semantics_version": SEMANTICS_VERSION,
            "source_sha256": engine_source_digest(),
        },
        "formula": certificate.formula.to_source(),
        "verdict": certificate.verdict.value,
        "trace": {
            "labels": [sorted(labels) for labels in certificate.trace.labels],
            "policy_id": certificate.trace.policy_id,
            "environment_id": certificate.trace.environment_id,
            "seed": certificate.trace.seed,
            "terminated": certificate.trace.terminated,
            "truncated": certificate.trace.truncated,
        },
        "grounding": {
            "present": certificate.grounded_count,
            "expected": certificate.grounding_expected_count,
            "complete": certificate.grounding_complete,
            "records": [
                {
                    "proposition": item.proposition,
                    "time": item.time,
                    "value": item.value,
                    "rule_id": item.rule_id,
                    "rule_version": item.rule_version,
                    "raw_fields": list(item.raw_fields),
                    "source_digest": item.source_digest,
                    "metadata": _json_ready(item.metadata),
                }
                for item in sorted(
                    certificate.trace.groundings,
                    key=lambda record: (record.time, record.proposition, record.rule_id),
                )
            ],
        },
        "occurrences": [
            {
                "id": item.id,
                "path": list(item.path),
                "operator": item.formula.op.value,
                "source": item.formula.to_source(),
                "parent_id": item.parent_id,
                "child_ids": list(item.child_ids),
            }
            for item in certificate.indexed_formula.occurrences
        ],
        "cells": [
            {
                "occurrence_id": cell.occurrence_id,
                "time": cell.time,
                "truth": cell.truth,
                "proof_node": cell.proof_node,
            }
            for cell in certificate.cells
        ],
        "circuit": {
            "root": certificate.circuit.root,
            "nodes": [_node_to_dict(node) for node in certificate.circuit.nodes],
        },
        "decisions": [
            {
                "occurrence_id": item.occurrence_id,
                "time": item.time,
                "operator": item.operator,
                "truth": item.truth,
                "kind": item.kind,
                "alternatives": list(item.alternatives),
            }
            for item in certificate.decisions
        ],
    }


def canonical_json(certificate: SemanticCertificate, *, indent: int | None = None) -> str:
    return json.dumps(
        certificate_to_dict(certificate),
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        ensure_ascii=False,
        allow_nan=False,
    )


def certificate_digest(certificate: SemanticCertificate) -> str:
    return hashlib.sha256(canonical_json(certificate).encode("utf-8")).hexdigest()


def _node_to_dict(node: CircuitNode) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": node.id,
        "kind": node.kind.value,
        "children": list(node.children),
    }
    if node.kind is NodeKind.EVIDENCE:
        assert node.evidence is not None
        result["evidence"] = _fact_to_dict(node.evidence)
    return result


def _fact_to_dict(fact: EvidenceFact) -> dict[str, object]:
    return {
        "kind": fact.kind.value,
        "time": fact.time,
        "proposition": fact.proposition,
        "value": fact.value,
    }


def _json_ready(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [_json_ready(item) for item in sorted(value, key=repr)]
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return {"type": type(value).__qualname__, "repr": repr(value)}
