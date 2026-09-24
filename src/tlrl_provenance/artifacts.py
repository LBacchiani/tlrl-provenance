"""Atomic certificate artifacts with replay verification and SHA-256 identity."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

from .evaluator import SemanticCertificate, evaluate_with_provenance
from .logic import parse_formula
from .serialization import SCHEMA_VERSION, canonical_json, certificate_to_dict
from .trace import GroundingRecord, LabeledTrace


def write_certificate_artifact(path: str | Path, certificate: SemanticCertificate) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(certificate, indent=2) + "\n"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    _atomic_text(path, payload)
    _atomic_text(path.with_suffix(path.suffix + ".sha256"), f"{digest}  {path.name}\n")
    return digest


def replay_certificate_artifact(path: str | Path) -> SemanticCertificate:
    """Recompute serialized label/grounding semantics and reject mismatches.

    Certificate artifacts intentionally do not embed raw states,
    observations, actions, rewards, or arbitrary trace metadata. Full
    raw-state proposition replay therefore requires the retained source trace
    plus ``verify_grounding_reconstruction``; this function verifies only the
    self-contained semantic and grounding-record layer stored in the artifact.
    """

    path = Path(path)
    payload = path.read_text(encoding="utf-8")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.exists():
        expected = sidecar.read_text(encoding="utf-8").split()[0]
        actual = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if expected != actual:
            raise ValueError("certificate artifact digest mismatch")
    data = json.loads(payload)
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported certificate schema: {data.get('schema_version')!r}")
    grounding = data["grounding"]
    records = tuple(
        GroundingRecord(
            proposition=item["proposition"],
            time=item["time"],
            value=item["value"],
            rule_id=item["rule_id"],
            rule_version=item["rule_version"],
            raw_fields=tuple(item["raw_fields"]),
            source_digest=item["source_digest"],
            metadata=item["metadata"],
        )
        for item in grounding["records"]
    )
    trace_data = data["trace"]
    trace = LabeledTrace(
        labels=trace_data["labels"],
        groundings=records,
        policy_id=trace_data["policy_id"],
        environment_id=trace_data["environment_id"],
        seed=trace_data["seed"],
        terminated=trace_data["terminated"],
        truncated=trace_data["truncated"],
    )
    certificate = evaluate_with_provenance(parse_formula(data["formula"]), trace)
    if certificate_to_dict(certificate) != data:
        raise ValueError("certificate replay does not reproduce the serialized semantics")
    return certificate


def _atomic_text(path: Path, payload: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
