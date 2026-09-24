"""Compile the retained Coq proof and write a hash-anchored verification record."""

from __future__ import annotations

import os
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PROOF = ROOT / "formal" / "coq" / "ProvenanceEvidence.v"
OUTPUT = ROOT / "evidence" / "provenance_evidence_coq_verification.json"
if os.environ.get("TLRL_OUTPUT_DIR"):  # redirect generated evidence; frozen files stay untouched
    OUTPUT = Path(os.environ["TLRL_OUTPUT_DIR"]) / OUTPUT.name


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coqc",
        default=shutil.which("coqc"),
        help="path to coqc (defaults to the executable on PATH)",
    )
    args = parser.parse_args()
    if not args.coqc:
        raise SystemExit("coqc was not found; pass --coqc PATH")

    executable = Path(args.coqc).resolve()
    version = subprocess.run(
        [str(executable), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    completed = subprocess.run(
        [str(executable), "-q", PROOF.name],
        cwd=PROOF.parent,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.stdout + completed.stderr)

    record = {
        "schema_version": "tlrl-provenance-coq-verification/1",
        "purpose": (
            "Machine-checks represented-support sufficiency, exact necessary/possible "
            "evidence extraction, and the necessary-subset-possible corollary for the "
            "paper's unfolded provenance-circuit model. This is a proof of the "
            "mathematical model, not extraction or formal verification of the Python "
            "implementation."
        ),
        "compiler": version,
        "command": "coqc -q ProvenanceEvidence.v",
        "source": "formal/coq/ProvenanceEvidence.v",
        "source_sha256": sha256(PROOF),
        "checked_results": [
            "represented_support_sound",
            "necessary_exact",
            "possible_exact",
            "necessary_is_possible",
        ],
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    payload = (json.dumps(record, indent=2) + "\n").encode("utf-8")
    OUTPUT.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    OUTPUT.with_suffix(OUTPUT.suffix + ".sha256").write_text(
        f"{digest}  {OUTPUT.name}\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"verified {PROOF.relative_to(ROOT)} with {version.splitlines()[0]}")
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({digest})")


if __name__ == "__main__":
    main()
