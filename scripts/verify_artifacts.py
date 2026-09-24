#!/usr/bin/env python
"""Verify the integrity of every frozen artifact shipped with the repository.

Two independent checks, both fail-closed:

* every ``evidence/**/*.sha256`` sidecar must match the file it names (a
  ``*.jsonl`` corpus stored as ``*.jsonl.gz`` is verified by hashing its
  decompressed stream);
* every file listed in ``experiments/CHECKSUMS.sha256`` (all trained
  checkpoints, logs, and manifests) must exist with the recorded SHA-256, and
  no unlisted file may sit in ``experiments/``.

Usage:

    python scripts/verify_artifacts.py
    python scripts/verify_artifacts.py --write-checkpoint-manifest   # maintainers only
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
EXPERIMENTS = ROOT / "experiments"
MANIFEST = EXPERIMENTS / "CHECKSUMS.sha256"


def sha256_stream(handle) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1 << 20), b""):
        digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return sha256_stream(handle)


def verify_evidence() -> tuple[int, list[str]]:
    failures: list[str] = []
    checked = 0
    for sidecar in sorted(EVIDENCE.rglob("*.sha256")):
        expected = sidecar.read_text(encoding="utf-8").split()[0].lower()
        target = sidecar.with_suffix("")
        compressed = target.with_name(target.name + ".gz")
        if target.is_file():
            actual = sha256_file(target)
        elif compressed.is_file():
            with gzip.open(compressed, "rb") as handle:
                actual = sha256_stream(handle)
        else:
            failures.append(f"missing artifact for {sidecar.relative_to(ROOT)}")
            continue
        checked += 1
        if actual != expected:
            failures.append(f"SHA-256 mismatch: {target.relative_to(ROOT)}")
    return checked, failures


def checkpoint_files() -> list[Path]:
    return sorted(p for p in EXPERIMENTS.rglob("*") if p.is_file() and p != MANIFEST and p.name != "README.md")


def write_manifest() -> None:
    lines = [f"{sha256_file(p)}  {p.relative_to(EXPERIMENTS).as_posix()}" for p in checkpoint_files()]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {MANIFEST.relative_to(ROOT)} ({len(lines)} files)")


def verify_checkpoints() -> tuple[int, list[str]]:
    if not MANIFEST.is_file():
        return 0, [f"missing {MANIFEST.relative_to(ROOT)}"]
    failures: list[str] = []
    listed: set[str] = set()
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        listed.add(name)
        path = EXPERIMENTS / name
        if not path.is_file():
            failures.append(f"missing checkpoint file: experiments/{name}")
        elif sha256_file(path) != expected:
            failures.append(f"SHA-256 mismatch: experiments/{name}")
    for path in checkpoint_files():
        if path.relative_to(EXPERIMENTS).as_posix() not in listed:
            failures.append(f"unlisted file in experiments/: {path.relative_to(EXPERIMENTS).as_posix()}")
    return len(listed), failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write-checkpoint-manifest", action="store_true")
    args = parser.parse_args()
    if args.write_checkpoint_manifest:
        write_manifest()
        return 0
    evidence_checked, failures = verify_evidence()
    print(f"evidence sidecars verified: {evidence_checked}")
    checkpoints_checked, checkpoint_failures = verify_checkpoints()
    print(f"checkpoint files verified: {checkpoints_checked}")
    failures += checkpoint_failures
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    print("ALL ARTIFACTS VERIFIED" if not failures else f"{len(failures)} FAILURE(S)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
