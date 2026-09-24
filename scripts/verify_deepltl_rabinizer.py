#!/usr/bin/env python
"""Verify Java/Rabinizer and upstream HOA parsing over all frozen tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tlrl_benchmarks.deepltl_point import ATOMS, load_tasks  # noqa: E402


UPSTREAM_COMMIT = "3157200b5910cd7f5c493ffba8913994f2207eaf"
RABINIZER_SHA256 = "46efc53d769d1016c75b6b76f7cb2db259d4426053591993a62aa874f3ef486d"
RABINIZER_MAIN = "owl.translations.ltl2ldba.LTL2LDBAModule"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--jar", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evidence" / "deepltl_point" / "validation" / "deepltl_point_rabinizer_verification.json",
    )
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    jar = args.jar.resolve()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=upstream, text=True
    ).strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"upstream commit mismatch: {commit}")
    jar_digest = sha256(jar)
    if jar_digest != RABINIZER_SHA256:
        raise RuntimeError(f"Rabinizer jar SHA-256 mismatch: {jar_digest}")

    sys.path.insert(0, str(upstream / "src"))
    from ltl.hoa import HOAParser  # noqa: E402

    java_version = subprocess.run(
        ["java", "-version"], capture_output=True, text=True, check=True
    ).stderr.strip()
    records = []
    started = time.perf_counter()
    for task in load_tasks():
        command = [
            "java",
            "-classpath",
            str(jar),
            RABINIZER_MAIN,
            "-i",
            task.source,
            "-p",
            "-e",
        ]
        run = subprocess.run(
            command, capture_output=True, text=True, timeout=120, check=False
        )
        if run.returncode != 0:
            raise RuntimeError(
                f"{task.task_id} conversion failed ({run.returncode}): {run.stderr}"
            )
        if run.stderr:
            raise RuntimeError(f"{task.task_id} wrote stderr: {run.stderr}")
        if not run.stdout.startswith("HOA: v1"):
            raise RuntimeError(f"{task.task_id} returned non-HOA output")
        if "acc-name: Buchi" not in run.stdout or "Acceptance: 1 Inf(0)" not in run.stdout:
            raise RuntimeError(f"{task.task_id} did not produce the required Buchi acceptance")
        ldba = HOAParser(
            task.source,
            run.stdout,
            set(ATOMS),
            simplify_labels=False,
        ).parse_hoa()
        if not ldba.check_valid():
            raise RuntimeError(f"{task.task_id} produced an invalid LDBA")
        records.append(
            {
                "task_id": task.task_id,
                "formula_sha256": hashlib.sha256(task.source.encode("utf-8")).hexdigest(),
                "hoa_sha256": hashlib.sha256(run.stdout.encode("utf-8")).hexdigest(),
                "states": ldba.num_states,
                "transitions": ldba.num_transitions,
            }
        )
    result = {
        "schema_version": "tlrl-deepltl-rabinizer-verification/1",
        "status": "passed",
        "scope": "development_only_formula_to_ldba_compatibility",
        "upstream_commit": commit,
        "rabinizer_jar_sha256": jar_digest,
        "java_version": java_version,
        "python_version": platform.python_version(),
        "rabinizer_flags": ["-p", "-e"],
        "obsolete_flag_omitted": "-d",
        "task_count": len(records),
        "all_upstream_ldba_checks_valid": True,
        "elapsed_seconds": time.perf_counter() - started,
        "tasks": records,
    }
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write(args.output, payload)
    atomic_write(
        args.output.with_suffix(args.output.suffix + ".sha256"),
        f"{hashlib.sha256(payload).hexdigest()}  {args.output.name}\n".encode("ascii"),
    )
    print(json.dumps({key: result[key] for key in result if key != "tasks"}, sort_keys=True))


if __name__ == "__main__":
    main()
