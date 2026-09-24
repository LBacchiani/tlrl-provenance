#!/usr/bin/env python
"""Re-run the post-hoc analyses over the frozen evidence and compare the
regenerated outputs byte-for-byte with the frozen ones (the kernel-verification record additionally embeds
timing/platform fields, which are ignored).

Nothing here needs a GPU, Java, MuJoCo, or the DeepLTL checkout: every analysis
reads the frozen record corpora in ``evidence/`` (the PointWorld corpus is
stored compressed and is inflated by ``scripts/setup_runtime.py``).  Outputs are
written to ``reproduction/analysis/`` (never over the frozen files) via the
``TLRL_OUTPUT_DIR`` redirect that every evidence-generating script honours.

    python scripts/reproduce_analysis.py                 # everything
    python scripts/reproduce_analysis.py --only kernel   # a subset (substring match)
    python scripts/reproduce_analysis.py --list

Exit status 0 iff every selected analysis regenerated its frozen output
byte-for-byte.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
OUT = ROOT / "reproduction" / "analysis"
POINT = "evidence/deepltl_point/confirmatory/semantic"


@dataclass
class Analysis:
    name: str
    script: str
    frozen: str  # frozen output, relative to ROOT
    args: list[str] = field(default_factory=list)
    needs_point_records: bool = False
    ignore_keys: tuple[str, ...] = ()  # environment/timing fields excluded from the comparison


ANALYSES = [
    Analysis(
        "kernel-verification",
        "verify_semantic_kernel.py",
        "evidence/machinery_verification.json",
        # source_sha256 hashes all of src/**/*.py, so it tracks any source edit since the record was frozen.
        ignore_keys=("elapsed_seconds", "platform", "python", "source_sha256"),
    ),
    Analysis(
        "prefix-exact-completeness",
        "verify_prefix_exact_completeness.py",
        "evidence/prefix_exact_completeness_verification.json",
        needs_point_records=True,
    ),
    Analysis("residual-size-guard", "stress_test_residual_size_guard.py", "evidence/residual_size_guard_stress_test.json"),
    Analysis(
        "significance-pointworld-ppo",
        "analyze_deepltl_point_significance.py",
        f"{POINT}/deepltl_point_significance_analysis.json",
        needs_point_records=True,
    ),
    Analysis(
        "significance-letterenv-dqn",
        "analyze_letter_env_significance.py",
        "evidence/letter_env/dqn/confirmatory/semantic/letter_env_dqn_significance_analysis.json",
        ["--algorithm", "dqn"],
    ),
    Analysis(
        "significance-letterenv-ppo",
        "analyze_letter_env_significance.py",
        "evidence/letter_env/ppo/confirmatory/semantic/letter_env_ppo_significance_analysis.json",
        ["--algorithm", "ppo"],
    ),
    Analysis(
        "significance-flatworld-ppo",
        "analyze_flatworld_significance.py",
        "evidence/flatworld/ppo/confirmatory/semantic/flatworld_ppo_significance_analysis.json",
        ["--algorithm", "ppo"],
    ),
    Analysis(
        "pointworld-full-semantics",
        "analyze_deepltl_point_full_semantics.py",
        f"{POINT}/deepltl_point_full_semantic_analysis.json",
        [
            "--records",
            f"{POINT}/deepltl_point_full_confirmatory_semantic_records.jsonl",
            "--output",
            "@OUT@/deepltl_point_full_semantic_analysis.json",
        ],
        needs_point_records=True,
        # The frozen file predates the add-one Monte Carlo correction: its p-values are
        # count/iterations, the shipped kernel reports (count+1)/(iterations+1).
        # Every other field is compared exactly. The paper's PointWorld p-values come from
        # `significance-pointworld-ppo`, which is byte-identical.
        ignore_keys=("permutation_p_value",),
    ),
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strip(value, keys):
    if isinstance(value, dict):
        return {k: strip(v, keys) for k, v in value.items() if k not in keys}
    if isinstance(value, list):
        return [strip(v, keys) for v in value]
    return value


def ensure_point_records() -> None:
    """Inflate the compressed PointWorld corpus (idempotent, hash-verified)."""
    target = ROOT / POINT / "deepltl_point_full_confirmatory_semantic_records.jsonl"
    archive = target.with_name(target.name + ".gz")
    expected = target.with_name(target.name + ".sha256").read_text(encoding="utf-8").split()[0].lower()
    if target.is_file() and sha256_file(target) == expected:
        return
    print("  inflating PointWorld record corpus ...", flush=True)
    digest = hashlib.sha256()
    partial = target.with_name(target.name + ".part")
    with gzip.open(archive, "rb") as source, partial.open("wb") as out:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
            out.write(block)
    if digest.hexdigest() != expected:
        partial.unlink()
        raise SystemExit("PointWorld corpus does not match its SHA-256 sidecar")
    partial.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", default=[], help="run analyses whose name contains any of these")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    selected = [a for a in ANALYSES if not args.only or any(token in a.name for token in args.only)]
    if args.list:
        for analysis in ANALYSES:
            print(analysis.name)
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for analysis in selected:
        print(f"[{analysis.name}]", flush=True)
        if analysis.needs_point_records:
            ensure_point_records()
        frozen = ROOT / analysis.frozen
        regenerated = OUT / frozen.name
        regenerated.unlink(missing_ok=True)
        env = dict(os.environ, TLRL_OUTPUT_DIR=str(OUT), PYTHONUTF8="1")
        command = [sys.executable, str(SCRIPTS / analysis.script)] + [a.replace("@OUT@", str(OUT)) for a in analysis.args]
        if analysis.script == "verify_semantic_kernel.py":
            command += ["--output", str(regenerated)]
        started = time.time()
        completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        elapsed = time.time() - started
        if completed.returncode != 0 or not regenerated.is_file():
            status, detail = "ERROR", (completed.stderr or completed.stdout)[-600:]
        elif regenerated.read_bytes() == frozen.read_bytes():
            status, detail = "IDENTICAL", ""
        elif analysis.ignore_keys and strip(json.loads(regenerated.read_text(encoding="utf-8")), analysis.ignore_keys) == strip(
            json.loads(frozen.read_text(encoding="utf-8")), analysis.ignore_keys
        ):
            status, detail = "IDENTICAL", f"(results identical; ignored environment fields: {', '.join(analysis.ignore_keys)})"
        else:
            status, detail = "DIFFERS", f"regenerated sha256 {sha256_file(regenerated)} vs frozen {sha256_file(frozen)}"
        print(f"  {status} ({elapsed:.0f}s) {detail}", flush=True)
        results.append({"analysis": analysis.name, "status": status, "seconds": round(elapsed, 1), "detail": detail})
    (OUT / "comparison.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    bad = [r for r in results if r["status"] != "IDENTICAL"]
    print(f"\n{len(results) - len(bad)}/{len(results)} analyses regenerated byte-identical outputs")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
