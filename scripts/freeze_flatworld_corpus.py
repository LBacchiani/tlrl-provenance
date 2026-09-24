#!/usr/bin/env python
"""Freeze DeepLTL FlatWorld's published formula corpus and circle geometry.

This script reads no learned policy and executes no episode. Unlike
LetterEnv, FlatWorld has no per-formula paired world file to freeze: its nine
colored circles are a fixed class attribute (``FlatWorld.CIRCLES``) baked
into the upstream source, so what needs freezing is the formula corpus, that
fixed geometry (as a source hash, plus the transcribed literal values already
embedded in ``flatworld/adapter.py``), and the source files that establish
the environment's semantics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tlrl_benchmarks.flatworld import (  # noqa: E402
    ATOMS,
    CIRCLES,
    EXPECTED_TASK_COUNT,
    EXPECTED_TASKS_SHA256,
    EXPECTED_UPSTREAM_TASKS_SHA256,
    POSITION_BOUND,
    load_tasks,
)


UPSTREAM = ROOT / ".runtime" / "deep-ltl"
UPSTREAM_COMMIT = "3157200b5910cd7f5c493ffba8913994f2207eaf"
UPSTREAM_REPOSITORY = "https://github.com/mathiasj33/deep-ltl"
UPSTREAM_TASKS = UPSTREAM / "eval_datasets" / "FlatWorld-v0" / "tasks.txt"
PACKAGE = ROOT / "src" / "tlrl_benchmarks" / "flatworld"
EVIDENCE = ROOT / "evidence" / "flatworld" / "validation"
ALLOWED_TRACKED_PATCHES = {
    "src/ltl/automata/rabinizer.py",
    "src/utils/logging/file_logger.py",
}
SEMANTIC_SOURCE_FILES = (
    "src/envs/flatworld/__init__.py",
    "src/envs/flatworld/flatworld.py",
    "src/envs/env_utils.py",
    "src/envs/ldba_wrapper.py",
    "src/sequence/samplers/curriculum.py",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
    temporary.replace(path)


def git(*arguments: str, binary: bool = False) -> str | bytes:
    result = subprocess.check_output(["git", "-C", str(UPSTREAM), *arguments])
    return result if binary else result.decode("utf-8").strip()


def normalized_lf(payload: bytes) -> bytes:
    result = payload.replace(b"\r\n", b"\n")
    if b"\r" in result:
        raise ValueError("formula corpus contains unsupported carriage returns")
    return result


def main() -> None:
    if not UPSTREAM.is_dir():
        raise RuntimeError(f"pinned DeepLTL checkout is missing: {UPSTREAM}")
    commit = git("rev-parse", "HEAD")
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"upstream commit mismatch: {commit}")

    modified = set(filter(None, str(git("diff", "--name-only", "HEAD")).splitlines()))
    unexpected = modified - ALLOWED_TRACKED_PATCHES
    if unexpected:
        raise RuntimeError(f"unexpected tracked upstream modifications: {sorted(unexpected)}")
    patch_payload = git("diff", "--binary", "HEAD", binary=True)

    upstream_raw = UPSTREAM_TASKS.read_bytes()
    packaged_raw = (PACKAGE / "tasks.txt").read_bytes()
    if sha256_bytes(upstream_raw) != EXPECTED_UPSTREAM_TASKS_SHA256:
        raise RuntimeError("upstream FlatWorld task-file byte hash changed")
    upstream_lf = normalized_lf(upstream_raw)
    packaged_lf = normalized_lf(packaged_raw)
    if upstream_lf != packaged_lf:
        raise RuntimeError("packaged FlatWorld formulas differ from the pinned upstream corpus")
    if sha256_bytes(packaged_lf) != EXPECTED_TASKS_SHA256:
        raise RuntimeError("normalized packaged FlatWorld task hash changed")

    tasks = load_tasks()

    # Independently re-import the live upstream FlatWorld class and compare
    # its CIRCLES attribute against this package's frozen, transcribed copy,
    # so drift between the two is caught here rather than silently trusted.
    upstream_src = UPSTREAM / "src"
    if str(upstream_src) not in sys.path:
        sys.path.insert(0, str(upstream_src))
    from envs.flatworld.flatworld import FlatWorld as _UpstreamFlatWorld

    live_env = _UpstreamFlatWorld(continuous_actions=False)
    live_circles = tuple(
        (circle.color, float(circle.center[0]), float(circle.center[1]), float(circle.radius))
        for circle in _UpstreamFlatWorld.CIRCLES
    )
    if live_circles != CIRCLES:
        raise RuntimeError(
            "frozen FlatWorld CIRCLES constant disagrees with the live upstream class attribute: "
            f"frozen={CIRCLES}, live={live_circles}"
        )
    live_propositions = sorted(live_env.get_propositions())
    if live_propositions != sorted(ATOMS):
        raise RuntimeError(f"live FlatWorld propositions {live_propositions} disagree with ATOMS {sorted(ATOMS)}")
    live_assignment_count = len(live_env.get_possible_assignments())

    formula_pairs = [
        {
            "task_id": task.task_id,
            "formula": task.source,
            "formula_source_sha256": sha256_bytes(task.source.encode("utf-8")),
            "formula_atoms": sorted(task.formula.atoms()),
        }
        for task in tasks
    ]
    source_hashes = {
        path: sha256(UPSTREAM / path)
        for path in SEMANTIC_SOURCE_FILES
    }
    manifest = {
        "schema_version": "tlrl-flatworld-corpus-freeze/1",
        "status": "formula_corpus_and_geometry_frozen",
        "benchmark_id": "deepltl-flatworld",
        "scope": "published_formulas_fixed_circle_geometry_and_grounding_sources_only",
        "outcomes_inspected": False,
        "training_executed": False,
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": commit,
            "tracked_patch_paths": sorted(modified),
            "tracked_patch_sha256": sha256_bytes(patch_payload),
            "patch_scope": {
                "src/ltl/automata/rabinizer.py": "native Rabinizer 4 jar invocation",
                "src/utils/logging/file_logger.py": "Windows-compatible exclusive file locking",
            },
            "semantic_source_sha256": source_hashes,
        },
        "formula_corpus": {
            "count": len(tasks),
            "source_path": "eval_datasets/FlatWorld-v0/tasks.txt",
            "upstream_byte_sha256": sha256_bytes(upstream_raw),
            "normalized_lf_sha256": sha256_bytes(upstream_lf),
            "packaged_byte_sha256": sha256_bytes(packaged_raw),
            "atoms": sorted(ATOMS),
            "eventually_chain_count": sum(
                task.source.lstrip().startswith("F") for task in tasks
            ),
            "until_chain_count": sum(" U " in task.source for task in tasks),
            "selection": "all 49 published FlatWorld-v0 evaluation formulas in source order",
        },
        "formula_pairs": formula_pairs,
        "circle_geometry": {
            "count": len(CIRCLES),
            "colors": sorted(ATOMS),
            "circles": [
                {"color": color, "center_x": cx, "center_y": cy, "radius": radius}
                for color, cx, cy, radius in CIRCLES
            ],
            "position_bound": POSITION_BOUND,
            "possible_assignment_count": live_assignment_count,
            "matches_live_upstream_class_attribute": True,
            "note": (
                "unlike LetterEnv's per-episode randomized letter map or "
                "PointLtl2's disjoint-zone invariant, FlatWorld's circles are "
                "a fixed constant; different-color circles are permitted to "
                "overlap by upstream design, so more than one proposition "
                "can be simultaneously true"
            ),
        },
        "contamination_controls": {
            "v2_dependency": False,
            "deepltl_point_policy_dependency": False,
            "letter_env_policy_dependency": False,
            "learned_policy_outcomes_inspected": False,
            "training_authorized_by_this_artifact": False,
        },
    }
    payload = canonical_json(manifest)

    package_output = PACKAGE / "corpus_manifest.json"
    evidence_output = EVIDENCE / "flatworld_corpus_freeze.json"
    for output in (package_output, evidence_output):
        atomic_write(output, payload)
        atomic_write(
            output.with_suffix(output.suffix + ".sha256"),
            f"{sha256_bytes(payload)}  {output.name}\n".encode("ascii"),
        )

    print(
        json.dumps(
            {
                "status": manifest["status"],
                "formula_count": len(tasks),
                "circle_count": len(CIRCLES),
                "manifest_sha256": sha256_bytes(payload),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
