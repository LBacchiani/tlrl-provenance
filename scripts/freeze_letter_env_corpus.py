#!/usr/bin/env python
"""Freeze DeepLTL LetterEnv's published formula/world evaluation pairs.

This script reads no learned policy and executes no episode.  It freezes only
the upstream task corpus, the position-matched world layouts, and the source
files that establish their semantics.
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

from tlrl_benchmarks.letter_env import (  # noqa: E402
    ATOMS,
    EXPECTED_LETTER_COPIES,
    EXPECTED_OCCUPIED_CELLS,
    EXPECTED_TASK_COUNT,
    EXPECTED_TASKS_SHA256,
    EXPECTED_UPSTREAM_TASKS_SHA256,
    canonical_world_digest,
    load_tasks,
    load_world_layout,
    world_path,
)


UPSTREAM = ROOT / ".runtime" / "deep-ltl"
UPSTREAM_COMMIT = "3157200b5910cd7f5c493ffba8913994f2207eaf"
UPSTREAM_REPOSITORY = "https://github.com/mathiasj33/deep-ltl"
UPSTREAM_TASKS = UPSTREAM / "eval_datasets" / "LetterEnv-v0" / "tasks.txt"
PACKAGE = ROOT / "src" / "tlrl_benchmarks" / "letter_env"
EVIDENCE = ROOT / "evidence" / "letter_env" / "validation"
ALLOWED_TRACKED_PATCHES = {
    "src/ltl/automata/rabinizer.py",
    "src/utils/logging/file_logger.py",
}
SEMANTIC_SOURCE_FILES = (
    "src/envs/letter_world/__init__.py",
    "src/envs/letter_world/letter_env.py",
    "src/envs/env_utils.py",
    "src/envs/ldba_wrapper.py",
    "src/evaluation/create_eval_dataset.py",
    "src/evaluation/eval_sync_env.py",
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
        raise RuntimeError("upstream LetterEnv task-file byte hash changed")
    upstream_lf = normalized_lf(upstream_raw)
    packaged_lf = normalized_lf(packaged_raw)
    if upstream_lf != packaged_lf:
        raise RuntimeError("packaged LetterEnv formulas differ from the pinned upstream corpus")
    if sha256_bytes(packaged_lf) != EXPECTED_TASKS_SHA256:
        raise RuntimeError("normalized packaged LetterEnv task hash changed")

    tasks = load_tasks()
    world_directory = world_path(ROOT, 0).parent
    actual_world_names = sorted(path.name for path in world_directory.glob("world_info_*.pkl"))
    expected_world_names = [f"world_info_{index}.pkl" for index in range(EXPECTED_TASK_COUNT)]
    if set(actual_world_names) != set(expected_world_names) or len(actual_world_names) != len(
        expected_world_names
    ):
        raise RuntimeError("upstream LetterEnv world corpus is missing files or contains extras")

    pairs: list[dict[str, object]] = []
    for task in tasks:
        path = world_path(ROOT, task.world_index)
        layout = load_world_layout(ROOT, task.world_index)
        counts = {letter: sum(value == letter for value in layout.values()) for letter in sorted(ATOMS)}
        if set(counts.values()) != {EXPECTED_LETTER_COPIES}:
            raise AssertionError(f"invalid frozen world counts for {task.task_id}: {counts}")
        pairs.append(
            {
                "task_id": task.task_id,
                "index": task.world_index,
                "formula": task.source,
                "formula_source_sha256": sha256_bytes(task.source.encode("utf-8")),
                "formula_atoms": sorted(task.formula.atoms()),
                "world_path": str(path.relative_to(UPSTREAM)).replace("\\", "/"),
                "world_pickle_sha256": sha256(path),
                "world_canonical_sha256": canonical_world_digest(layout),
                "world_occupied_cells": len(layout),
                "world_letter_counts": counts,
            }
        )

    pair_identity = [
        {
            "index": pair["index"],
            "formula_source_sha256": pair["formula_source_sha256"],
            "world_pickle_sha256": pair["world_pickle_sha256"],
            "world_canonical_sha256": pair["world_canonical_sha256"],
        }
        for pair in pairs
    ]
    source_hashes = {
        path: sha256(UPSTREAM / path)
        for path in SEMANTIC_SOURCE_FILES
    }
    manifest = {
        "schema_version": "tlrl-letter-env-corpus-freeze/1",
        "status": "formula_world_corpus_frozen",
        "benchmark_id": "deepltl-letter-env-dqn",
        "scope": "published_formulas_worlds_and_grounding_sources_only",
        "outcomes_inspected": False,
        "training_executed": False,
        "dqn_implemented": False,
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
            "source_path": "eval_datasets/LetterEnv-v0/tasks.txt",
            "upstream_byte_sha256": sha256_bytes(upstream_raw),
            "normalized_lf_sha256": sha256_bytes(upstream_lf),
            "packaged_byte_sha256": sha256_bytes(packaged_raw),
            "atoms": sorted(ATOMS),
            "eventually_chain_count": sum(
                task.source.lstrip().startswith("F") for task in tasks
            ),
            "until_chain_count": sum(" U " in task.source for task in tasks),
            "selection": "all 50 published LetterEnv-v0 evaluation formulas in source order",
        },
        "world_corpus": {
            "count": len(expected_world_names),
            "directory": "eval_datasets/LetterEnv-v0/worlds",
            "grid_size": 7,
            "occupied_cells_per_world": EXPECTED_OCCUPIED_CELLS,
            "copies_per_letter": EXPECTED_LETTER_COPIES,
            "selection": "the position-matched world_info_N.pkl for every formula line N",
        },
        "pairing": {
            "rule": "formula line N is paired with world_info_N.pkl",
            "construction_source": "src/evaluation/create_eval_dataset.py writes both artifacts in loop iteration N",
            "evaluation_source": "src/evaluation/eval_sync_env.py loads tasks and world paths by the same current index",
            "paired_identity_sha256": sha256_bytes(
                json.dumps(pair_identity, separators=(",", ":"), sort_keys=True).encode("utf-8")
            ),
        },
        "pairs": pairs,
        "contamination_controls": {
            "v2_dependency": False,
            "deepltl_point_policy_dependency": False,
            "learned_policy_outcomes_inspected": False,
            "training_authorized_by_this_artifact": False,
        },
    }
    payload = canonical_json(manifest)

    package_output = PACKAGE / "corpus_manifest.json"
    evidence_output = EVIDENCE / "letter_env_corpus_freeze.json"
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
                "world_count": len(expected_world_names),
                "pair_digest": manifest["pairing"]["paired_identity_sha256"],
                "manifest_sha256": sha256_bytes(payload),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
