#!/usr/bin/env python
"""Regenerate the deterministic DeepLTL PointWorld design manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tlrl_benchmarks.deepltl_point import (  # noqa: E402
    ATOMS,
    ENVIRONMENT_ID,
    EXPECTED_TASKS_SHA256,
    EXPECTED_UPSTREAM_TASKS_SHA256,
    load_tasks,
)
from tlrl_benchmarks.deepltl_point import adapter  # noqa: E402
from tlrl_provenance import ENGINE_VERSION, SEMANTICS_VERSION, engine_source_digest  # noqa: E402


PACKAGE = ROOT / "src" / "tlrl_benchmarks" / "deepltl_point"
EVIDENCE = ROOT / "evidence" / "deepltl_point" / "validation"


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
    tasks = load_tasks()
    eventual = sum(task.source.lstrip().startswith("F") for task in tasks)
    until = len(tasks) - eventual
    smoke_path = EVIDENCE / "deepltl_point_environment_smoke.json"
    smoke = None
    if smoke_path.exists():
        smoke_payload = json.loads(smoke_path.read_text(encoding="utf-8"))
        if smoke_payload.get("status") != "passed":
            raise RuntimeError("environment smoke evidence is not passing")
        smoke = {
            "path": "evidence/deepltl_point/validation/deepltl_point_environment_smoke.json",
            "sha256": sha256(smoke_path),
            "checked_steps": smoke_payload["checked_steps"],
            "geometry_info_mismatches": smoke_payload["geometry_info_mismatches"],
            "positive_contact_probes": smoke_payload["positive_contact_probes"],
            "runtime": smoke_payload["runtime"],
        }
    rabinizer_path = EVIDENCE / "deepltl_point_rabinizer_verification.json"
    if not rabinizer_path.exists():
        raise RuntimeError("Rabinizer verification evidence is missing")
    rabinizer_payload = json.loads(rabinizer_path.read_text(encoding="utf-8"))
    if rabinizer_payload.get("status") != "passed" or rabinizer_payload.get("task_count") != 50:
        raise RuntimeError("Rabinizer verification evidence is not passing for all 50 tasks")
    rabinizer = {
        "path": "evidence/deepltl_point/validation/deepltl_point_rabinizer_verification.json",
        "sha256": sha256(rabinizer_path),
        "java_version": rabinizer_payload["java_version"],
        "rabinizer_jar_sha256": rabinizer_payload["rabinizer_jar_sha256"],
        "flags": rabinizer_payload["rabinizer_flags"],
        "obsolete_flag_omitted": rabinizer_payload["obsolete_flag_omitted"],
        "task_count": rabinizer_payload["task_count"],
        "all_upstream_ldba_checks_valid": rabinizer_payload[
            "all_upstream_ldba_checks_valid"
        ],
    }
    ldba_smoke_path = EVIDENCE / "deepltl_point_official_ldba_env_smoke.json"
    if not ldba_smoke_path.exists():
        raise RuntimeError("official LDBA environment smoke evidence is missing")
    ldba_smoke_payload = json.loads(ldba_smoke_path.read_text(encoding="utf-8"))
    if ldba_smoke_payload.get("status") != "passed":
        raise RuntimeError("official LDBA environment smoke evidence is not passing")
    ldba_smoke = {
        "path": "evidence/deepltl_point/validation/deepltl_point_official_ldba_env_smoke.json",
        "sha256": sha256(ldba_smoke_path),
        "checked_steps": ldba_smoke_payload["checked_steps"],
        "geometry_info_mismatches": ldba_smoke_payload["geometry_info_mismatches"],
        "ldba_valid": ldba_smoke_payload["ldba_valid"],
        "forced_ordered_eventuality_probe": ldba_smoke_payload[
            "forced_ordered_eventuality_probe"
        ],
        "termination_truncation_preserved": ldba_smoke_payload[
            "termination_truncation_preserved"
        ],
        "runtime": ldba_smoke_payload["runtime"],
    }
    ppo_smoke_path = EVIDENCE / "deepltl_point_ppo_gpu_smoke.json"
    if not ppo_smoke_path.exists():
        raise RuntimeError("PPO GPU smoke evidence is missing")
    ppo_smoke_payload = json.loads(ppo_smoke_path.read_text(encoding="utf-8"))
    if ppo_smoke_payload.get("status") != "passed":
        raise RuntimeError("PPO GPU smoke evidence is not passing")
    ppo_smoke = {
        "path": "evidence/deepltl_point/validation/deepltl_point_ppo_gpu_smoke.json",
        "sha256": sha256(ppo_smoke_path),
        "algorithm": ppo_smoke_payload["algorithm"],
        "num_steps": ppo_smoke_payload["num_steps"],
        "num_updates": ppo_smoke_payload["num_updates"],
        "checkpoint_complete": ppo_smoke_payload["checkpoint_complete"],
        "runtime": ppo_smoke_payload["runtime"],
        "methodology_outcomes_inspected": ppo_smoke_payload[
            "methodology_outcomes_inspected"
        ],
    }
    manifest = {
        "schema_version": "tlrl-benchmark-design/1",
        "benchmark_id": "deepltl-pointworld-official-eval-50",
        "status": "runtime_qualified_for_development_training_not_confirmatory_utility",
        "upstream": {
            "repository": "https://github.com/mathiasj33/deep-ltl",
            "commit": "3157200b5910cd7f5c493ffba8913994f2207eaf",
            "environment_id": ENVIRONMENT_ID,
            "task_source": "eval_datasets/PointLtl2-v0/tasks.txt",
            "task_source_url": "https://raw.githubusercontent.com/mathiasj33/deep-ltl/main/eval_datasets/PointLtl2-v0/tasks.txt",
            "task_source_raw_sha256": EXPECTED_UPSTREAM_TASKS_SHA256,
            "license": {
                "deep_ltl": "MIT",
                "vendored_safety_gymnasium_fork": "Apache-2.0",
            },
        },
        "formula_corpus": {
            "count": len(tasks),
            "normalized_lf_sha256": EXPECTED_TASKS_SHA256,
            "packaged_file_sha256": sha256(PACKAGE / "tasks.txt"),
            "atoms": sorted(ATOMS),
            "families": {
                "ordered_eventuality": eventual,
                "avoidance_until": until,
            },
            "selection": "all tasks in the upstream official evaluation dataset",
        },
        "grounding": {
            "rule": "color is true iff Euclidean agent-to-zone-center distance <= zone radius",
            "zone_radius": 0.4,
            "zones_per_color": 2,
            "label_timing": "post_action_before_ldba_transition",
            "official_output_cross_check": "required at every retained step",
            "adapter_sha256": sha256(Path(adapter.__file__).resolve()),
            "real_environment_smoke": smoke,
        },
        "outcomes": {
            "official_success": "decided satisfaction, independently checked",
            "official_violation": "decided violation, independently checked",
            "time_limit_truncation": "censored; never silently counted as violation",
            "collection_api": "bypass only upstream RemoveTruncWrapper and retain TimeLimit's terminated/truncated booleans",
            "other_environment_termination": "separate non-semantic outcome",
        },
        "official_subject": {
            "algorithm": "formula-conditioned recurrent PPO",
            "steps": 15000000,
            "num_processes": 16,
            "steps_per_process": 4096,
            "batch_size": 2048,
            "learning_rate": 0.0003,
            "discount": 0.998,
            "entropy_coefficient": 0.003,
            "epochs": 10,
        },
        "official_runtime": {
            "rabinizer_verification": rabinizer,
            "official_ldba_environment_smoke": ldba_smoke,
            "compatibility_patch": {
                "path": "benchmarks/deep_ltl_runtime/patches/deepltl-rabinizer-java25.patch",
                "sha256": sha256(
                    ROOT
                    / "benchmarks"
                    / "deep_ltl_runtime"
                    / "patches"
                    / "deepltl-rabinizer-java25.patch"
                ),
                "scope": "process_boundary_only",
                "shared_with": ["letter_env"],
            },
            "windows_file_lock_patch": {
                "path": "benchmarks/deep_ltl_runtime/patches/deepltl-windows-file-lock.patch",
                "sha256": sha256(
                    ROOT
                    / "benchmarks"
                    / "deep_ltl_runtime"
                    / "patches"
                    / "deepltl-windows-file-lock.patch"
                ),
                "scope": "cross_platform_logging_only",
                "shared_with": ["letter_env"],
            },
            "ppo_gpu_smoke": ppo_smoke,
        },
        "semantic_engine": {
            "version": ENGINE_VERSION,
            "semantics_version": SEMANTICS_VERSION,
            "source_sha256": engine_source_digest(),
        },
        "contamination_controls": {
            "v2_dependency": False,
            "utility_outcomes_inspected": False,
            "development_training_authorized": True,
            "confirmatory_usefulness_evaluation_authorized": False,
        },
        "open_gates": [
            "freeze the post-smoke PPO development-pilot budget and train independent seeds",
            "freeze any second deep-RL algorithm as a separately justified comparison subject",
            "freeze audit/held-out splits and usefulness endpoint before inspecting outcomes",
        ],
    }
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output = PACKAGE / "manifest.json"
    atomic_write(output, payload)
    sidecar = f"{hashlib.sha256(payload).hexdigest()}  manifest.json\n".encode("ascii")
    atomic_write(PACKAGE / "manifest.json.sha256", sidecar)


if __name__ == "__main__":
    main()
