#!/usr/bin/env python
"""Re-run a frozen confirmatory audit campaign from the shipped checkpoints and
compare it, record by record, with the frozen results in ``evidence/``.

The frozen campaign configuration is reused unchanged (same subjects, tasks,
seed namespaces, scenario pairing, episode counts) except that all outputs are
redirected to a scratch directory (default ``reproduction/<campaign>/``);
the frozen evidence is never modified.

Two modes:

    # Quick check: re-run an evenly spread sample of shards (each shard is
    # 5 formulas x 20 episodes for one policy) and compare exactly those records.
    python scripts/reproduce_audit.py letter_env_ppo --max-shards 4

    # Full reproduction of the entire campaign (thousands of episodes).
    python scripts/reproduce_audit.py letter_env_ppo --full

Campaigns: deepltl_point, letter_env_dqn, letter_env_ppo, flatworld_ppo.

Prerequisites: ``python scripts/setup_runtime.py`` (Rabinizer, DeepLTL, checkpoints).

Compared fields are the semantic/behavioural outcome of each episode (length,
termination, official flags, independent verdict, disposition, return,
evidence counts, atom counts, scenario identity).  Engine/plan digests are
deliberately *not* compared: frozen plans record the engine source digest at
freeze time, and the kernel shipped here is the final revision (see README).
The exit status is 0 iff every compared record agrees on every compared field.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tlrl_provenance.campaign import (  # noqa: E402
    build_plan,
    canonical_json,
    freeze_plan,
    load_adapter,
    load_campaign_config,
    read_jsonl,
    record_key,
    task_shards,
)

CAMPAIGNS: dict[str, tuple[str, str]] = {
    "deepltl_point": (
        "evidence/deepltl_point/confirmatory/semantic/deepltl_point_full_confirmatory_semantic_campaign.json",
        "evidence/deepltl_point/confirmatory/semantic/deepltl_point_full_confirmatory_semantic_records.jsonl",
    ),
    "letter_env_dqn": (
        "evidence/letter_env/dqn/confirmatory/semantic/letter_env_dqn_confirmatory_campaign_config.json",
        "evidence/letter_env/dqn/confirmatory/semantic/letter_env_dqn_full_confirmatory_semantic_records.jsonl",
    ),
    "letter_env_ppo": (
        "evidence/letter_env/ppo/confirmatory/semantic/letter_env_ppo_confirmatory_campaign_config.json",
        "evidence/letter_env/ppo/confirmatory/semantic/letter_env_ppo_full_confirmatory_semantic_records.jsonl",
    ),
    "flatworld_ppo": (
        "evidence/flatworld/ppo/confirmatory/semantic/flatworld_ppo_confirmatory_campaign_config.json",
        "evidence/flatworld/ppo/confirmatory/semantic/flatworld_ppo_full_confirmatory_semantic_records.jsonl",
    ),
}

COMPARED_FIELDS = (
    "steps",
    "terminated",
    "truncated",
    "official_success",
    "official_violation",
    "independent_verdict",
    "disposition",
    "failure_class",
    "official_independent_mismatch",
    "possible_evidence_count",
    "necessary_evidence_count",
    "atom_counts",
    "return_sum",
    "scenario_seed",
    "start_position",
    "generated_map_canonical_sha256",
)


def read_frozen_records(path: Path) -> list[dict[str, Any]]:
    if path.is_file():
        return read_jsonl(path)
    compressed = path.with_name(path.name + ".gz")
    if not compressed.is_file():
        raise FileNotFoundError(f"frozen records not found: {path} (run scripts/setup_runtime.py)")
    with gzip.open(compressed, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        try:
            return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return left == right


def compare(frozen: dict[tuple, dict[str, Any]], fresh: list[dict[str, Any]]) -> dict[str, Any]:
    mismatched_fields: dict[str, int] = {}
    mismatched_records = 0
    missing = 0
    examples: list[dict[str, Any]] = []
    for record in fresh:
        reference = frozen.get(record_key(record))
        if reference is None:
            missing += 1
            continue
        bad = [
            field
            for field in COMPARED_FIELDS
            if field in reference or field in record
            if not values_equal(reference.get(field), record.get(field))
        ]
        if bad:
            mismatched_records += 1
            for field in bad:
                mismatched_fields[field] = mismatched_fields.get(field, 0) + 1
            if len(examples) < 5:
                examples.append(
                    {
                        "key": list(record_key(record)),
                        "fields": {f: {"frozen": reference.get(f), "reproduced": record.get(f)} for f in bad},
                    }
                )
    return {
        "records_compared": len(fresh) - missing,
        "records_not_in_frozen_corpus": missing,
        "records_with_any_difference": mismatched_records,
        "differences_by_field": mismatched_fields,
        "examples": examples,
    }


def evenly_spread(items: list[Any], count: int) -> list[Any]:
    if count >= len(items):
        return list(items)
    step = len(items) / count
    return [items[int(index * step)] for index in range(count)]


def run_shards(config_path: Path, shard_ids: list[str], workers: int) -> None:
    runner = SCRIPTS / "run_confirmatory_semantic_audit.py"
    pending = list(shard_ids)
    active: list[tuple[str, subprocess.Popen]] = []
    finished = 0
    while pending or active:
        while pending and len(active) < workers:
            shard_id = pending.pop(0)
            process = subprocess.Popen(
                [sys.executable, str(runner), "--config", str(config_path), "--worker", "--shard-id", shard_id],
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            active.append((shard_id, process))
        time.sleep(0.5)
        still: list[tuple[str, subprocess.Popen]] = []
        for shard_id, process in active:
            code = process.poll()
            if code is None:
                still.append((shard_id, process))
                continue
            if code != 0:
                raise RuntimeError(f"shard {shard_id} failed ({code}):\n{process.stderr.read()}")
            finished += 1
            print(f"  shard {shard_id} done ({finished}/{len(shard_ids)})", flush=True)
        active = still


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("campaign", choices=sorted(CAMPAIGNS))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="re-run the entire campaign")
    mode.add_argument("--max-shards", type=int, default=2, help="quick mode: number of shards to re-run (default 2)")
    parser.add_argument("--workers", type=int, default=None, help="parallel worker processes")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config_rel, records_rel = CAMPAIGNS[args.campaign]
    config = json.loads((ROOT / config_rel).read_text(encoding="utf-8"))
    output_dir = (args.output_dir or ROOT / "reproduction" / args.campaign).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.workers:
        config["design"]["workers"] = args.workers
    workers = int(config["design"]["workers"])
    relative_output = output_dir.relative_to(ROOT).as_posix() if output_dir.is_relative_to(ROOT) else str(output_dir)
    stem = args.campaign
    config["outputs"] = {
        "plan": f"{relative_output}/{stem}_plan.json",
        "records": f"{relative_output}/{stem}_records.jsonl",
        "summary": f"{relative_output}/{stem}_summary.json",
        "report": f"{relative_output}/{stem}_report.md",
        "shard_directory": f"{relative_output}/shards",
    }
    config_path = output_dir / f"{stem}_reproduction_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[{args.campaign}] loading frozen records ...", flush=True)
    frozen = {record_key(r): r for r in read_frozen_records(ROOT / records_rel)}

    loaded = load_campaign_config(config_path)
    adapter = load_adapter(loaded["adapter"], loaded.get("adapter_config", {}), ROOT)
    plan = build_plan(loaded, adapter, runner_path=SCRIPTS / "run_confirmatory_semantic_audit.py")
    plan_path = ROOT / config["outputs"]["plan"] if not Path(config["outputs"]["plan"]).is_absolute() else Path(config["outputs"]["plan"])
    freeze_plan(plan_path, plan)
    shards = task_shards(plan, int(loaded["design"]["task_shard_size"]))

    started = time.time()
    if args.full:
        chosen = shards
        print(f"[{args.campaign}] full reproduction: {len(chosen)} shards, {workers} workers", flush=True)
    else:
        chosen = evenly_spread(shards, max(1, args.max_shards))
        print(f"[{args.campaign}] quick mode: {len(chosen)} of {len(shards)} shards, {workers} workers", flush=True)
    run_shards(config_path, [s["shard_id"] for s in chosen], workers)

    fresh: list[dict[str, Any]] = []
    for shard in chosen:
        fresh.extend(read_jsonl(output_dir / "shards" / f"{shard['shard_id']}.jsonl"))
    result = compare(frozen, fresh)
    result.update(
        {
            "campaign": args.campaign,
            "mode": "full" if args.full else "quick",
            "shards_rerun": len(chosen),
            "shards_total": len(shards),
            "compared_fields": list(COMPARED_FIELDS),
            "wall_clock_seconds": round(time.time() - started, 1),
        }
    )
    (output_dir / "comparison.json").write_bytes(canonical_json(result))
    ok = (
        result["records_with_any_difference"] == 0
        and result["records_not_in_frozen_corpus"] == 0
        and result["records_compared"] > 0
    )
    print(json.dumps({k: v for k, v in result.items() if k != "compared_fields"}, indent=2, sort_keys=True))
    print(f"[{args.campaign}] " + ("REPRODUCED: all compared records agree." if ok else "DIFFERENCES FOUND (see comparison.json)."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
