from __future__ import annotations

from pathlib import Path
import json

import pytest

from tlrl_provenance.campaign import (
    RECORD_SCHEMA,
    AuditSubject,
    AuditTask,
    build_plan,
    build_summary,
    expected_keys,
    freeze_plan,
    record_key,
    task_shards,
)
from scripts import run_confirmatory_semantic_audit as runner


class FakeSession:
    def run_episode(self, *, evaluation_seed: int, deterministic: bool, verification_mode: str):
        assert deterministic is True
        assert verification_mode == "fast"
        return {
            "steps": 5,
            "terminated": True,
            "truncated": False,
            "official_success": True,
            "official_violation": False,
            "independent_verdict": "satisfied",
            "disposition": "decided_satisfaction",
            "return_sum": float(evaluation_seed % 7),
            "possible_evidence_count": 2,
            "necessary_evidence_count": 1,
            "atom_counts": {"a": 1},
            "failure_class": "decided_satisfaction",
            "official_independent_mismatch": False,
        }

    def close(self) -> None:
        pass


class FakeAdapter:
    def __init__(self) -> None:
        self._subjects = (
            AuditSubject("policy-a", {"seed": 1}),
            AuditSubject("policy-b", {"seed": 2}),
        )
        self._tasks = (
            AuditTask("formula-0", "F a", "eventually"),
            AuditTask("formula-1", "G !b", "safety"),
            AuditTask("formula-2", "a U b", "until"),
        )

    def subjects(self):
        return self._subjects

    def tasks(self):
        return self._tasks

    def provenance(self):
        return {"adapter": "fake/1"}

    def open_session(self, subject, task):
        return FakeSession()


def config() -> dict:
    return {
        "schema_version": "tlrl-confirmatory-semantic-campaign/1",
        "campaign_id": "test-campaign",
        "benchmark_id": "fake-benchmark",
        "adapter": "fake:adapter",
        "design": {
            "episodes_per_task": 2,
            "deterministic": True,
            "verification_mode": "fast",
            "workers": 2,
            "task_shard_size": 2,
            "confidence": 0.95,
            "interpretation_guardrails": {"hold_formula_fixed": True},
        },
        "seed_namespace": {
            "base": 10_000,
            "subject_stride": 1_000,
            "task_stride": 10,
            "episode_stride": 1,
            "forbidden_values": [999],
            "forbidden_ranges": [[20_000, 30_000]],
        },
        "outputs": {},
    }


def records_for(plan: dict) -> list[dict]:
    records = []
    for index, cell in enumerate(plan["cells"]):
        satisfied = index % 3 != 0
        disposition = "decided_satisfaction" if satisfied else "decided_violation"
        records.append(
            {
                "schema_version": RECORD_SCHEMA,
                "campaign_id": plan["campaign_id"],
                "benchmark_id": plan["benchmark_id"],
                "plan_sha256": plan["plan_sha256"],
                **cell,
                "steps": 10,
                "terminated": True,
                "truncated": False,
                "official_success": satisfied,
                "official_violation": not satisfied,
                "independent_verdict": "satisfied" if satisfied else "violated",
                "disposition": disposition,
                "return_sum": 1.0,
                "possible_evidence_count": 4,
                "necessary_evidence_count": 0 if satisfied else 2,
                "atom_counts": {"a": 1},
                "failure_class": (
                    "decided_satisfaction" if satisfied else "forbidden_or_temporal_decided_violation"
                ),
                "official_independent_mismatch": False,
            }
        )
    return records


def test_plan_preallocates_unique_reproducible_cells_and_shards() -> None:
    first = build_plan(config(), FakeAdapter())
    second = build_plan(config(), FakeAdapter())
    assert first == second
    assert first["expected"] == {
        "subject_count": 2,
        "task_count": 3,
        "episodes_per_task": 2,
        "record_count": 12,
    }
    assert len(expected_keys(first)) == 12
    assert len({cell["evaluation_seed"] for cell in first["cells"]}) == 12
    assert [len(shard["task_ids"]) for shard in task_shards(first, 2)] == [2, 1, 2, 1]


def test_seed_namespace_collision_fails_before_execution() -> None:
    bad = config()
    bad["seed_namespace"]["forbidden_values"] = [10_010]
    with pytest.raises(ValueError, match="collides"):
        build_plan(bad, FakeAdapter())


def test_frozen_plan_cannot_be_replaced(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    plan = build_plan(config(), FakeAdapter())
    freeze_plan(path, plan)
    freeze_plan(path, plan)
    changed = dict(plan)
    changed["benchmark_id"] = "changed"
    with pytest.raises(RuntimeError, match="differs"):
        freeze_plan(path, changed)


def test_complete_summary_has_fixed_formula_and_subject_views() -> None:
    plan = build_plan(config(), FakeAdapter())
    summary = build_summary(records_for(plan), plan)
    assert summary["completeness"]["status"] == "passed"
    assert summary["aggregate"]["satisfaction"]["successes"] == 8
    assert set(summary["by_subject"]) == {"policy-a", "policy-b"}
    assert set(summary["by_task"]) == {"formula-0", "formula-1", "formula-2"}
    assert len(summary["by_subject_task"]) == 6
    assert summary["interpretation_guardrails"] == {"hold_formula_fixed": True}


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unexpected"])
def test_summary_fails_closed_on_nonexact_record_frame(mutation: str) -> None:
    plan = build_plan(config(), FakeAdapter())
    records = records_for(plan)
    if mutation == "missing":
        records.pop()
    elif mutation == "duplicate":
        records.append(dict(records[0]))
    else:
        records[0] = dict(records[0], evaluation_seed=123456789)
    with pytest.raises((RuntimeError, ValueError)):
        build_summary(records, plan)


def test_summary_fails_closed_on_official_independent_mismatch() -> None:
    plan = build_plan(config(), FakeAdapter())
    records = records_for(plan)
    records[0]["official_independent_mismatch"] = True
    with pytest.raises(ValueError, match="mismatch"):
        build_summary(records, plan)


def test_record_key_includes_seed_to_detect_seed_plan_corruption() -> None:
    plan = build_plan(config(), FakeAdapter())
    record = records_for(plan)[0]
    assert record_key(record) == ("policy-a", "formula-0", 0, 10_000)


def test_worker_resume_and_merge_are_end_to_end_exact(tmp_path: Path, monkeypatch) -> None:
    campaign = config()
    campaign["outputs"] = {
        "plan": str(tmp_path / "plan.json"),
        "shard_directory": str(tmp_path / "shards"),
        "records": str(tmp_path / "records.jsonl"),
        "summary": str(tmp_path / "summary.json"),
        "report": str(tmp_path / "report.md"),
    }
    config_path = tmp_path / "campaign.json"
    config_path.write_text(json.dumps(campaign), encoding="utf-8")
    adapter = FakeAdapter()
    plan = build_plan(campaign, adapter)
    freeze_plan(tmp_path / "plan.json", plan)
    monkeypatch.setattr(runner, "load_adapter", lambda *args, **kwargs: adapter)
    shards = task_shards(plan, campaign["design"]["task_shard_size"])
    for shard in shards:
        runner.run_worker(config_path, tmp_path / "plan.json", shard["shard_id"])
    first_payloads = {
        shard["shard_id"]: (tmp_path / "shards" / f"{shard['shard_id']}.jsonl").read_bytes()
        for shard in shards
    }
    # A resumed completed worker must perform no rollout and change no byte.
    runner.run_worker(config_path, tmp_path / "plan.json", shards[0]["shard_id"])
    assert (tmp_path / "shards" / f"{shards[0]['shard_id']}.jsonl").read_bytes() == first_payloads[shards[0]["shard_id"]]
    summary = runner.merge_and_report(campaign, plan, runner.output_paths(campaign))
    assert summary["completeness"]["observed_records"] == 12
    assert summary["aggregate"]["satisfaction"]["value"] == 1.0
    assert (tmp_path / "report.md").is_file()
