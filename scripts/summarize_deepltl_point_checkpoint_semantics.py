#!/usr/bin/env python
"""Render a practitioner-facing report from a DeepLTL semantic checkpoint sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_EVIDENCE = ROOT / "evidence" / "deepltl_point" / "development" / "semantic"
DEFAULT_INPUT = DEVELOPMENT_EVIDENCE / "deepltl_point_checkpoint_semantic_summary.json"
DEFAULT_OUTPUT = DEVELOPMENT_EVIDENCE / "deepltl_point_checkpoint_semantic_report.md"


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
    temporary.replace(path)


def resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def fmt_rate(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100 * float(value):.1f}%"


def fmt_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def checkpoint_rows(summary: dict[str, Any]) -> list[str]:
    rows = [
        "| checkpoint | episodes | P_sat | violations | censored | mean steps | zero-necessary satisfied | necessary/step median | necessary/step range |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for checkpoint, group in sorted(
        summary["by_checkpoint"].items(),
        key=lambda item: int(item[0]),
    ):
        verdicts = group["independent_verdicts"]
        ratios = group["satisfied_necessary_per_step"]
        rows.append(
            "| "
            + " | ".join(
                [
                    checkpoint,
                    str(group["episodes"]),
                    fmt_rate(group["satisfied_rate"]),
                    str(verdicts.get("violated", 0)),
                    str(group["dispositions"].get("censored", 0)),
                    fmt_float(group["mean_steps"]),
                    fmt_rate(group["satisfied_zero_necessary_rate"]),
                    fmt_float(ratios["median"]),
                    f"{fmt_float(ratios['min'])}-{fmt_float(ratios['max'])}",
                ]
            )
            + " |"
        )
    return rows


def family_rows(summary: dict[str, Any]) -> list[str]:
    rows = [
        "| family | checkpoint | episodes | P_sat | mean necessary evidence | zero-necessary satisfied | necessary/step median |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family, by_checkpoint in sorted(summary["by_formula_family_checkpoint"].items()):
        for checkpoint, group in sorted(by_checkpoint.items(), key=lambda item: int(item[0])):
            ratios = group["satisfied_necessary_per_step"]
            rows.append(
                "| "
                + " | ".join(
                    [
                        family,
                        checkpoint,
                        str(group["episodes"]),
                        fmt_rate(group["satisfied_rate"]),
                        fmt_float(group["mean_necessary_evidence"]),
                        fmt_rate(group["satisfied_zero_necessary_rate"]),
                        fmt_float(ratios["median"]),
                    ]
                )
                + " |"
            )
    return rows


def failure_rows(summary: dict[str, Any], *, limit: int) -> list[str]:
    rows = [
        "| checkpoint | formula | family | disposition | steps | atoms touched | possible | necessary |",
        "|---:|---|---|---|---:|---|---:|---:|",
    ]
    failures = sorted(
        summary["failures"],
        key=lambda item: (int(item["checkpoint_steps"]), item["task_id"], item["seed"]),
    )
    for failure in failures[:limit]:
        atoms = ", ".join(
            f"{atom}:{count}" for atom, count in sorted(failure["atom_counts"].items())
        )
        rows.append(
            "| "
            + " | ".join(
                [
                    str(failure["checkpoint_steps"]),
                    failure["task_id"],
                    failure["formula_family"],
                    failure["disposition"],
                    str(failure["steps"]),
                    atoms if atoms else "none",
                    str(failure["mean_possible_evidence"]),
                    str(failure["mean_necessary_evidence"]),
                ]
            )
            + " |"
        )
    if len(failures) > limit:
        rows.append(f"| ... | ... | ... | ... | ... | {len(failures) - limit} more | ... | ... |")
    return rows


def render(summary: dict[str, Any], *, failure_limit: int) -> str:
    aggregate = summary["aggregate"]
    guardrails = summary["interpretation_guardrails"]
    lines = [
        "# DeepLTL PointWorld semantic checkpoint analysis",
        "",
        "Scope: development-only formula-conditioned analysis of existing trained checkpoints. "
        "No retraining is performed by this report.",
        "",
        "## Headline",
        "",
        f"- Records: `{summary['record_count']}`",
        f"- Checkpoints: `{len(summary['checkpoint_steps'])}`",
        f"- Episodes per formula per checkpoint: `{summary['episodes_per_formula']}`",
        f"- Aggregate `P_sat`: `{fmt_rate(aggregate['satisfied_rate'])}`",
        f"- Aggregate censoring: `{fmt_rate(aggregate['censored_rate'])}`",
        f"- Official/independent mismatches: `{aggregate['official_independent_mismatch_count']}`",
        "",
        "## Interpretation guardrails",
        "",
    ]
    for key, value in sorted(guardrails.items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "In plain terms: compare policies/checkpoints with the formula held fixed. "
            "Do not read raw cross-formula evidence averages as controller robustness.",
            "",
            "## By checkpoint",
            "",
            *checkpoint_rows(summary),
            "",
            "## By formula family and checkpoint",
            "",
            *family_rows(summary),
            "",
            "## Failure catalogue",
            "",
            *failure_rows(summary, limit=failure_limit),
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--failure-limit", type=int, default=80)
    args = parser.parse_args()
    if args.failure_limit < 1:
        raise ValueError("failure-limit must be positive")
    input_path = resolve(args.input)
    output_path = resolve(args.output)
    summary = json.loads(input_path.read_text(encoding="utf-8"))
    report = render(summary, failure_limit=args.failure_limit)
    payload = report.encode("utf-8")
    atomic_write(output_path, payload)
    atomic_write(
        output_path.with_suffix(output_path.suffix + ".sha256"),
        f"{hashlib.sha256(payload).hexdigest()}  {output_path.name}\n".encode("ascii"),
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "input": str(input_path),
                "output": str(output_path),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
