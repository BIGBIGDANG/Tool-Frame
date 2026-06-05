"""Aggregate PartToolGen benchmark task reports."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return (sum(1 for row in rows if row.get(key)) / float(len(rows))) if rows else 0.0


def aggregate_reports(reports: list[dict[str, Any]], *, output_dir: Path) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        by_family[str(report.get("family"))].append(report)
    summary = {
        "schema_version": "parttoolgen_tool_use_eval_summary_v0",
        "task_count": len(reports),
        "passed_count": sum(1 for row in reports if row.get("passed")),
        "pass_rate": _rate(reports, "passed"),
        "tracking_pass_rate": _rate(reports, "tracking_passed"),
        "task_success_rate": _rate(reports, "task_success"),
        "by_family": {
            family: {
                "task_count": len(rows),
                "pass_rate": _rate(rows, "passed"),
                "tracking_pass_rate": _rate(rows, "tracking_passed"),
                "task_success_rate": _rate(rows, "task_success"),
            }
            for family, rows in sorted(by_family.items())
        },
        "failure_categories": {},
        "reports": reports,
    }
    failures: dict[str, int] = defaultdict(int)
    for report in reports:
        if not report.get("passed"):
            failures[str(report.get("failure_category") or "unknown")] += 1
    summary["failure_categories"] = dict(sorted(failures.items()))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# PartToolGen Tool-Use Benchmark Summary",
        "",
        f"- tasks: {summary['task_count']}",
        f"- pass rate: {summary['pass_rate']:.3f}",
        f"- tracking pass rate: {summary['tracking_pass_rate']:.3f}",
        f"- task success rate: {summary['task_success_rate']:.3f}",
        "",
        "## By Family",
        "",
    ]
    for family, row in summary["by_family"].items():
        lines.append(
            f"- {family}: pass={row['pass_rate']:.3f}, tracking={row['tracking_pass_rate']:.3f}, task={row['task_success_rate']:.3f}, n={row['task_count']}"
        )
    lines.extend(["", "## Failure Categories", ""])
    for category, count in summary["failure_categories"].items():
        lines.append(f"- {category}: {count}")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
