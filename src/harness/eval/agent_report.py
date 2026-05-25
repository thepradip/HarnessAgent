"""AgentEvalReport — aggregated evaluation report for any agentic app."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

from harness.eval.agent_scorer import AgentScores
from harness.eval.failure_taxonomy import FailureCategory, attribute_to_component


@dataclass
class AgentEvalReport:
    """Aggregated evaluation report over a list of AgentScores."""

    dataset_name: str
    scores: list[AgentScores]
    tasks: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)

    def overall_pass_rate(self) -> float:
        if not self.scores:
            return 0.0
        return sum(1 for s in self.scores if s.verdict == "PASS") / len(self.scores)

    def by_hardness(self) -> dict[str, float]:
        """Pass rate per hardness tier."""
        buckets: dict[str, list[AgentScores]] = {}
        for s in self.scores:
            buckets.setdefault(s.hardness, []).append(s)
        return {
            h: sum(1 for s in bucket if s.verdict == "PASS") / len(bucket)
            for h, bucket in sorted(buckets.items())
        }

    def by_dimension(self) -> dict[str, float]:
        """Average score per dimension."""
        if not self.scores:
            return {"correctness": 0.0, "quality": 0.0, "safety": 0.0}
        n = len(self.scores)
        return {
            "correctness": round(sum(s.correctness_score for s in self.scores) / n, 4),
            "quality": round(sum(s.quality_score for s in self.scores) / n, 4),
            "safety": round(sum(s.safety_score for s in self.scores) / n, 4),
        }

    def failure_distribution(self) -> dict[str, int]:
        """Count per FailureCategory across all FAIL cases."""
        dist: dict[str, int] = {}
        for s in self.scores:
            if s.verdict == "FAIL" and s.failure_analysis is not None:
                key = s.failure_analysis.primary.value
                dist[key] = dist.get(key, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: -x[1]))

    def component_attribution(self) -> dict[str, int]:
        """Count FAIL cases per harness component (LLM / TOOL / SAFETY / BUDGET …).

        Uses ``AgentScores.harness_failure_class`` when available, falls back to
        inferring the component from the content-level ``FailureCategory``.
        Provides a coarse-grained signal for which infrastructure layer to tune.
        """
        dist: dict[str, int] = {}
        for s in self.scores:
            if s.verdict != "FAIL":
                continue
            comp = attribute_to_component(s.harness_failure_class).value
            dist[comp] = dist.get(comp, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: -x[1]))

    def to_markdown(self) -> str:
        dims = self.by_dimension()
        hardness = self.by_hardness()
        failures = self.failure_distribution()
        total = len(self.scores)
        passed = sum(1 for s in self.scores if s.verdict == "PASS")

        lines = [
            f"# Eval Report: {self.dataset_name}",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total cases | {total} |",
            f"| Passed | {passed} |",
            f"| Pass rate | {self.overall_pass_rate():.1%} |",
            f"| Avg correctness | {dims['correctness']:.3f} |",
            f"| Avg quality | {dims['quality']:.3f} |",
            f"| Avg safety | {dims['safety']:.3f} |",
            "",
        ]

        if hardness:
            lines += [
                "## By Hardness",
                "",
                "| Hardness | Pass Rate |",
                "|----------|-----------|",
            ]
            for h, rate in hardness.items():
                lines.append(f"| {h} | {rate:.1%} |")
            lines.append("")

        attribution = self.component_attribution()
        if attribution:
            lines += [
                "## Harness Component Attribution",
                "",
                "| Component | Failures | % of FAILs |",
                "|-----------|----------|------------|",
            ]
            total_fails = sum(attribution.values()) or 1
            for comp, count in attribution.items():
                pct = count / total_fails * 100
                lines.append(f"| `{comp}` | {count} | {pct:.0f}% |")
            lines.append("")

        if failures:
            lines += [
                "## Top Failure Categories",
                "",
                "| Category | Count |",
                "|----------|-------|",
            ]
            for cat, count in list(failures.items())[:5]:
                lines.append(f"| `{cat}` | {count} |")
            lines.append("")

        # Per-case table (capped at 20)
        cases = list(zip(
            self.scores,
            self.tasks or [""] * total,
            self.actions or [""] * total,
        ))
        if cases:
            lines += [
                "## Per-Case Results (first 20)",
                "",
                "| # | Task | Verdict | Correctness | Quality | Safety | Hardness | Failure |",
                "|---|------|---------|-------------|---------|--------|----------|---------|",
            ]
            for i, (s, task, _action) in enumerate(cases[:20]):
                fail_cat = s.failure_analysis.primary.value if s.failure_analysis else "—"
                task_short = (task[:40] + "…") if len(task) > 40 else task
                lines.append(
                    f"| {i+1} | {task_short} | {s.verdict} | {s.correctness_score:.2f} "
                    f"| {s.quality_score:.2f} | {s.safety_score:.2f} "
                    f"| {s.hardness} | {fail_cat} |"
                )

        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps({
            "dataset_name": self.dataset_name,
            "total": len(self.scores),
            "pass_rate": self.overall_pass_rate(),
            "by_dimension": self.by_dimension(),
            "by_hardness": self.by_hardness(),
            "failure_distribution": self.failure_distribution(),
            "component_attribution": self.component_attribution(),
            "cases": [json.loads(s.to_json()) for s in self.scores],
        }, indent=2)


def generate_report(
    scores: list[AgentScores],
    tasks: list[str] | None = None,
    actions: list[str] | None = None,
    dataset_name: str = "eval",
    format: Literal["markdown", "json"] = "markdown",
) -> str:
    """Generate a markdown or JSON report from a list of AgentScores."""
    report = AgentEvalReport(
        dataset_name=dataset_name,
        scores=scores,
        tasks=tasks or [],
        actions=actions or [],
    )
    return report.to_markdown() if format == "markdown" else report.to_json()
