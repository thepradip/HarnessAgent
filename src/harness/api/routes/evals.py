"""Evaluation API routes for agent harness diagnostics."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from harness.api.deps import get_current_tenant
from harness.core.context import AgentResult
from harness.eval import (
    MULTI_AGENT_EVAL_CASES,
    EvalCase,
    EvalDataset,
    EvalRunner,
    MultiAgentEvalDataset,
)
from harness.orchestrator.runner import RunRecord

router = APIRouter()


class EvalRunRequest(BaseModel):
    """Optional controls for an eval run."""

    prompt_version: str = Field(default="ui", description="Prompt version label")


class EvalCompareRequest(BaseModel):
    """Optional controls for a prompt comparison."""

    baseline_prompt_version: str = "baseline"
    patched_prompt_version: str = "patched"


class _DemoEvalRunner:
    """AgentRunner-shaped local runner for deterministic UI smoke evals.

    Production deployments can replace this route implementation with a runner
    backed by real queues. Keeping this deterministic lets the console verify
    eval logic without Redis, API keys, or model spend.
    """

    def __init__(self) -> None:
        self._records: dict[str, RunRecord] = {}

    async def create_run(
        self,
        tenant_id: str,
        agent_type: str,
        task: str,
        metadata: dict[str, Any] | None = None,
    ) -> RunRecord:
        record = RunRecord(
            tenant_id=tenant_id,
            agent_type=agent_type,
            task=task,
            metadata=metadata or {},
        )
        self._records[record.run_id] = record
        return record

    async def execute_run(self, run_id: str) -> RunRecord:
        record = self._records[run_id]
        tokens = 180 + len(record.task.split()) * 16
        is_handoff = bool(record.metadata.get("depends_on"))
        result = AgentResult(
            run_id=record.run_id,
            output=(
                f"{record.agent_type} completed eval task. "
                "risk summary, remediation checklist, and routing decision"
            ),
            steps=3 if not is_handoff else 4,
            tokens=tokens,
            success=True,
            cost_usd=tokens * 0.0000004,
            elapsed_seconds=0.08,
            tool_calls=1,
            tool_errors=0,
            guardrail_hits=0,
            handoff_count=int(record.metadata.get("handoff_count", 0) or 0),
            cache_hits=1,
            cache_read_tokens=64,
        )
        record.status = "completed"
        record.result = _result_to_dict(result)
        return record


def _result_to_dict(result: AgentResult) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "output": result.output,
        "steps": result.steps,
        "tokens": result.tokens,
        "success": result.success,
        "failure_class": result.failure_class,
        "error_message": result.error_message,
        "elapsed_seconds": result.elapsed_seconds,
        "cost_usd": result.cost_usd,
        "tool_calls": result.tool_calls,
        "tool_errors": result.tool_errors,
        "guardrail_hits": result.guardrail_hits,
        "handoff_count": result.handoff_count,
        "cache_hits": result.cache_hits,
        "cache_read_tokens": result.cache_read_tokens,
    }


def _single_smoke_dataset() -> EvalDataset:
    return EvalDataset(
        name="single_smoke",
        agent_type="code",
        cases=[
            EvalCase(
                case_id="single_code_smoke",
                agent_type="code",
                task="Inspect one repository change and produce a risk summary.",
                expected_output="risk",
                tags=["smoke", "code", "cost"],
            ),
            EvalCase(
                case_id="single_sql_smoke",
                agent_type="sql",
                task="Inspect database shape and produce a remediation checklist.",
                expected_output="remediation",
                tags=["smoke", "sql", "tool"],
            ),
        ],
    )


@router.get("/suites")
async def list_eval_suites() -> dict[str, Any]:
    """List built-in eval suites available from the operator console."""
    return {
        "suites": [
            {
                "id": "smoke",
                "name": "Single-agent smoke",
                "scope": ["sql", "code", "tool", "cost"],
            },
            {
                "id": "multi",
                "name": "Multi-agent handoff",
                "scope": ["planner", "scheduler", "guardrails", "handoff"],
            },
            {
                "id": "compare",
                "name": "Prompt comparison",
                "scope": ["prompt", "tokens", "cost"],
            },
        ]
    }


@router.post("/smoke/run")
async def run_smoke_eval(
    body: EvalRunRequest | None = None,
    tenant_id: str = Depends(get_current_tenant),
) -> dict[str, Any]:
    """Run the built-in single-agent smoke evaluation."""
    runner = EvalRunner(_DemoEvalRunner())
    report = await runner.run(
        _single_smoke_dataset(),
        prompt_version=(body.prompt_version if body else "ui"),
    )
    return report.to_dict()


@router.post("/multi/run")
async def run_multi_eval(
    body: EvalRunRequest | None = None,
    tenant_id: str = Depends(get_current_tenant),
) -> dict[str, Any]:
    """Run the built-in multi-agent handoff evaluation."""
    runner = EvalRunner(_DemoEvalRunner())
    report = await runner.run_multi_agent(
        MultiAgentEvalDataset(
            name="multi_handoff",
            cases=MULTI_AGENT_EVAL_CASES,
        ),
        prompt_version=(body.prompt_version if body else "ui"),
    )
    return report.to_dict()


@router.post("/compare")
async def compare_prompt_eval(
    body: EvalCompareRequest | None = None,
    tenant_id: str = Depends(get_current_tenant),
) -> dict[str, Any]:
    """Compare baseline and patched prompt labels over the smoke suite."""
    body = body or EvalCompareRequest()
    runner = EvalRunner(_DemoEvalRunner())
    baseline = await runner.run(
        _single_smoke_dataset(),
        prompt_version=body.baseline_prompt_version,
    )
    patched = await runner.run(
        _single_smoke_dataset(),
        prompt_version=body.patched_prompt_version,
    )
    return {
        "baseline": baseline.to_dict(),
        "patched": patched.to_dict(),
        "markdown": patched.compare(baseline),
    }
