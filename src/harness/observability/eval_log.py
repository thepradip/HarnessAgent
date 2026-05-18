"""log_all — dispatch eval results to MLflow, W&B, and LangSmith."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from harness.eval.agent_report import AgentEvalReport
    from harness.eval.runner import EvalReport


def log_all(
    report: "AgentEvalReport | EvalReport",
    mlflow_experiment: str | None = None,
    wandb_project: str | None = None,
    langsmith_project: str | None = None,
) -> None:
    """Log eval results to all configured backends. Each backend is optional."""
    if mlflow_experiment:
        _log_mlflow(report, mlflow_experiment)
    if wandb_project:
        _log_wandb(report, wandb_project)
    if langsmith_project:
        _log_langsmith(report, langsmith_project)


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------

def _log_mlflow(report: Any, experiment: str) -> None:
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow not installed — skipping MLflow logging")
        return
    try:
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=getattr(report, "dataset_name", "eval")):
            _log_mlflow_metrics(report)
    except Exception as exc:
        logger.warning("MLflow logging failed: %s", exc)


def _log_mlflow_metrics(report: Any) -> None:
    import mlflow

    # AgentEvalReport
    if hasattr(report, "overall_pass_rate"):
        mlflow.log_metric("pass_rate", report.overall_pass_rate())
        for dim, val in report.by_dimension().items():
            mlflow.log_metric(f"avg_{dim}", val)
        for hardness, rate in report.by_hardness().items():
            mlflow.log_metric(f"pass_rate_{hardness}", rate)
        for cat, count in report.failure_distribution().items():
            mlflow.log_metric(f"failure_{cat}", count)
        mlflow.log_param("dataset", getattr(report, "dataset_name", ""))
        mlflow.log_param("total_cases", len(getattr(report, "scores", [])))
        return

    # Legacy EvalReport
    if hasattr(report, "success_rate"):
        mlflow.log_metric("success_rate", report.success_rate)
        mlflow.log_metric("avg_tokens", report.avg_tokens)
        mlflow.log_metric("avg_cost_usd", report.avg_cost_usd)
        mlflow.log_metric("avg_latency_seconds", report.avg_latency_seconds)
        mlflow.log_param("dataset", report.dataset_name)
        mlflow.log_param("agent_type", report.agent_type)
        mlflow.log_param("total_cases", report.total_cases)


# ---------------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------------

def _log_wandb(report: Any, project: str) -> None:
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed — skipping W&B logging (pip install wandb)")
        return
    try:
        run = wandb.init(
            project=project,
            name=getattr(report, "dataset_name", "eval"),
            reinit=True,
        )
        _log_wandb_metrics(report, run)
        run.finish()
    except Exception as exc:
        logger.warning("W&B logging failed: %s", exc)


def _log_wandb_metrics(report: Any, run: Any) -> None:
    if hasattr(report, "overall_pass_rate"):
        metrics = {"pass_rate": report.overall_pass_rate()}
        metrics.update({f"avg_{k}": v for k, v in report.by_dimension().items()})
        metrics.update({f"pass_rate_{h}": v for h, v in report.by_hardness().items()})
        metrics.update({f"failure_{c}": n for c, n in report.failure_distribution().items()})
        run.log(metrics)
        # Per-case table
        cases = getattr(report, "scores", [])
        tasks = getattr(report, "tasks", [])
        if cases:
            import wandb
            table = wandb.Table(columns=["task", "verdict", "correctness", "quality", "safety", "hardness"])
            for i, s in enumerate(cases):
                table.add_data(
                    tasks[i] if i < len(tasks) else "",
                    s.verdict, s.correctness_score, s.quality_score, s.safety_score, s.hardness,
                )
            run.log({"cases": table})
        return

    if hasattr(report, "success_rate"):
        run.log({
            "success_rate": report.success_rate,
            "avg_tokens": report.avg_tokens,
            "avg_cost_usd": report.avg_cost_usd,
        })


# ---------------------------------------------------------------------------
# LangSmith
# ---------------------------------------------------------------------------

def _log_langsmith(report: Any, project: str) -> None:
    try:
        from langsmith import Client
    except ImportError:
        logger.warning("langsmith not installed — skipping LangSmith logging (pip install langsmith)")
        return
    try:
        client = Client()
        dataset_name = getattr(report, "dataset_name", "eval")
        scores = getattr(report, "scores", [])
        tasks = getattr(report, "tasks", [])
        actions = getattr(report, "actions", [])

        for i, s in enumerate(scores):
            run_name = f"{dataset_name}_case_{i}"
            task_str = tasks[i] if i < len(tasks) else ""
            action_str = actions[i] if i < len(actions) else ""
            try:
                client.create_run(
                    name=run_name,
                    project_name=project,
                    run_type="chain",
                    inputs={"task": task_str, "action": action_str},
                    outputs={
                        "verdict": s.verdict,
                        "correctness": s.correctness_score,
                        "quality": s.quality_score,
                        "safety": s.safety_score,
                        "overall": s.overall_score,
                    },
                    tags=[s.hardness, s.verdict.lower()],
                )
            except Exception as exc:
                logger.debug("LangSmith run %d failed: %s", i, exc)
    except Exception as exc:
        logger.warning("LangSmith logging failed: %s", exc)
