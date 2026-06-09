"""OnlineLearningMonitor — tracks per-version rolling performance and detects regressions."""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_METRICS_KEY = "harness:online_metrics:{agent_type}:{version_id}"
_WINDOW_KEY = "harness:online_window:{agent_type}:{version_id}"
_PENDING_CHECK_KEY = "harness:hermes:pending_check:{agent_type}"


@dataclass
class VersionMetrics:
    """Snapshot of rolling performance for one prompt version."""

    version_id: str
    agent_type: str
    success_rate: float
    avg_cost_usd: float
    avg_steps: float
    sample_count: int
    min_reliable_samples: int = 10

    @property
    def is_reliable(self) -> bool:
        """True when there are enough samples to trust the metrics."""
        return self.sample_count >= self.min_reliable_samples


@dataclass
class PendingRollbackCheck:
    """Stored when a patch is auto-applied so the next cycle can verify it."""

    agent_type: str
    patch_id: str
    baseline_version_id: str
    new_version_id: str
    baseline_error_count: int
    applied_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "agent_type": self.agent_type,
            "patch_id": self.patch_id,
            "baseline_version_id": self.baseline_version_id,
            "new_version_id": self.new_version_id,
            "baseline_error_count": self.baseline_error_count,
            "applied_at": self.applied_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "PendingRollbackCheck":
        return cls(
            agent_type=d["agent_type"],
            patch_id=d["patch_id"],
            baseline_version_id=d["baseline_version_id"],
            new_version_id=d["new_version_id"],
            baseline_error_count=int(d["baseline_error_count"]),
            applied_at=float(d.get("applied_at", time.time())),
        )

    @classmethod
    def from_json(cls, raw: str) -> "PendingRollbackCheck":
        return cls.from_dict(json.loads(raw))


class OnlineLearningMonitor:
    """Tracks rolling per-version performance in Redis and detects regressions.

    Designed to be called from ``BaseAgent.run()`` after every run completes,
    and from ``HermesLoop`` to schedule and execute post-apply rollback checks.

    Args:
        redis:                Async Redis client.
        mlflow_tracer:        Optional MLflowAgentTracer for metric logging.
        window_size:          Number of most-recent runs to keep per version.
        regression_threshold: Fraction by which error count must grow to trigger rollback.
        min_samples:          Minimum window samples before declaring reliable metrics.
        snapshot_interval:    Log to MLflow every N recorded runs per version.
    """

    def __init__(
        self,
        redis: Any,
        mlflow_tracer: Any | None = None,
        window_size: int = 100,
        regression_threshold: float = 0.30,
        min_samples: int = 10,
        snapshot_interval: int = 20,
    ) -> None:
        self._redis = redis
        self._mlflow = mlflow_tracer
        self.window_size = window_size
        self.regression_threshold = regression_threshold
        self.min_samples = min_samples
        self.snapshot_interval = snapshot_interval

    # ------------------------------------------------------------------
    # Per-run recording
    # ------------------------------------------------------------------

    async def record_run(
        self,
        agent_type: str,
        version_id: str,
        version_number: int,
        success: bool,
        cost_usd: float = 0.0,
        steps: int = 0,
    ) -> None:
        """Record one completed run against the given prompt version.

        Args:
            agent_type:     The agent type (e.g. "sql").
            version_id:     The active prompt version ID at run time.
            version_number: The human-readable version number.
            success:        Whether the run succeeded.
            cost_usd:       USD cost of the run.
            steps:          Number of agent steps taken.
        """
        metrics_key = _METRICS_KEY.format(agent_type=agent_type, version_id=version_id)
        window_key = _WINDOW_KEY.format(agent_type=agent_type, version_id=version_id)
        now = time.time()

        # Include a unique ID so Redis ZADD never deduplicates concurrent entries
        entry = json.dumps({"s": int(success), "c": cost_usd, "st": steps,
                            "id": uuid.uuid4().hex[:8]})

        pipe = self._redis.pipeline(transaction=False)
        # Aggregate counters (all-time, not windowed)
        pipe.hincrbyfloat(metrics_key, "success_count" if success else "failure_count", 1)
        pipe.hincrbyfloat(metrics_key, "total_cost_usd", cost_usd)
        pipe.hincrbyfloat(metrics_key, "total_steps", steps)
        pipe.hset(metrics_key, "version_number", version_number)
        pipe.hsetnx(metrics_key, "first_run_at", now)
        pipe.hset(metrics_key, "last_run_at", now)
        pipe.expire(metrics_key, 86400 * 90)

        # Sliding window sorted set (score = timestamp)
        pipe.zadd(window_key, {entry: now})
        pipe.zremrangebyrank(window_key, 0, -(self.window_size + 2))
        pipe.expire(window_key, 86400 * 90)
        await pipe.execute()

        # Periodically snapshot to MLflow
        window_count = await self._redis.zcard(window_key)
        if self._mlflow and int(window_count) % self.snapshot_interval == 0:
            vm = await self.get_windowed_metrics(agent_type, version_id)
            try:
                await self._mlflow.log_online_metrics(
                    agent_type=agent_type,
                    version_id=version_id,
                    version_number=version_number,
                    success_rate=vm.success_rate,
                    avg_cost=vm.avg_cost_usd,
                    avg_steps=vm.avg_steps,
                    sample_count=vm.sample_count,
                )
            except Exception as exc:
                logger.debug("online_monitor MLflow snapshot failed: %s", exc)

    # ------------------------------------------------------------------
    # Metrics retrieval
    # ------------------------------------------------------------------

    async def get_windowed_metrics(
        self, agent_type: str, version_id: str
    ) -> VersionMetrics:
        """Compute metrics from the recent sliding window.

        Args:
            agent_type: Agent type.
            version_id: Prompt version ID.

        Returns:
            VersionMetrics — ``is_reliable`` is False if window is too small.
        """
        window_key = _WINDOW_KEY.format(agent_type=agent_type, version_id=version_id)
        entries = await self._redis.zrange(window_key, -self.window_size, -1)

        if not entries:
            return VersionMetrics(
                version_id=version_id,
                agent_type=agent_type,
                success_rate=0.0,
                avg_cost_usd=0.0,
                avg_steps=0.0,
                sample_count=0,
                min_reliable_samples=self.min_samples,
            )

        successes = 0
        total_cost = 0.0
        total_steps = 0
        for raw in entries:
            data = json.loads(raw if isinstance(raw, str) else raw.decode())
            successes += data.get("s", 0)
            total_cost += data.get("c", 0.0)
            total_steps += data.get("st", 0)

        n = len(entries)
        return VersionMetrics(
            version_id=version_id,
            agent_type=agent_type,
            success_rate=successes / n,
            avg_cost_usd=total_cost / n,
            avg_steps=total_steps / n,
            sample_count=n,
            min_reliable_samples=self.min_samples,
        )

    # ------------------------------------------------------------------
    # Post-apply rollback scheduling
    # ------------------------------------------------------------------

    async def schedule_rollback_check(
        self,
        agent_type: str,
        patch_id: str,
        baseline_version_id: str,
        new_version_id: str,
        baseline_error_count: int,
    ) -> None:
        """Store a pending rollback check after a patch is auto-applied.

        Called by HermesLoop immediately after applying a patch. The check
        is evaluated at the start of the NEXT Hermes cycle.

        Args:
            agent_type:            The agent type that was patched.
            patch_id:              The patch that was applied.
            baseline_version_id:   The prompt version active before the patch.
            new_version_id:        The new prompt version ID after the patch.
            baseline_error_count:  Error count at the time of patching (used
                                   as regression baseline).
        """
        check = PendingRollbackCheck(
            agent_type=agent_type,
            patch_id=patch_id,
            baseline_version_id=baseline_version_id,
            new_version_id=new_version_id,
            baseline_error_count=baseline_error_count,
        )
        key = _PENDING_CHECK_KEY.format(agent_type=agent_type)
        await self._redis.set(key, check.to_json(), ex=86400 * 3)
        logger.info(
            "Scheduled rollback check for agent_type=%s patch=%s",
            agent_type,
            patch_id[:8],
        )

    async def pop_pending_check(
        self, agent_type: str
    ) -> PendingRollbackCheck | None:
        """Retrieve and delete any pending rollback check for agent_type."""
        key = _PENDING_CHECK_KEY.format(agent_type=agent_type)
        raw = await self._redis.getdel(key)
        if not raw:
            return None
        try:
            return PendingRollbackCheck.from_json(
                raw if isinstance(raw, str) else raw.decode()
            )
        except Exception as exc:
            logger.warning("Failed to parse pending rollback check: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Regression detection + rollback
    # ------------------------------------------------------------------

    async def _detect_rate_regression(
        self, check: PendingRollbackCheck
    ) -> tuple[bool | None, str]:
        """Compare windowed error *rates* of new vs baseline prompt version.

        Returns ``(regressed, detail)``. ``regressed`` is ``None`` when the
        windowed metrics are not usable (no version ids, or the new version has
        no samples yet) so the caller can fall back to the count heuristic.

        Error rate = 1 - success_rate over the recent sliding window. The new
        version regresses when its error rate exceeds the baseline's by more
        than ``regression_threshold`` (relative).
        """
        if not check.new_version_id:
            return None, ""
        try:
            new_vm = await self.get_windowed_metrics(
                check.agent_type, check.new_version_id
            )
        except Exception as exc:
            logger.warning("rollback check: windowed metrics failed: %s", exc)
            return None, ""

        # Need at least a few post-patch runs to judge.
        if new_vm.sample_count < 3:
            return None, ""

        new_err = 1.0 - new_vm.success_rate

        baseline_err: float | None = None
        if check.baseline_version_id:
            try:
                base_vm = await self.get_windowed_metrics(
                    check.agent_type, check.baseline_version_id
                )
                if base_vm.sample_count > 0:
                    baseline_err = 1.0 - base_vm.success_rate
            except Exception as exc:
                logger.debug("rollback check: baseline metrics failed: %s", exc)

        if baseline_err is None:
            # No comparable baseline window — treat a high absolute post-patch
            # error rate as a regression.
            regressed = new_err >= 0.5
            return regressed, (
                f"rate-based (new_err={new_err:.2f} over n={new_vm.sample_count}, "
                f"no baseline window)"
            )

        if baseline_err <= 0:
            # Baseline was perfect; any non-trivial error rate is a regression.
            regressed = new_err > self.regression_threshold
        else:
            regressed = new_err > baseline_err * (1 + self.regression_threshold)
        return regressed, (
            f"rate-based (baseline_err={baseline_err:.2f} new_err={new_err:.2f} "
            f"over n={new_vm.sample_count})"
        )

    async def check_and_maybe_rollback(
        self,
        agent_type: str,
        error_collector: Any,
        prompt_manager: Any,
    ) -> bool:
        """Run the pending rollback check for *agent_type* if one exists.

        Compares current error count against baseline. If errors increased
        by more than ``regression_threshold``, rolls back the prompt and
        logs the event.

        Args:
            agent_type:      Agent type to check.
            error_collector: ErrorCollector for current error count.
            prompt_manager:  PromptManager to call rollback on.

        Returns:
            True if a rollback was performed, False otherwise.
        """
        check = await self.pop_pending_check(agent_type)
        if check is None:
            return False

        # Preferred path: compare *windowed error rates* (errors per run) between
        # the post-patch version and the pre-patch baseline version. Cumulative
        # all-time error counts (error_collector.count -> zcard) only ever grow,
        # so a count-based comparison flags a regression on any healthy system
        # that simply keeps running.
        regressed, detail = await self._detect_rate_regression(check)

        if regressed is None:
            # Windowed metrics unavailable (e.g. no per-version samples yet) —
            # fall back to the count-based heuristic.
            try:
                current_error_count = await error_collector.count(agent_type)
            except Exception as exc:
                logger.warning("rollback check: error count failed: %s", exc)
                return False

            baseline = check.baseline_error_count
            regressed = (
                baseline > 0
                and current_error_count > baseline * (1 + self.regression_threshold)
            ) or (
                baseline == 0
                and current_error_count >= 3
            )
            detail = (
                f"count-based (baseline={baseline} current={current_error_count})"
            )

        if not regressed:
            logger.info(
                "Post-apply check for agent_type=%s: no regression "
                "[%s] — patch %s retained",
                agent_type,
                detail,
                check.patch_id[:8],
            )
            return False

        # Regression detected — roll back
        logger.warning(
            "Regression detected for agent_type=%s [%s] "
            "(threshold=%.0f%%) — rolling back patch %s",
            agent_type,
            detail,
            self.regression_threshold * 100,
            check.patch_id[:8],
        )
        try:
            await prompt_manager.rollback(agent_type, steps=1)
            logger.info(
                "Rolled back prompt for agent_type=%s to version before patch %s",
                agent_type,
                check.patch_id[:8],
            )
            return True
        except Exception as exc:
            logger.error("Rollback failed for agent_type=%s: %s", agent_type, exc)
            return False
