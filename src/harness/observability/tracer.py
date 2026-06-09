"""OpenTelemetry step tracer for HarnessAgent."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generator, Iterator

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace  # type: ignore[import]
    from opentelemetry.propagate import inject  # type: ignore[import]
    from opentelemetry.sdk.resources import Resource  # type: ignore[import]
    from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import]
    from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore[import]
    from opentelemetry.trace import StatusCode  # type: ignore[import]
    from opentelemetry.trace.propagation.tracecontext import (  # type: ignore[import]
        TraceContextTextMapPropagator,
    )

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False
    logger.warning(
        "opentelemetry packages not installed — tracing is a no-op. "
        "Install with: pip install opentelemetry-sdk opentelemetry-exporter-otlp"
    )

if TYPE_CHECKING:
    from harness.core.context import AgentContext


# The OTel TracerProvider is process-global. Setting it more than once leaks
# BatchSpanProcessor background threads (the old provider is never shut down)
# and warns. Guard it so only the first StepTracer installs the provider.
_PROVIDER: Any | None = None


def _setup_tracer_provider(
    service_name: str,
    exporter_endpoint: str | None = None,
) -> Any:
    """Configure and register the global TracerProvider once (idempotent)."""
    global _PROVIDER
    if not _OTEL_AVAILABLE:
        return None
    if _PROVIDER is not None:
        # Already installed — reuse it instead of overwriting the global and
        # orphaning the previous provider's exporter threads.
        return _PROVIDER

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if exporter_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore[import]
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(endpoint=exporter_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info(
                "OTLP trace exporter configured → %s", exporter_endpoint
            )
        except ImportError:
            logger.warning(
                "opentelemetry-exporter-otlp not installed — traces won't be exported."
            )

    trace.set_tracer_provider(provider)
    _PROVIDER = provider
    return provider


def shutdown_tracer_provider() -> None:
    """Flush and shut down the global TracerProvider, releasing exporter threads.

    Safe to call multiple times and when OTel is not installed. Call this on
    application shutdown so BatchSpanProcessor threads exit cleanly.
    """
    global _PROVIDER
    if _PROVIDER is None:
        return
    try:
        _PROVIDER.shutdown()
    except Exception as exc:  # pragma: no cover - best-effort cleanup
        logger.debug("TracerProvider shutdown failed: %s", exc)
    finally:
        _PROVIDER = None


class _NoOpSpan:
    """No-op span used when OpenTelemetry is not available."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def record_exception(self, exc: Exception) -> None:
        pass

    def set_status(self, status: Any, description: str = "") -> None:
        pass

    def add_event(self, name: str, attributes: dict | None = None) -> None:
        pass


class StepTracer:
    """
    OpenTelemetry-based tracer for agent steps.

    Wraps the OTel Tracer API to provide:
    - Context-manager span creation with automatic error recording.
    - AgentContext attribute injection.
    - W3C TraceContext propagation helpers.
    """

    def __init__(
        self,
        service_name: str,
        exporter_endpoint: str | None = None,
    ) -> None:
        self._service_name = service_name
        if _OTEL_AVAILABLE:
            _setup_tracer_provider(service_name, exporter_endpoint)
            self._tracer = trace.get_tracer(service_name)
        else:
            self._tracer = None

    @contextmanager
    def span(
        self,
        name: str,
        ctx: "AgentContext | None" = None,
        **attrs: Any,
    ) -> Generator[Any, None, None]:
        """
        Context manager that wraps a code block in an OTel span.

        Automatically:
        - Attaches AgentContext attributes (run_id, tenant_id, etc.).
        - Records exceptions and sets ERROR status on unhandled exceptions.
        - Applies any extra keyword arguments as span attributes.
        """
        if not _OTEL_AVAILABLE or self._tracer is None:
            yield _NoOpSpan()
            return

        with self._tracer.start_as_current_span(name) as otel_span:
            # Attach AgentContext fields
            if ctx is not None:
                otel_span.set_attribute("run_id", ctx.run_id)
                otel_span.set_attribute("tenant_id", ctx.tenant_id)
                otel_span.set_attribute("agent_type", ctx.agent_type)
                otel_span.set_attribute("step_count", ctx.step_count)
                otel_span.set_attribute("token_count", ctx.token_count)
                if ctx.trace_id:
                    otel_span.set_attribute("harness.trace_id", ctx.trace_id)

            # Extra keyword attributes
            for key, value in attrs.items():
                otel_span.set_attribute(key, str(value))

            try:
                yield otel_span
            except Exception as exc:
                otel_span.record_exception(exc)
                otel_span.set_status(StatusCode.ERROR, description=str(exc))
                raise

    def get_current_trace_id(self) -> str | None:
        """Return the hex trace ID of the currently active span, or None."""
        if not _OTEL_AVAILABLE:
            return None
        try:
            span = trace.get_current_span()
            ctx = span.get_span_context()
            if ctx and ctx.is_valid:
                return format(ctx.trace_id, "032x")
        except Exception:
            pass
        return None

    def inject_context_to_dict(self, carrier: dict[str, Any]) -> dict[str, Any]:
        """
        Inject W3C TraceContext headers into ``carrier`` for cross-process propagation.

        Returns the carrier dict (mutated in place for convenience).
        """
        if not _OTEL_AVAILABLE:
            return carrier
        try:
            inject(carrier)
        except Exception as exc:
            logger.debug("TraceContext injection failed: %s", exc)
        return carrier
