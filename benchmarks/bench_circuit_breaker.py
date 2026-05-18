"""Benchmark 3: Circuit breaker — failure injection, state transitions, recovery timing.

Simulates three realistic failure scenarios:
  A) Gradual degradation   — failures ramp up, circuit opens, recovers
  B) Burst failure         — 5 consecutive failures, immediate trip, recover
  C) Intermittent failure  — failures never reach threshold (no false trip)

Measures:
  - Calls-to-open: how many failures before circuit opens
  - Time-to-open (ms): wall-clock from first failure to OPEN state
  - Time-in-open (ms): duration in OPEN state before probing
  - Calls-to-close: how many successes after HALF_OPEN before CLOSED
  - False-positive trips: did CLOSED trip on healthy traffic?

Run:
    PYTHONPATH=src python benchmarks/bench_circuit_breaker.py

Output:
    benchmarks/results/circuit_breaker_reliability.json
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RESULTS_DIR = ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TransitionRecord:
    from_state: str
    to_state: str
    at_ms: float
    call_number: int


@dataclass
class ScenarioResult:
    name: str
    description: str
    transitions: list[TransitionRecord]
    calls_rejected: int
    calls_allowed: int
    time_to_open_ms: float | None
    time_in_open_ms: float | None
    calls_to_close: int
    false_positive_trip: bool
    final_state: str


async def _scenario_a_gradual(failure_threshold: int, recovery_timeout_s: float) -> ScenarioResult:
    """Phase 1: 20 healthy calls. Phase 2: inject failures one-by-one. Phase 3: recover."""
    from harness.core.circuit_breaker import CircuitBreaker, CircuitState
    from harness.core.errors import CircuitOpenError

    transitions: list[TransitionRecord] = []
    call_number = 0
    calls_rejected = 0
    calls_allowed = 0
    t_first_failure: float | None = None
    t_opened: float | None = None
    t_closed_again: float | None = None
    calls_to_close = 0
    false_positive_trip = False
    open_detected = False

    def on_change(name, old, new):
        nonlocal t_opened
        ms = time.monotonic() * 1000
        transitions.append(TransitionRecord(old.value, new.value, ms, call_number))
        if new == CircuitState.OPEN:
            t_opened = ms

    cb = CircuitBreaker(
        name="scenario-a",
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout_s,
        success_threshold=2,
        on_state_change=on_change,
    )

    t_start = time.monotonic() * 1000

    # Phase 1: 20 healthy calls
    for _ in range(20):
        call_number += 1
        if await cb.can_proceed():
            calls_allowed += 1
            await cb.record_success()
        else:
            calls_rejected += 1
            false_positive_trip = True  # tripped on healthy traffic

    state_after_healthy = cb.state.value

    # Phase 2: inject failures one by one until circuit opens
    for i in range(failure_threshold + 2):
        call_number += 1
        if await cb.can_proceed():
            calls_allowed += 1
            if t_first_failure is None:
                t_first_failure = time.monotonic() * 1000
            await cb.record_failure()
        else:
            calls_rejected += 1
            if not open_detected:
                open_detected = True
        if cb.state == CircuitBreaker.__class__ or True:  # always check
            pass

    time_to_open_ms = (
        (t_opened - t_first_failure) if t_opened and t_first_failure else None
    )

    # Phase 3: wait for recovery timeout, then let 2 successes close it
    t_before_wait = time.monotonic() * 1000
    await asyncio.sleep(recovery_timeout_s + 0.01)
    t_after_wait = time.monotonic() * 1000
    time_in_open_ms = t_after_wait - t_before_wait

    # Probe call (HALF_OPEN)
    for _ in range(5):
        call_number += 1
        if await cb.can_proceed():
            calls_allowed += 1
            calls_to_close += 1
            await cb.record_success()
        else:
            calls_rejected += 1
        if cb.state.value == "CLOSED":
            break

    return ScenarioResult(
        name="A_gradual_degradation",
        description="20 healthy → ramp failures → recover",
        transitions=transitions,
        calls_rejected=calls_rejected,
        calls_allowed=calls_allowed,
        time_to_open_ms=round(time_to_open_ms, 2) if time_to_open_ms else None,
        time_in_open_ms=round(time_in_open_ms, 2),
        calls_to_close=calls_to_close,
        false_positive_trip=false_positive_trip,
        final_state=cb.state.value,
    )


async def _scenario_b_burst(failure_threshold: int, recovery_timeout_s: float) -> ScenarioResult:
    """5 consecutive failures immediately trip the circuit."""
    from harness.core.circuit_breaker import CircuitBreaker, CircuitState
    from harness.core.errors import CircuitOpenError

    transitions: list[TransitionRecord] = []
    call_number = 0
    calls_rejected = 0
    calls_allowed = 0
    t_opened: float | None = None
    t_first_failure: float | None = None

    def on_change(name, old, new):
        nonlocal t_opened
        transitions.append(TransitionRecord(old.value, new.value, time.monotonic() * 1000, call_number))
        if new == CircuitState.OPEN:
            t_opened = time.monotonic() * 1000

    cb = CircuitBreaker(
        name="scenario-b",
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout_s,
        success_threshold=2,
        on_state_change=on_change,
    )

    # Burst: failure_threshold failures in a row
    for _ in range(failure_threshold):
        call_number += 1
        if await cb.can_proceed():
            calls_allowed += 1
            if t_first_failure is None:
                t_first_failure = time.monotonic() * 1000
            await cb.record_failure()
        else:
            calls_rejected += 1

    # Try 3 more calls — all should be rejected
    for _ in range(3):
        call_number += 1
        if await cb.can_proceed():
            calls_allowed += 1
            await cb.record_success()
        else:
            calls_rejected += 1

    state_after_burst = cb.state.value

    # Wait for recovery
    await asyncio.sleep(recovery_timeout_s + 0.01)

    # Recover with successes
    calls_to_close = 0
    for _ in range(5):
        call_number += 1
        if await cb.can_proceed():
            calls_allowed += 1
            calls_to_close += 1
            await cb.record_success()
        else:
            calls_rejected += 1
        if cb.state.value == "CLOSED":
            break

    time_to_open_ms = (t_opened - t_first_failure) if t_opened and t_first_failure else None

    return ScenarioResult(
        name="B_burst_failure",
        description=f"{failure_threshold} consecutive failures → immediate trip → recover",
        transitions=transitions,
        calls_rejected=calls_rejected,
        calls_allowed=calls_allowed,
        time_to_open_ms=round(time_to_open_ms, 2) if time_to_open_ms else None,
        time_in_open_ms=round(recovery_timeout_s * 1000, 0),
        calls_to_close=calls_to_close,
        false_positive_trip=False,
        final_state=cb.state.value,
    )


async def _scenario_c_intermittent(failure_threshold: int) -> ScenarioResult:
    """Failures never hit threshold — circuit should never trip (no false positive)."""
    from harness.core.circuit_breaker import CircuitBreaker

    transitions: list[TransitionRecord] = []
    call_number = 0
    calls_rejected = 0
    calls_allowed = 0
    false_positive_trip = False

    def on_change(name, old, new):
        transitions.append(TransitionRecord(old.value, new.value, time.monotonic() * 1000, call_number))
        nonlocal false_positive_trip
        false_positive_trip = True  # any transition here is a false positive

    cb = CircuitBreaker(
        name="scenario-c",
        failure_threshold=failure_threshold,
        recovery_timeout=0.05,
        success_threshold=2,
        on_state_change=on_change,
    )

    # Pattern: (threshold-1) failures then a success, repeated 10 times
    for cycle in range(10):
        for _ in range(failure_threshold - 1):
            call_number += 1
            if await cb.can_proceed():
                calls_allowed += 1
                await cb.record_failure()
            else:
                calls_rejected += 1
        # Success resets the failure count
        call_number += 1
        if await cb.can_proceed():
            calls_allowed += 1
            await cb.record_success()
        else:
            calls_rejected += 1

    return ScenarioResult(
        name="C_intermittent_failure",
        description=f"{failure_threshold-1} failures + 1 success × 10 cycles — circuit must stay CLOSED",
        transitions=transitions,
        calls_rejected=calls_rejected,
        calls_allowed=calls_allowed,
        time_to_open_ms=None,
        time_in_open_ms=None,
        calls_to_close=0,
        false_positive_trip=false_positive_trip,
        final_state=cb.state.value,
    )


async def run() -> None:
    FAILURE_THRESHOLD = 5
    RECOVERY_TIMEOUT_S = 0.05   # 50 ms for fast benchmark; README default is 60 s

    print(f"\nCircuit Breaker Reliability Benchmark")
    print(f"  failure_threshold  = {FAILURE_THRESHOLD}")
    print(f"  recovery_timeout   = {RECOVERY_TIMEOUT_S * 1000:.0f} ms (production default: 60 000 ms)")
    print(f"  success_threshold  = 2\n")

    results = []
    for scenario_fn, kwargs in [
        (_scenario_a_gradual, {"failure_threshold": FAILURE_THRESHOLD, "recovery_timeout_s": RECOVERY_TIMEOUT_S}),
        (_scenario_b_burst,   {"failure_threshold": FAILURE_THRESHOLD, "recovery_timeout_s": RECOVERY_TIMEOUT_S}),
        (_scenario_c_intermittent, {"failure_threshold": FAILURE_THRESHOLD}),
    ]:
        r = await scenario_fn(**kwargs)
        results.append(r)
        print(f"Scenario {r.name}")
        print(f"  {r.description}")
        print(f"  Final state        : {r.final_state}")
        print(f"  Calls allowed      : {r.calls_allowed}")
        print(f"  Calls rejected     : {r.calls_rejected}")
        print(f"  Time-to-OPEN       : {r.time_to_open_ms} ms")
        print(f"  Time-in-OPEN       : {r.time_in_open_ms} ms")
        print(f"  Calls to re-CLOSE  : {r.calls_to_close}")
        print(f"  False positive trip: {r.false_positive_trip}")
        print(f"  State transitions  : {[(t.from_state, t.to_state) for t in r.transitions]}")
        print()

    output = {
        "benchmark": "circuit_breaker_reliability",
        "config": {
            "failure_threshold": FAILURE_THRESHOLD,
            "recovery_timeout_ms": RECOVERY_TIMEOUT_S * 1000,
            "success_threshold": 2,
        },
        "scenarios": [
            {
                "name": r.name,
                "description": r.description,
                "final_state": r.final_state,
                "calls_allowed": r.calls_allowed,
                "calls_rejected": r.calls_rejected,
                "time_to_open_ms": r.time_to_open_ms,
                "time_in_open_ms": r.time_in_open_ms,
                "calls_to_close": r.calls_to_close,
                "false_positive_trip": r.false_positive_trip,
                "transitions": [
                    {"from": t.from_state, "to": t.to_state, "at_ms": t.at_ms, "call": t.call_number}
                    for t in r.transitions
                ],
            }
            for r in results
        ],
    }
    out_path = RESULTS_DIR / "circuit_breaker_reliability.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    asyncio.run(run())
