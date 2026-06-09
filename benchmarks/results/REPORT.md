# HarnessAgent Benchmark Report
_Generated 2026-06-09 20:50 UTC · Python 3.11.4 · Darwin arm64_

**System benchmarks** below are deterministic and reproducible with `PYTHONPATH=src python benchmarks/run_all.py` — no API keys or datasets required (Redis is faked in-process). **Task/accuracy benchmarks** require live LLM calls and dataset downloads; their commands are listed at the end and must be run with your own keys/data.

## System benchmarks (reproducible)

### GraphRAG token efficiency  (✅ fresh, 0.4s)

_Measures schema-token reduction vs naive vector retrieval on a synthetic 10-table schema._

| Metric | Value |
|---|---|
| Naive vector-search avg tokens | 2208.8 |
| GraphRAG avg tokens | 378.1 |
| Token savings | 82.9% |
| Schema-table coverage | 100.0% |
| Queries | 20 over 10 tables |

### Semantic cache hit-rate  (✅ fresh, 12.4s)

_Real all-MiniLM-L6-v2 embeddings over curated near-exact/rephrase/paraphrase/decoy query tiers._

| Metric | Value |
|---|---|
| Embedder | all-MiniLM-L6-v2 (real=True) |
| Default threshold | 0.97 |
| Near-exact hit-rate @default | 100% |
| Minor-rephrase hit-rate @default | 20% |
| Decoy false-positive @default | 0% |
| Lookup p50 @default | 8.76 ms |
| Best zero-FP threshold | 0.99 (near-exact 40%) |

### Circuit-breaker reliability  (✅ fresh, 0.1s)

_Failure-injection scenarios against the real CircuitBreaker; verifies it opens under failure and recovers._

| Metric | Value |
|---|---|
| A_gradual_degradation | opened in 0.01 ms → final CLOSED |
| B_burst_failure | opened in 0.01 ms → final CLOSED |
| C_intermittent_failure | did not open (final CLOSED) |

### Span recording overhead  (✅ fresh, 6.5s)

_Wall-clock cost the TraceRecorder adds over a bare async no-op (fakeredis backend; real Redis adds network)._

| Metric | Value |
|---|---|
| Iterations / backend | 2000 / fakeredis |
| Baseline p50 | 0.1 µs |
| start_span+end_span net p50 | 899.1 µs |
| Context-manager net p50 | 921.5 µs |
| set_llm_usage net p50 | 0.0 µs |

### Hermes self-improvement (synthetic)  (✅ fresh, 0.0s)

_SYNTHETIC loop-mechanics demo: mock LLM + mock agent runner modeling a 10%→95% golden fix. Exercises detect→propose→evaluate→auto-apply→converge; the improvement magnitude is a property of the model, NOT a measured LLM gain. Real self-improvement: run bench_hermes_real.py with a live LLM._

| Metric | Value |
|---|---|
| Seed failures | 10 |
| Pre-patch pass@1 | 5.0% |
| Post-patch pass@1 | 100.0% |
| Improvement | +95.0 pp |
| Cycles run | 5 |
| Converged at cycle | 1 |
| Patches applied | 3 |

## Task / accuracy benchmarks (require API keys + datasets)

Not run by this script — run individually with your own credentials and datasets. Prior result JSONs (if any) live in `benchmarks/results/` but are only valid for the exact model + dataset version they were produced with.

| Benchmark | Command (prefix `PYTHONPATH=src python benchmarks/`) |
|---|---|
| SQL exec accuracy (BIRD) | `bench_agent.py --agent sql --dataset bird --data-dir <path>` |
| Code pass@1 (HumanEval) | `bench_agent.py --agent code --dataset humaneval --data-dir <path>` |
| Reasoning (GSM8K) | `bench_agent.py --agent base --dataset gsm8k --data-dir <path>` |
| BIRD (live LLM) | `bench_bird_real.py --data-dir <path>` |
| Hermes (live LLM) | `bench_hermes_real.py` |
| Safety trajectories (tau-bench) | `bench_taubench.py` |
| AgencyBench ablation | `bench_agencybench_ablation.py` |
