# GEPA Integration Plan — HarnessAgent (Hermes optimizer upgrade)

**Status:** Proposed (review before build)
**Date:** 2026-05-31
**Backend:** standalone [`gepa`](https://github.com/gepa-ai/gepa) package (not DSPy)
**Target repo:** `/Users/pradip/Desktop/Learning/Codex/HarnessAgent` (`agent-haas`)

> Supersedes the earlier draft written against `Medical_graph_rag` — that repo lacked a prompt registry and an optimization loop, so it needed both built. HarnessAgent already has them. This plan is mostly **reuse + one new strategy**, not greenfield.

---

## 0. Verdict

**Does HarnessAgent need GEPA?** It already has the *function* (Hermes self-improvement). GEPA is a principled **replacement for the weakest link in that loop — `PatchGenerator`** — reusing everything else. Worth doing because the current generator is the naïve version of exactly what GEPA does well.

**One sentence:** add a `GepaPatchGenerator` (and Pareto candidate selection) behind the existing `HermesLoop` interface; reuse `PromptStore`, `EvalRunner`, `scorers`, `failure_taxonomy`, rollback, and auto-apply gate unchanged.

---

## 1. What Hermes does today (grounded)

`improvement/hermes.py::HermesLoop.run_cycle(agent_type)`:
1. Rollback check on last auto-applied patch.
2. Sample errors from `ErrorCollector`; split `generation_errors` / held-out validation (`hermes.py:145-163`).
3. Route by **dominant failure class** → specialized generator (`generate_retry_patch` / `generate_permission_patch` / `generate_tool_patch`), else fall back to prompt patch (`hermes.py:187-230`).
4. `PatchGenerator.generate()` → **one** `Patch{op: append|prepend|replace, path, value}` from an LLM analyzing error records (`patch_generator.py:129-174`).
5. `Evaluator` replays held-out failing tasks → score.
6. If `score > threshold (0.7)` and `auto_apply` → `prompt_store.apply_patch`; else hold/rollback (`hermes.py:89`, `:48-50`).

**Limits this plan targets:**
- Single greedy candidate per cycle (no population, no search).
- Failure-class routing (`:210-229`) is a manual workaround for single-objective collapse.
- String-surgery ops can't restructure a prompt.
- No memory of which mutations historically helped.

---

## 2. What GEPA adds (and what it reuses)

| Concern | Reuse (exists) | GEPA adds |
|---|---|---|
| Prompt storage/versioning | `prompts/store.py`, `prompts/manager.py` (`PromptVersion`, `set_active`, `apply_patch`) | writes evolved prompts as new `PromptVersion`s with `metadata.gepa_run_id` |
| Metric | `eval/runner.py::EvalRunner`, `eval/scorers.py` (`score_sql_correctness`, similarity, exact) | scalarized multi-metric score per candidate |
| Reflection feedback | `eval/failure_taxonomy.py`, `eval/diagnostics.py`, hierarchical spans | packages per-case failure category + diagnostic text as GEPA reflection input |
| Candidate selection | (Hermes: accept/reject one) | **Pareto front** across eval cases — replaces failure-class routing |
| Loop / rollback / gate | `HermesLoop` cycle, threshold, rollback | runs as the generation strategy *inside* the cycle |
| Datasets | `eval/benchmark_loaders.py` (BIRD, TauBench, GAiA), `EvalDataset` JSONL | train/val split for evolution vs promotion gate |

Net new code is small and isolated.

---

## 3. Design

### 3.1 New module
```
src/harness/improvement/gepa/
├── __init__.py
├── generator.py        # GepaPatchGenerator — same interface as PatchGenerator
├── adapter.py          # HarnessGEPAAdapter — bridges gepa <-> EvalRunner
├── feedback.py         # failure_taxonomy + diagnostics -> reflection text
└── config.py           # budget, train/val split, reflection model, weights
```

### 3.2 Drop-in generator (interface-compatible)
`GepaPatchGenerator` exposes the same surface `HermesLoop` already calls, so the loop, rollback, and gate are untouched:

```python
class GepaPatchGenerator:
    """GEPA-backed replacement for PatchGenerator. Returns a Patch the
    existing HermesLoop can apply/rollback exactly as today."""

    async def generate(self, agent_type, errors, current_prompt) -> Patch | None:
        seed = {"system_prompt": current_prompt}
        result = gepa.optimize(
            seed_candidate=seed,
            adapter=HarnessGEPAAdapter(self._eval_runner, self._scorers, agent_type),
            trainset=self._to_eval_cases(errors),          # from sampled failures
            valset=self._holdout,                            # Hermes' held-out split
            reflection_lm=self._reflection_model,
            max_metric_calls=self._cfg.budget,
        )
        best = result.best_candidate["system_prompt"]
        # Express the evolved prompt as an op the existing store understands:
        return Patch(agent_type=agent_type, target="prompt", op="set", value=best,
                     metadata={"gepa_run_id": result.run_id, "scores": result.val_scores})
```

> `op="set"` (full replace) is already in the `Patch` op enum (`patch_generator.py:26`) and `PromptStore` versioning — so evolved whole-prompts flow through the existing apply/rollback path with **zero changes** to the store.

### 3.3 Adapter — the only real bridge code
```python
class HarnessGEPAAdapter(GEPAAdapter):
    def evaluate(self, batch, candidate, capture_traces=False):
        # swap candidate prompt for this eval run, then reuse EvalRunner
        report = self._eval_runner.run(
            dataset=batch, agent_type=self._agent_type,
            prompt_override=candidate["system_prompt"],
        )
        scores   = [self._scalarize(c) for c in report.cases]      # from scorers
        feedback = [build_feedback(c) for c in report.cases]       # feedback.py
        return EvaluationBatch(outputs=report.cases, scores=scores, feedback=feedback)
```
`build_feedback(case)` = `failure_taxonomy.top_hint(case)` + diagnostic stage + scorer reason → the natural-language critique GEPA reflects on. This is the piece HarnessAgent is unusually well-equipped for.

### 3.4 Wiring into HermesLoop
One config switch, no behavior change unless enabled:
```python
generator = (
    GepaPatchGenerator(...) if config.hermes_strategy == "gepa"
    else PatchGenerator(...)            # current default, untouched
)
HermesLoop(generator=generator, evaluator=..., prompt_store=..., ...)
```
`prompt_override` is the only `EvalRunner` addition needed (thread a candidate prompt through a run without mutating the store). If that hook doesn't exist yet, it's a small, well-contained change.

### 3.5 Pareto replaces failure-class routing
GEPA keeps a Pareto front over eval cases, so a candidate that fixes `TOOL_LIMIT_EXCEEDED` without regressing `sql/join` survives on its own merits. The manual `dominant in {timeout,safety,tool,prompt}` routing (`hermes.py:210-229`) becomes redundant for the prompt path — keep the specialized non-prompt generators (retry/permission/tool config) as-is; let GEPA own the **prompt** patches.

---

## 4. Cost / safety / risk

- **Offline only** — runs inside the Hermes background cycle (`workers/`), never in a live agent run. No runtime latency change.
- **Budget** — `max_metric_calls` + subsampled trainset (Hermes already samples). Dry-run mode: 5 cases, 1 iteration, before spending.
- **Gate unchanged** — evolved prompts still pass the `> threshold` validation gate + rollback before `set_active`. Default `auto_apply` stays human-in-the-loop.
- **MLflow** — log GEPA rollouts under the existing tracking (the repo already wires MLflow into eval).
- **Async safety** — `prompt_override` via param, not shared mutation; GEPA runs one agent_type per cycle.
- **Dependency** — add `gepa` (pure-Python, light) to `pyproject.toml` optional group `[improvement]`. No DSPy.

---

## 5. Validation (does the upgrade actually beat Hermes?)

| Exp | Comparison | Dataset | Metric | Claim |
|---|---|---|---|---|
| V1 | `PatchGenerator` (current) vs `GepaPatchGenerator` | BIRD (sql) via `benchmark_loaders` | success_rate, regression count | GEPA lifts score with fewer regressions |
| V2 | greedy vs Pareto on mixed failure classes | TauBench / AgencyBench | per-class success | Pareto avoids the cross-class regressions routing was hiding |
| V3 | rollout efficiency | any | score vs # metric calls | GEPA reaches Hermes' score in fewer rollouts |
| V4 | feedback ablation | any | convergence | rich `failure_taxonomy` feedback > scalar-only reward |

V2 + V4 are the strongest stories: they directly justify GEPA over the existing loop using HarnessAgent's own infra.

---

## 6. Build sequence

1. Add `gepa` dep (optional group) + `improvement/gepa/` skeleton.
2. `EvalRunner` `prompt_override` hook (if absent) + unit test.
3. `feedback.py` from `failure_taxonomy`/`diagnostics`.
4. `adapter.py` + `generator.py`; dry-run (5 cases, 1 iter).
5. Wire `config.hermes_strategy == "gepa"` switch.
6. V1 on BIRD; review evolved prompt by hand before any auto-apply.
7. V2–V4.

**Estimate:** ~2–3 days to first real run (most infra exists). vs the medical repo, which needed ~1 day of registry work *before* any of this was possible.

---

## 7. Open questions

- Let GEPA own **only** prompt patches, leaving retry/permission/tool-config patches to the existing specialized generators? (Recommend yes — clean separation.)
- Per-`agent_type` GEPA runs (sql, code, research, orchestrator) independently, or a joint multi-prompt compound optimization? (Start per-type; compound is a strong follow-up — it's GEPA's signature strength.)
- Keep `auto_apply` human-gated for GEPA-evolved whole-prompt rewrites even if heuristic patches are auto-applied? (Recommend yes — full rewrites are higher-variance than append ops.)
- Does `EvalRunner` already support a non-persisted `prompt_override`, or is that the one real core change needed?
