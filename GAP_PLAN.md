# HarnessAgent — Gap Plan

## Runtime

- LLM retry with exponential backoff (1s → 2s → 4s ±20% jitter) per provider before fallback
- Health check TTL cache (5s) — avoid per-request HTTP health checks on every LLM call
- Semantic cache exact-hash fast path + scan cap (200 entries) — fixes O(N) linear scan
- Tool result size cap (8k chars) — truncate large tool outputs before they enter agent history

## Eval

- `EvalSandbox` protocol with four backends: SQL, Code (wraps DockerSandbox), ToolCall, HTTP
- `score_execution_match` — run action + gold action in sandbox, compare outputs
- `score_execution_best_of` — evaluate against multiple valid gold actions, take best
- `score_output_match` — exact / subset / numeric-close structured output comparison
- `score_row_count_match` — tabular row count ratio
- `score_schema_match` — fraction of gold columns/keys present in predicted output
- `FailureCategory` enum — 21 named failure categories covering output, tool, planning, safety, retrieval
- `FailureAnalysis` — primary category + evidence dict + `top_hint()` actionable fix string
- `classify_failure()` — detect primary failure from output, scores, trace, schema names
- `classify_task_hardness()` — easy / medium / hard / extra-hard for any task
- `classify_sql_hardness()` — SQL-specific hardness via sqlglot AST
- `detect_nondeterministic()` — flag SQL/code actions that produce non-repeatable results
- `EvalCase.gold_actions` — multiple valid actions per case (multi-gold)
- `EvalCase.sandbox_type` — sql / code / tool / http / none
- `EvalCase.db_path`, `EvalCase.hardness` — per-case fixtures and difficulty label
- `AgentScores` — three-dimension score (correctness / quality / safety) with per-dimension thresholds; PASS requires all three
- `evaluate_agent_output()` — produces `AgentScores` for any agent action
- `EvalJudgeCache` — in-memory SHA-256 cache for LLM judge calls; opt-in per run
- `load_jsonl()`, `load_csv()` — generic eval dataset loaders
- `load_spider()`, `load_bird()` — SQL benchmark loaders
- `load_humaneval()`, `load_gsm8k()` — code / reasoning benchmark loaders
- `AgentEvalReport` — by_hardness, by_dimension, failure_distribution, to_markdown, to_json
- `generate_report()` — markdown or JSON report from a list of AgentScores
- `log_all()` — log eval results to MLflow, W&B, and LangSmith (each optional)
