"""Scoring functions for evaluating agent output quality."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Execution-based scorers (require an EvalSandbox)
# ---------------------------------------------------------------------------


async def score_execution_match(
    action: str,
    gold_action: str,
    sandbox: Any,
    **exec_kwargs: Any,
) -> ScoreResult:
    """Run action and gold_action in sandbox; compare outputs. 1.0 if equal, 0.0 otherwise."""
    try:
        pred_res = await sandbox.execute(action, **exec_kwargs)
        gold_res = await sandbox.execute(gold_action, **exec_kwargs)
    except Exception as exc:
        return ScoreResult(score=0.0, method="execution_match",
                           details=f"sandbox error: {exc}")

    if pred_res.error:
        return ScoreResult(score=0.0, method="execution_match",
                           details=f"pred error: {pred_res.error}")
    if gold_res.error:
        return ScoreResult(score=0.5, method="execution_match",
                           details="gold action failed — cannot compare")

    pred_out = pred_res.output
    gold_out = gold_res.output

    # Tabular comparison: normalise rows to sets of tuples
    if isinstance(pred_out, dict) and "rows" in pred_out and isinstance(gold_out, dict) and "rows" in gold_out:
        pred_rows = {tuple(r) for r in pred_out.get("rows", [])}
        gold_rows = {tuple(r) for r in gold_out.get("rows", [])}
        if pred_rows == gold_rows:
            return ScoreResult(score=1.0, method="execution_match", details="result sets equal")
        if not gold_rows:
            # A non-empty prediction against an empty gold is wrong, not perfect.
            return ScoreResult(score=0.0, method="execution_match", details="gold result is empty")
        inter = len(pred_rows & gold_rows)
        # Symmetric (Jaccard-style) overlap so a *superset* of gold cannot score
        # 1.0 — recall-only would reward-hack by returning extra rows.
        denom = max(len(pred_rows), len(gold_rows))
        overlap = inter / denom
        return ScoreResult(score=round(overlap, 4), method="execution_match",
                           details=f"partial overlap {inter}/{denom} "
                                   f"(pred={len(pred_rows)} gold={len(gold_rows)})")

    # Generic equality
    match = str(pred_out).strip() == str(gold_out).strip()
    return ScoreResult(score=1.0 if match else 0.0, method="execution_match",
                       details="exact match" if match else "outputs differ")


async def score_execution_best_of(
    action: str,
    gold_actions: list[str],
    sandbox: Any,
    **exec_kwargs: Any,
) -> ScoreResult:
    """Evaluate action against all gold actions; return the best score."""
    if not gold_actions:
        return ScoreResult(score=0.0, method="execution_best_of", details="no gold actions")
    best = ScoreResult(score=0.0, method="execution_best_of", details="no match")
    for gold in gold_actions:
        result = await score_execution_match(action, gold, sandbox, **exec_kwargs)
        if result.score > best.score:
            best = ScoreResult(score=result.score, method="execution_best_of",
                               details=result.details)
        if best.score == 1.0:
            break
    return best


def score_output_match(
    pred_output: Any,
    gold_output: Any,
    match_type: str = "exact",
    tolerance: float = 0.01,
) -> ScoreResult:
    """Compare structured outputs: exact / subset / numeric_close."""
    if match_type == "exact":
        match = str(pred_output).strip() == str(gold_output).strip()
        return ScoreResult(score=1.0 if match else 0.0, method="output_match_exact")

    if match_type == "subset":
        if isinstance(gold_output, dict) and isinstance(pred_output, dict):
            match = all(pred_output.get(k) == v for k, v in gold_output.items())
            return ScoreResult(score=1.0 if match else 0.0, method="output_match_subset")
        if isinstance(gold_output, list) and isinstance(pred_output, list):
            gold_set = set(map(str, gold_output))
            pred_set = set(map(str, pred_output))
            score = len(gold_set & pred_set) / len(gold_set) if gold_set else 1.0
            return ScoreResult(score=score, method="output_match_subset")
        return score_output_match(pred_output, gold_output, "exact")

    if match_type == "numeric_close":
        try:
            p, g = float(pred_output), float(gold_output)
            denom = max(abs(g), 1.0)
            rel_err = abs(p - g) / denom
            return ScoreResult(
                score=1.0 if rel_err <= tolerance else 0.0,
                method="output_match_numeric",
                details=f"rel_err={rel_err:.4f}",
            )
        except (TypeError, ValueError):
            return ScoreResult(score=0.0, method="output_match_numeric",
                               details="non-numeric values")

    return ScoreResult(score=0.0, method="output_match", details=f"unknown match_type={match_type}")


def score_row_count_match(pred_result: Any, gold_result: Any) -> ScoreResult:
    """Ratio score: min/max row count between predicted and gold results."""
    pred_n = _extract_row_count(pred_result)
    gold_n = _extract_row_count(gold_result)
    if gold_n == 0 and pred_n == 0:
        return ScoreResult(score=1.0, method="row_count_match", details="both empty")
    if gold_n == 0 or pred_n == 0:
        return ScoreResult(score=0.0, method="row_count_match",
                           details=f"pred={pred_n} gold={gold_n}")
    score = min(pred_n, gold_n) / max(pred_n, gold_n)
    return ScoreResult(score=round(score, 4), method="row_count_match",
                       details=f"pred={pred_n} gold={gold_n}")


def score_schema_match(pred_result: Any, gold_result: Any) -> ScoreResult:
    """Fraction of gold column/key names present in predicted result."""
    pred_cols = _extract_columns(pred_result)
    gold_cols = _extract_columns(gold_result)
    if not gold_cols:
        return ScoreResult(score=1.0, method="schema_match", details="gold has no columns")
    overlap = len({c.lower() for c in pred_cols} & {c.lower() for c in gold_cols})
    score = overlap / len(gold_cols)
    return ScoreResult(score=round(score, 4), method="schema_match",
                       details=f"{overlap}/{len(gold_cols)} columns matched")


def _extract_row_count(result: Any) -> int:
    if hasattr(result, "output") and isinstance(result.output, dict):
        return int(result.output.get("row_count", len(result.output.get("rows", []))))
    if isinstance(result, dict):
        return int(result.get("row_count", len(result.get("rows", []))))
    if isinstance(result, list):
        return len(result)
    return 0


def _extract_columns(result: Any) -> list[str]:
    if hasattr(result, "output") and isinstance(result.output, dict):
        return result.output.get("columns", [])
    if isinstance(result, dict):
        return result.get("columns", list(result.keys()))
    return []


@dataclass
class ScoreResult:
    """Container for a scorer's output.

    Attributes:
        score:   Float in [0.0, 1.0]; higher is better.
        method:  Name of the scoring method used.
        details: Optional human-readable explanation of the score.
    """

    score: float
    method: str
    details: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            self.score = max(0.0, min(1.0, self.score))


# ---------------------------------------------------------------------------
# Deterministic scorers
# ---------------------------------------------------------------------------


_NUMBER_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")


def _as_number(text: str) -> float | None:
    """Parse *text* as a single number (commas allowed), else None."""
    s = text.strip().replace(",", "")
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def score_exact_match(output: str, expected: str) -> float:
    """Return 1.0 if *expected* matches *output* (case-insensitive).

    For numeric expected values the *final* number in the output is extracted
    and compared numerically — plain substring containment would let "142"
    satisfy an expected "42" (GSM8K-style inflation). For non-numeric expected
    values the match is anchored on word boundaries so "42" does not match
    inside "142".

    Args:
        output:   The agent's actual output text.
        expected: The expected answer / substring to search for.

    Returns:
        1.0 if expected is found, 0.0 otherwise.
    """
    if not expected:
        return 1.0

    expected_num = _as_number(expected)
    if expected_num is not None:
        # Compare against the LAST number emitted (the conventional final answer).
        matches = _NUMBER_RE.findall(output)
        if not matches:
            return 0.0
        final = _as_number(matches[-1])
        if final is None:
            return 0.0
        return 1.0 if final == expected_num else 0.0

    # Non-numeric: anchor on word boundaries to avoid spurious substring hits
    # (e.g. expected "class" matching inside "classification").
    stripped = expected.strip()
    edges_are_word_chars = bool(stripped) and (
        stripped[0].isalnum() or stripped[0] == "_"
    ) and (stripped[-1].isalnum() or stripped[-1] == "_")
    if edges_are_word_chars:
        pattern = re.compile(r"\b" + re.escape(stripped) + r"\b", re.IGNORECASE)
        return 1.0 if pattern.search(output) else 0.0
    # Edges are punctuation/symbols where \b can never anchor — plain containment.
    return 1.0 if expected.lower() in output.lower() else 0.0


def score_contains_all(output: str, expected_keywords: list[str]) -> float:
    """Return the fraction of keywords present in *output* (case-insensitive).

    Args:
        output:            The agent's actual output text.
        expected_keywords: List of keywords to check for.

    Returns:
        Float in [0.0, 1.0] representing how many keywords were found.
    """
    if not expected_keywords:
        return 1.0
    output_lower = output.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in output_lower)
    return found / len(expected_keywords)


def score_success_rate(results: list[Any]) -> float:
    """Return the fraction of AgentResult objects with success=True.

    Args:
        results: List of AgentResult (or any object with a .success bool).

    Returns:
        Float in [0.0, 1.0].
    """
    if not results:
        return 0.0
    successes = sum(1 for r in results if getattr(r, "success", False))
    return successes / len(results)


# ---------------------------------------------------------------------------
# SQL equivalence scorer
# ---------------------------------------------------------------------------


def _normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison: uppercase keywords, collapse whitespace."""
    sql = sql.strip()
    # Collapse all whitespace to single space
    sql = re.sub(r"\s+", " ", sql)
    # Uppercase SQL keywords
    keywords = [
        "SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY", "HAVING",
        "JOIN", "LEFT JOIN", "RIGHT JOIN", "INNER JOIN", "OUTER JOIN",
        "ON", "AND", "OR", "NOT", "IN", "IS", "NULL", "LIKE",
        "COUNT", "SUM", "AVG", "MIN", "MAX", "DISTINCT",
        "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER",
        "AS", "BY", "LIMIT", "OFFSET", "UNION", "ALL", "EXISTS",
    ]
    for kw in keywords:
        sql = re.sub(r"\b" + kw + r"\b", kw, sql, flags=re.IGNORECASE)
    return sql.strip()


def score_sql_equivalence(sql1: str, sql2: str) -> float:
    """Compare two SQL strings for semantic equivalence.

    Uses sqlparse for AST-level comparison when available, falling back to
    normalized string comparison.

    Scoring:
        1.0  — Exactly equivalent (same normalized form or same AST tokens).
        0.5  — Similar structure (same statement type and overlapping tokens).
        0.0  — Different or incomparable.

    Args:
        sql1: First SQL string.
        sql2: Second SQL string.

    Returns:
        Float in {0.0, 0.5, 1.0}.
    """
    if not sql1 or not sql2:
        return 0.0

    norm1 = _normalize_sql(sql1)
    norm2 = _normalize_sql(sql2)

    if norm1 == norm2:
        return 1.0

    try:
        import sqlparse  # type: ignore

        parsed1 = sqlparse.parse(sql1)
        parsed2 = sqlparse.parse(sql2)

        if not parsed1 or not parsed2:
            return 0.0

        stmt1 = parsed1[0]
        stmt2 = parsed2[0]

        # Compare statement type
        type1 = stmt1.get_type()
        type2 = stmt2.get_type()
        if type1 != type2:
            return 0.0

        # Compare flattened token values (stripped of whitespace)
        def token_values(stmt) -> list[str]:
            return [
                t.normalized.upper()
                for t in stmt.flatten()
                if not t.is_whitespace
            ]

        tokens1 = token_values(stmt1)
        tokens2 = token_values(stmt2)

        if tokens1 == tokens2:
            return 1.0

        # Calculate token overlap
        set1 = set(tokens1)
        set2 = set(tokens2)
        if not set1 or not set2:
            return 0.0

        overlap = len(set1 & set2) / max(len(set1), len(set2))
        if overlap >= 0.8:
            return 0.5

        return 0.0

    except ImportError:
        logger.debug("sqlparse not available; falling back to string comparison")

    # Fallback: word-overlap similarity
    words1 = set(norm1.split())
    words2 = set(norm2.split())
    if not words1 or not words2:
        return 0.0
    overlap = len(words1 & words2) / max(len(words1), len(words2))
    if overlap >= 0.9:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# LLM-based judge scorer
# ---------------------------------------------------------------------------

_JUDGE_PROMPT_TEMPLATE = """\
You are an evaluation judge for an AI agent system.

## Task
{task}

## Agent Output
{output}

## Expected Output (may be null)
{expected}

## Rubric
{rubric}

Evaluate the agent output on a scale from 0.0 to 1.0, where:
- 1.0 = Perfect: fully correct, complete, and well-reasoned
- 0.8 = Good: mostly correct with minor issues
- 0.6 = Acceptable: partially correct, some important parts missing
- 0.4 = Poor: shows some understanding but significant problems
- 0.2 = Very poor: mostly incorrect but shows minimal relevance
- 0.0 = Completely wrong or irrelevant

Respond ONLY with a JSON object in this exact format:
{{"score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}}
"""

_DEFAULT_RUBRIC = (
    "Evaluate correctness, completeness, and relevance to the task. "
    "If expected output is provided, check that the key content is present."
)


async def score_llm_judge(
    task: str,
    output: str,
    expected: str | None,
    llm_provider: Any,
    rubric: str | None = None,
    cache: Any | None = None,
) -> ScoreResult:
    """Use an LLM to evaluate agent output quality on a 0-1 scale.

    Constructs a structured judge prompt, calls the LLM provider, and parses
    the response JSON to extract score and reasoning.

    Args:
        task:         The original task given to the agent.
        output:       The agent's output to be evaluated.
        expected:     Optional ground-truth expected output (can be None).
        llm_provider: An LLMProvider-compatible object with a .complete() method.
        rubric:       Optional custom evaluation rubric.  Falls back to default.

    Returns:
        ScoreResult with score, method="llm_judge", and reasoning in details.

    Raises:
        Exception: Propagated from LLM provider on connectivity failure.

    Note:
        On judge unavailability (unparseable response or provider exception) the
        score is 0.0, NOT 0.5. The runner's pass threshold is 0.5 with ``>=``, so
        returning 0.5 on an outage would silently mark every case as passed. A
        failed judge must score below the pass threshold and flag itself.
    """
    effective_rubric = rubric or _DEFAULT_RUBRIC
    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        task=task,
        output=output,
        expected=expected or "(none provided)",
        rubric=effective_rubric,
    )

    # Check judge cache (explicit arg takes priority, then global cache)
    _cache = cache
    if _cache is None:
        try:
            from harness.eval.judge_cache import get_global_cache
            _cache = get_global_cache()
        except Exception:
            pass
    if _cache is not None:
        cached = _cache.get(prompt)
        if cached is not None:
            return cached

    try:
        response = await llm_provider.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            system="You are a strict but fair AI evaluation judge. Always respond with valid JSON.",
        )
        raw = response.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"\n?```$", "", raw, flags=re.MULTILINE)
            raw = raw.strip()

        parsed = json.loads(raw)
        score = float(parsed.get("score", 0.5))
        reasoning = parsed.get("reasoning", "")
        score = max(0.0, min(1.0, score))

        result = ScoreResult(score=score, method="llm_judge", details=reasoning)
        if _cache is not None:
            _cache.set(prompt, result)
        return result

    except json.JSONDecodeError as exc:
        logger.warning("LLM judge response was not valid JSON: %s", exc)
        # Fallback: try to extract a number from the response
        try:
            raw_text = response.content if hasattr(response, "content") else ""
            match = re.search(r"\b(0\.\d+|1\.0|0|1)\b", raw_text)
            if match:
                score = float(match.group(1))
                return ScoreResult(
                    score=max(0.0, min(1.0, score)),
                    method="llm_judge",
                    details=f"Extracted from non-JSON response: {raw_text[:200]}",
                )
        except Exception:
            pass
        # Judge produced no usable signal — treat as unavailable, NOT uncertain.
        # Scoring 0.5 here would pass the >=0.5 runner threshold on an outage.
        return ScoreResult(
            score=0.0,
            method="llm_judge",
            details="judge unavailable: unparseable response (below pass threshold)",
        )

    except Exception as exc:
        logger.warning("LLM judge call failed: %s", exc)
        return ScoreResult(
            score=0.0,
            method="llm_judge",
            details=f"judge unavailable: {exc} (below pass threshold)",
        )


# ---------------------------------------------------------------------------
# Code-execution scoring (HumanEval-style pass@1)
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_]*\n?|\n?```$", re.MULTILINE)


def _strip_code_fences(text: str) -> str:
    """Remove Markdown code fences an agent may wrap code in."""
    return _CODE_FENCE_RE.sub("", text or "").strip()


async def score_code_execution(output: str, case: Any, sandbox: Any) -> ScoreResult:
    """Score a code-generation case by executing it against its unit tests.

    Assembles a runnable program from the candidate output plus the case's test
    harness — HumanEval style: ``case.metadata["test"]`` defines ``check(...)`` and
    ``case.metadata["entry_point"]`` names the function under test — then runs it in
    ``sandbox`` (an EvalSandbox, e.g. CodeSandbox). Pass (score 1.0) iff the program
    exits cleanly; any assertion error / exception → 0.0.

    If the model returned only the function body/completion (no ``def
    {entry_point}``), the prompt (signature + docstring) is prepended so the
    candidate becomes a complete module.
    """
    meta = getattr(case, "metadata", {}) or {}
    test = meta.get("test", "")
    entry_point = meta.get("entry_point", "")
    prompt = meta.get("prompt", "") or getattr(case, "task", "")

    if not test:
        return ScoreResult(score=0.0, method="code_execution", details="no test harness in case")

    candidate = _strip_code_fences(output)
    if entry_point and f"def {entry_point}" not in candidate:
        program = f"{prompt}\n{candidate}"
    else:
        program = candidate

    program = f"{program}\n\n{test}"
    if entry_point:
        program += f"\n\ncheck({entry_point})\n"

    try:
        result = await sandbox.execute(program, language="python")
    except Exception as exc:  # sandbox unavailable / crashed
        return ScoreResult(score=0.0, method="code_execution", details=f"sandbox error: {exc}")

    if getattr(result, "success", False):
        return ScoreResult(score=1.0, method="code_execution", details="passed unit tests")
    detail = getattr(result, "error", "") or getattr(result, "raw_text", "")
    return ScoreResult(score=0.0, method="code_execution", details=f"failed: {str(detail)[:200]}")
