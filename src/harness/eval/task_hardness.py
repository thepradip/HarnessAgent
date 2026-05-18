"""Task hardness classification and non-determinism detection for any agent action."""

from __future__ import annotations

import re
from typing import Literal

HardnessLevel = Literal["easy", "medium", "hard", "extra-hard"]

# ---------------------------------------------------------------------------
# Non-determinism detection
# ---------------------------------------------------------------------------

_NONDETERMINISTIC_SQL = re.compile(
    r"\b(NOW|RANDOM|RAND|CURRENT_TIMESTAMP|CURRENT_DATE|CURRENT_TIME"
    r"|GETDATE|SYSDATETIME|NEWID|UUID|GEN_RANDOM_UUID)\s*\(",
    re.IGNORECASE,
)
_NONDETERMINISTIC_CODE = re.compile(
    r"\b(random\.|uuid\.|datetime\.now|time\.time\(\)|os\.urandom|secrets\.)\b",
    re.IGNORECASE,
)


def detect_nondeterministic(action: str, action_type: str = "sql") -> bool:
    """Return True if the action uses non-deterministic functions."""
    if action_type == "sql":
        return bool(_NONDETERMINISTIC_SQL.search(action))
    if action_type in ("python", "code"):
        return bool(_NONDETERMINISTIC_CODE.search(action))
    return bool(_NONDETERMINISTIC_SQL.search(action) or _NONDETERMINISTIC_CODE.search(action))


# ---------------------------------------------------------------------------
# SQL hardness (sqlglot AST, regex fallback)
# ---------------------------------------------------------------------------

def classify_sql_hardness(sql: str) -> HardnessLevel:
    """Classify SQL complexity per BIRD benchmark criteria."""
    try:
        import sqlglot
        import sqlglot.expressions as exp

        parsed = sqlglot.parse(sql)
        if not parsed:
            return _sql_hardness_regex(sql)
        stmt = parsed[0]
        if stmt is None:
            return _sql_hardness_regex(sql)

        has_cte = bool(stmt.find(exp.With))
        has_window = bool(stmt.find(exp.Window))
        has_subquery = bool(stmt.find(exp.Subquery))

        if has_cte or has_window:
            return "extra-hard"

        joins = list(stmt.find_all(exp.Join))
        has_group = bool(stmt.find(exp.Group))
        has_having = bool(stmt.find(exp.Having))

        if has_subquery and len(joins) >= 1:
            return "extra-hard"
        if len(joins) >= 2 and (has_group or has_having):
            return "hard"
        if len(joins) >= 1 and has_subquery:
            return "hard"
        if len(joins) == 1 or (has_group and has_having) or has_subquery:
            return "medium"
        return "easy"

    except Exception:
        return _sql_hardness_regex(sql)


def _sql_hardness_regex(sql: str) -> HardnessLevel:
    upper = sql.upper()
    if re.search(r"\bWITH\b.+\bAS\s*\(", upper) or re.search(r"\bOVER\s*\(", upper):
        return "extra-hard"
    join_count = len(re.findall(r"\bJOIN\b", upper))
    has_subquery = "SELECT" in upper[upper.find("FROM"):] if "FROM" in upper else False
    has_group = bool(re.search(r"\bGROUP\s+BY\b", upper))
    has_having = bool(re.search(r"\bHAVING\b", upper))
    if join_count >= 2 and (has_group or has_having):
        return "hard"
    if join_count >= 1 and has_subquery:
        return "hard"
    if join_count == 1 or (has_group and has_having) or has_subquery:
        return "medium"
    return "easy"


# ---------------------------------------------------------------------------
# General task hardness
# ---------------------------------------------------------------------------

_LONG_HORIZON_MARKERS = re.compile(
    r"\b(multi.?agent|orchestrat|pipeline|workflow|sequenc|iterativ"
    r"|over\s+multiple|across\s+multiple|step.by.step\s+plan)\b",
    re.IGNORECASE,
)
_COMPLEX_MARKERS = re.compile(
    r"\b(analyz|compar|evaluat|investigat|debug|refactor|optimiz|architect)\b",
    re.IGNORECASE,
)


def classify_task_hardness(
    task: str,
    tools_required: list[str] | None = None,
    expected_steps: int | None = None,
    agents_required: int = 1,
) -> HardnessLevel:
    """Classify task complexity for any agentic app."""
    n_tools = len(tools_required) if tools_required else 0
    steps = expected_steps or 0

    if agents_required > 1 or steps > 10 or _LONG_HORIZON_MARKERS.search(task):
        return "extra-hard"
    if steps >= 5 or n_tools >= 3 or _COMPLEX_MARKERS.search(task):
        return "hard"
    if steps >= 2 or n_tools >= 1:
        return "medium"
    return "easy"
