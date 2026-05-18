"""EvalDataset and EvalCase definitions for HarnessAgent evaluation."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalCase:
    """A single evaluation case with task, optional expected output, and tags.

    Attributes:
        case_id:         Unique identifier for the case.
        agent_type:      Which agent to run this case against.
        task:            The user task string to submit.
        expected_output: Ground-truth substring or None if success-only check.
        gold_actions:    Multiple valid actions/queries (SQL, code, tool calls).
        sandbox_type:    Which EvalSandbox backend to use for execution scoring.
        db_path:         Path to eval database or fixture file.
        hardness:        Pre-labelled difficulty (easy/medium/hard/extra-hard).
        metadata:        Arbitrary extra data (e.g. difficulty, dataset source).
        tags:            List of classification tags for filtering.
    """

    case_id: str
    agent_type: str
    task: str
    expected_output: str | None = None
    gold_actions: list[str] = field(default_factory=list)
    sandbox_type: str = "none"   # "sql" | "code" | "tool" | "http" | "none"
    db_path: str | None = None
    hardness: str | None = None
    metadata: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def all_gold_actions(self) -> list[str]:
        """Deduped union of expected_output + gold_actions, preserving order."""
        seen: set[str] = set()
        result: list[str] = []
        for item in ([self.expected_output] if self.expected_output else []) + self.gold_actions:
            if item and item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for JSONL storage."""
        return {
            "case_id": self.case_id,
            "agent_type": self.agent_type,
            "task": self.task,
            "expected_output": self.expected_output,
            "gold_actions": self.gold_actions,
            "sandbox_type": self.sandbox_type,
            "db_path": self.db_path,
            "hardness": self.hardness,
            "metadata": self.metadata,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EvalCase:
        """Deserialise from a dict (e.g. loaded from JSONL)."""
        return cls(
            case_id=data["case_id"],
            agent_type=data["agent_type"],
            task=data["task"],
            expected_output=data.get("expected_output"),
            gold_actions=data.get("gold_actions", []),
            sandbox_type=data.get("sandbox_type", "none"),
            db_path=data.get("db_path"),
            hardness=data.get("hardness"),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
        )


@dataclass
class EvalDataset:
    """A named collection of EvalCases for a specific agent type.

    Attributes:
        name:       Human-readable dataset name.
        agent_type: The agent type all cases belong to.
        cases:      Ordered list of EvalCase instances.
    """

    name: str
    agent_type: str
    cases: list[EvalCase]

    # ------------------------------------------------------------------
    # Filtering and sampling
    # ------------------------------------------------------------------

    def filter(
        self,
        tags: list[str] | None = None,
        n: int | None = None,
    ) -> EvalDataset:
        """Return a new EvalDataset containing only cases matching the given tags.

        Args:
            tags: If provided, only include cases that have at least one
                  matching tag.  If None, all cases pass.
            n:    If provided, cap the result at n cases (preserves order).

        Returns:
            A new EvalDataset with the filtered subset of cases.
        """
        filtered = self.cases
        if tags:
            tag_set = set(tags)
            filtered = [c for c in filtered if tag_set.intersection(c.tags)]
        if n is not None:
            filtered = filtered[:n]
        return EvalDataset(name=self.name, agent_type=self.agent_type, cases=filtered)

    def sample(self, n: int) -> EvalDataset:
        """Return a new EvalDataset with n cases randomly sampled without replacement.

        Args:
            n: Number of cases to sample; clamped to len(self.cases).

        Returns:
            A new EvalDataset with the sampled cases in random order.
        """
        k = min(n, len(self.cases))
        sampled = random.sample(self.cases, k)
        return EvalDataset(name=self.name, agent_type=self.agent_type, cases=sampled)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @classmethod
    def from_jsonl(cls, path: str) -> EvalDataset:
        """Load a dataset from a JSONL file.

        Each line must be a JSON object matching the EvalCase schema.
        The first case's agent_type determines the dataset's agent_type.
        The file stem is used as the dataset name.

        Args:
            path: Filesystem path to the .jsonl file.

        Returns:
            An EvalDataset populated from the file.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If any line is invalid JSON.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"EvalDataset file not found: {path}")

        cases: list[EvalCase] = []
        with p.open("r", encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    cases.append(EvalCase.from_dict(data))
                except (json.JSONDecodeError, KeyError) as exc:
                    raise ValueError(
                        f"Invalid JSONL at {path}:{line_num}: {exc}"
                    ) from exc

        agent_type = cases[0].agent_type if cases else "unknown"
        return cls(name=p.stem, agent_type=agent_type, cases=cases)

    def to_jsonl(self, path: str) -> None:
        """Persist the dataset to a JSONL file (one EvalCase per line).

        Args:
            path: Destination file path.  Parent directories must exist.
        """
        p = Path(path)
        with p.open("w", encoding="utf-8") as fh:
            for case in self.cases:
                fh.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    def __repr__(self) -> str:
        return (
            f"EvalDataset(name={self.name!r}, agent_type={self.agent_type!r}, "
            f"cases={len(self.cases)})"
        )


@dataclass
class MultiAgentEvalCase:
    """Evaluation case for a planned multi-agent DAG.

    ``subtasks`` is a list of dictionaries with the same shape as
    :class:`harness.orchestrator.planner.SubTask`: id, agent_type, task,
    depends_on, and optional metadata. Keeping it dict-backed makes JSONL
    fixtures easy to edit by hand.
    """

    case_id: str
    task: str
    subtasks: list[dict[str, Any]]
    expected_output: str | None = None
    metadata: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "task": self.task,
            "subtasks": self.subtasks,
            "expected_output": self.expected_output,
            "metadata": self.metadata,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MultiAgentEvalCase:
        return cls(
            case_id=data["case_id"],
            task=data["task"],
            subtasks=list(data.get("subtasks", [])),
            expected_output=data.get("expected_output"),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
        )

    def to_task_plan(self):
        """Convert this case into a validated TaskPlan."""
        from harness.orchestrator.planner import SubTask, TaskPlan

        subtasks = [
            SubTask(
                id=str(item["id"]),
                agent_type=str(item["agent_type"]),
                task=str(item["task"]),
                depends_on=[str(dep) for dep in item.get("depends_on", [])],
                metadata=dict(item.get("metadata", {})),
            )
            for item in self.subtasks
        ]
        return TaskPlan(
            plan_id=f"eval_{self.case_id}",
            original_task=self.task,
            subtasks=subtasks,
        )


@dataclass
class MultiAgentEvalDataset:
    """A named collection of multi-agent plan evaluation cases."""

    name: str
    cases: list[MultiAgentEvalCase]

    def filter(
        self,
        tags: list[str] | None = None,
        n: int | None = None,
    ) -> MultiAgentEvalDataset:
        filtered = self.cases
        if tags:
            tag_set = set(tags)
            filtered = [case for case in filtered if tag_set.intersection(case.tags)]
        if n is not None:
            filtered = filtered[:n]
        return MultiAgentEvalDataset(name=self.name, cases=filtered)

    def to_jsonl(self, path: str) -> None:
        p = Path(path)
        with p.open("w", encoding="utf-8") as fh:
            for case in self.cases:
                fh.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def from_jsonl(cls, path: str) -> MultiAgentEvalDataset:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"MultiAgentEvalDataset file not found: {path}")
        cases: list[MultiAgentEvalCase] = []
        with p.open("r", encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    cases.append(MultiAgentEvalCase.from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError) as exc:
                    raise ValueError(
                        f"Invalid JSONL at {path}:{line_num}: {exc}"
                    ) from exc
        return cls(name=p.stem, cases=cases)

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)


# ---------------------------------------------------------------------------
# Built-in minimal datasets
# ---------------------------------------------------------------------------

SQL_EVAL_CASES: list[EvalCase] = [
    EvalCase(
        case_id="sql_01",
        agent_type="sql",
        task="List all tables in the database",
        tags=["basic"],
    ),
    EvalCase(
        case_id="sql_02",
        agent_type="sql",
        task="Count rows in each table",
        tags=["basic", "count"],
    ),
    EvalCase(
        case_id="sql_03",
        agent_type="sql",
        task="Show the schema for the users table",
        tags=["schema"],
    ),
    EvalCase(
        case_id="sql_04",
        agent_type="sql",
        task="Find users who placed more than 5 orders",
        expected_output="SELECT",
        tags=["join", "aggregate"],
    ),
    EvalCase(
        case_id="sql_05",
        agent_type="sql",
        task="Calculate total revenue per month",
        expected_output="GROUP BY",
        tags=["aggregate", "date"],
    ),
]

CODE_EVAL_CASES: list[EvalCase] = [
    EvalCase(
        case_id="code_01",
        agent_type="code",
        task="Write a Python function to reverse a string and test it",
        expected_output="def reverse_string",
        tags=["basic"],
    ),
    EvalCase(
        case_id="code_02",
        agent_type="code",
        task="Debug this code: def add(a,b): return a-b",
        tags=["debug"],
    ),
    EvalCase(
        case_id="code_03",
        agent_type="code",
        task="Write a function that checks if a number is prime",
        expected_output="def is_prime",
        tags=["basic", "math"],
    ),
    EvalCase(
        case_id="code_04",
        agent_type="code",
        task="Implement a binary search algorithm with tests",
        expected_output="def binary_search",
        tags=["algorithm", "search"],
    ),
]


MULTI_AGENT_EVAL_CASES: list[MultiAgentEvalCase] = [
    MultiAgentEvalCase(
        case_id="multi_01",
        task="Investigate data quality, then write a concise remediation plan.",
        expected_output="remediation",
        tags=["sql", "code", "handoff"],
        subtasks=[
            {
                "id": "inspect_schema",
                "agent_type": "sql",
                "task": "Inspect available tables and identify likely data quality risks.",
                "depends_on": [],
                "metadata": {"eval_role": "data_inspector"},
            },
            {
                "id": "draft_plan",
                "agent_type": "code",
                "task": "Turn the data quality risks into a concise remediation checklist.",
                "depends_on": ["inspect_schema"],
                "metadata": {"eval_role": "remediation_writer"},
            },
        ],
    ),
    MultiAgentEvalCase(
        case_id="multi_02",
        task="Analyze a code task, run a verification step, and summarize release risk.",
        expected_output="risk",
        tags=["code", "review", "verification"],
        subtasks=[
            {
                "id": "implement",
                "agent_type": "code",
                "task": "Implement a small utility function and include a smoke test.",
                "depends_on": [],
                "metadata": {"eval_role": "implementer"},
            },
            {
                "id": "review",
                "agent_type": "code",
                "task": "Review the implementation output and summarize release risk.",
                "depends_on": ["implement"],
                "metadata": {"eval_role": "reviewer"},
            },
        ],
    ),
]
