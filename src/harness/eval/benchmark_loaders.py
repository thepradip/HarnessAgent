"""Benchmark dataset loaders: generic (JSONL/CSV) and domain-specific (Spider, BIRD, HumanEval, GSM8K)."""

from __future__ import annotations

import csv
import json
import logging
import os
import random
from pathlib import Path
from typing import Literal

from harness.eval.datasets import EvalCase, EvalDataset
from harness.eval.task_hardness import classify_sql_hardness

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generic loaders
# ---------------------------------------------------------------------------

def load_jsonl(
    path: str,
    task_field: str = "task",
    gold_field: str = "gold",
    agent_type: str = "base",
    n_samples: int | None = None,
) -> EvalDataset:
    """Load any JSONL file into an EvalDataset."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    cases: list[EvalCase] = []
    with p.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            case = EvalCase(
                case_id=str(row.get("id", row.get("case_id", i))),
                agent_type=row.get("agent_type", agent_type),
                task=str(row[task_field]),
                expected_output=str(row[gold_field]) if gold_field in row else None,
                gold_actions=row.get("gold_actions", []),
                sandbox_type=row.get("sandbox_type", "none"),
                db_path=row.get("db_path"),
                hardness=row.get("hardness"),
                metadata={k: v for k, v in row.items()
                          if k not in (task_field, gold_field, "id", "case_id",
                                       "agent_type", "tags", "gold_actions",
                                       "sandbox_type", "db_path", "hardness")},
                tags=row.get("tags", []),
            )
            cases.append(case)
    if n_samples is not None:
        cases = random.sample(cases, min(n_samples, len(cases)))
    return EvalDataset(name=p.stem, agent_type=agent_type, cases=cases)


def load_csv(
    path: str,
    task_col: str = "task",
    gold_col: str = "gold",
    agent_type: str = "base",
    n_samples: int | None = None,
) -> EvalDataset:
    """Load a CSV file into an EvalDataset."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    cases: list[EvalCase] = []
    with p.open("r", encoding="utf-8", newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            cases.append(EvalCase(
                case_id=str(row.get("id", row.get("case_id", i))),
                agent_type=row.get("agent_type", agent_type),
                task=str(row[task_col]),
                expected_output=str(row[gold_col]) if gold_col in row else None,
                tags=row.get("tags", "").split(",") if row.get("tags") else [],
            ))
    if n_samples is not None:
        cases = random.sample(cases, min(n_samples, len(cases)))
    return EvalDataset(name=p.stem, agent_type=agent_type, cases=cases)


# ---------------------------------------------------------------------------
# Spider loader
# ---------------------------------------------------------------------------

def load_spider(
    spider_dir: str,
    split: Literal["train", "dev"] = "dev",
    n_samples: int | None = None,
    stratify_by_hardness: bool = True,
) -> EvalDataset:
    """Load Spider benchmark into EvalDataset.

    spider_dir must contain:
      {split}.json (or train_spider.json / dev.json)
      database/{db_id}/{db_id}.sqlite
    """
    base = Path(spider_dir)
    candidates = [
        base / f"{split}.json",
        base / f"{split}_spider.json",
        base / "spider" / f"{split}.json",
    ]
    data_file = next((f for f in candidates if f.exists()), None)
    if data_file is None:
        raise FileNotFoundError(f"Spider {split} file not found in {spider_dir}")

    with data_file.open("r", encoding="utf-8") as fh:
        rows = json.load(fh)

    cases: list[EvalCase] = []
    for i, row in enumerate(rows):
        db_id = row.get("db_id", "")
        db_path = str(base / "database" / db_id / f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            db_path_alt = str(base / "spider" / "database" / db_id / f"{db_id}.sqlite")
            db_path = db_path_alt if os.path.exists(db_path_alt) else db_path

        gold_sql = row.get("query", row.get("gold", ""))
        hardness = classify_sql_hardness(gold_sql) if gold_sql else None

        cases.append(EvalCase(
            case_id=f"spider_{split}_{i}",
            agent_type="sql",
            task=row.get("question", ""),
            expected_output=gold_sql,
            gold_actions=[gold_sql] if gold_sql else [],
            sandbox_type="sql",
            db_path=db_path,
            hardness=hardness,
            metadata={"db_id": db_id, "source": "spider"},
            tags=["spider", split, hardness or "unknown"],
        ))

    if n_samples is not None:
        if stratify_by_hardness:
            cases = _stratified_sample(cases, n_samples)
        else:
            cases = random.sample(cases, min(n_samples, len(cases)))

    return EvalDataset(name=f"spider_{split}", agent_type="sql", cases=cases)


# ---------------------------------------------------------------------------
# BIRD loader
# ---------------------------------------------------------------------------

def load_bird(
    bird_dir: str,
    split: Literal["train", "dev"] = "dev",
    n_samples: int | None = None,
) -> EvalDataset:
    """Load BIRD benchmark into EvalDataset.

    bird_dir must contain {split}/{split}.json and {split}/databases/{db_id}/{db_id}.sqlite
    """
    base = Path(bird_dir)
    candidates = [
        base / split / f"{split}.json",
        base / f"{split}.json",
    ]
    data_file = next((f for f in candidates if f.exists()), None)
    if data_file is None:
        raise FileNotFoundError(f"BIRD {split} file not found in {bird_dir}")

    with data_file.open("r", encoding="utf-8") as fh:
        rows = json.load(fh)

    cases: list[EvalCase] = []
    for i, row in enumerate(rows):
        db_id = row.get("db_id", "")
        db_root = base / split / "databases" / db_id
        if not db_root.exists():
            db_root = base / "databases" / db_id
        db_path = str(db_root / f"{db_id}.sqlite")

        gold_sql = row.get("SQL", row.get("query", ""))
        gold_sqls = row.get("gold_sqls", [gold_sql] if gold_sql else [])
        hardness = row.get("difficulty", classify_sql_hardness(gold_sql) if gold_sql else None)

        cases.append(EvalCase(
            case_id=f"bird_{split}_{i}",
            agent_type="sql",
            task=row.get("question", ""),
            expected_output=gold_sql,
            gold_actions=gold_sqls,
            sandbox_type="sql",
            db_path=db_path,
            hardness=hardness,
            metadata={"db_id": db_id, "source": "bird"},
            tags=["bird", split, str(hardness or "unknown")],
        ))

    if n_samples is not None:
        cases = random.sample(cases, min(n_samples, len(cases)))

    return EvalDataset(name=f"bird_{split}", agent_type="sql", cases=cases)


# ---------------------------------------------------------------------------
# HumanEval loader
# ---------------------------------------------------------------------------

def load_humaneval(
    humaneval_path: str,
    n_samples: int | None = None,
) -> EvalDataset:
    """Load HumanEval (openai/human-eval JSONL format) into EvalDataset."""
    p = Path(humaneval_path)
    if not p.exists():
        raise FileNotFoundError(humaneval_path)
    cases: list[EvalCase] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            task_id = row.get("task_id", f"humaneval_{len(cases)}")
            prompt = row.get("prompt", "")
            canonical = row.get("canonical_solution", "")
            cases.append(EvalCase(
                case_id=task_id,
                agent_type="code",
                task=prompt,
                expected_output=canonical,
                gold_actions=[canonical] if canonical else [],
                sandbox_type="code",
                hardness="medium",
                metadata={"entry_point": row.get("entry_point", ""), "source": "humaneval"},
                tags=["humaneval"],
            ))
    if n_samples is not None:
        cases = random.sample(cases, min(n_samples, len(cases)))
    return EvalDataset(name="humaneval", agent_type="code", cases=cases)


# ---------------------------------------------------------------------------
# GSM8K loader
# ---------------------------------------------------------------------------

def load_gsm8k(
    gsm8k_path: str,
    split: Literal["train", "test"] = "test",
    n_samples: int | None = None,
) -> EvalDataset:
    """Load GSM8K (JSONL format) into EvalDataset."""
    p = Path(gsm8k_path)
    if not p.exists():
        raise FileNotFoundError(gsm8k_path)
    cases: list[EvalCase] = []
    with p.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = row.get("question", "")
            answer = row.get("answer", "")
            # GSM8K answers end with #### <number>
            final_answer = answer.split("####")[-1].strip() if "####" in answer else answer
            cases.append(EvalCase(
                case_id=f"gsm8k_{split}_{i}",
                agent_type="base",
                task=question,
                expected_output=final_answer,
                sandbox_type="none",
                hardness="medium",
                metadata={"full_answer": answer, "source": "gsm8k"},
                tags=["gsm8k", split],
            ))
    if n_samples is not None:
        cases = random.sample(cases, min(n_samples, len(cases)))
    return EvalDataset(name=f"gsm8k_{split}", agent_type="base", cases=cases)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stratified_sample(cases: list[EvalCase], n: int) -> list[EvalCase]:
    """Sample n cases stratified by hardness."""
    buckets: dict[str, list[EvalCase]] = {}
    for c in cases:
        buckets.setdefault(c.hardness or "unknown", []).append(c)
    result: list[EvalCase] = []
    per_bucket = max(1, n // len(buckets))
    for bucket in buckets.values():
        result.extend(random.sample(bucket, min(per_bucket, len(bucket))))
    if len(result) < n:
        remaining = [c for c in cases if c not in result]
        result.extend(random.sample(remaining, min(n - len(result), len(remaining))))
    return result[:n]
