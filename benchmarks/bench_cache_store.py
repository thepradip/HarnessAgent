"""
BenchmarkCache — persistent disk cache for benchmark intermediate results.

Saves and reloads:
  1. Built SQLite databases (keyed by DDL hash)
  2. NexusSql generated SQL (keyed by question + DDL hash)
  3. SQL execution results (keyed by DB hash + SQL)
  4. Full per-case results for any completed run

This means subsequent experiments (different verifier, Hermes re-run,
ablation) reload from disk without re-calling the LLM.

Cache layout:
  benchmarks/cache/
    dbs/           {ddl_hash}.sqlite          — reusable SQLite databases
    generations/   {run_tag}.jsonl            — generated SQL per question
    executions/    {run_tag}_exec.jsonl       — execution results
    cases/         {run_tag}_cases.jsonl      — full per-case results
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CACHE_ROOT = Path(__file__).resolve().parent / "cache"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class BenchmarkCache:
    """
    All read/write methods are synchronous — safe to call from asyncio via
    run_in_executor or directly (no I/O blocking in practice).
    """

    def __init__(self, run_tag: str) -> None:
        self.run_tag = run_tag
        self._db_dir   = CACHE_ROOT / "dbs"
        self._gen_file = CACHE_ROOT / "generations" / f"{run_tag}.jsonl"
        self._exc_file = CACHE_ROOT / "executions"  / f"{run_tag}_exec.jsonl"
        self._cas_file = CACHE_ROOT / "cases"       / f"{run_tag}_cases.jsonl"

        for d in (self._db_dir,
                  self._gen_file.parent,
                  self._exc_file.parent,
                  self._cas_file.parent):
            d.mkdir(parents=True, exist_ok=True)

        # In-memory indices (loaded lazily)
        self._gen_index: dict[str, str]  | None = None
        self._exc_index: dict[str, dict] | None = None
        self._cas_index: dict[str, dict] | None = None

    # ── SQLite DB cache ───────────────────────────────────────────────────

    def db_path(self, ddl: str) -> Path:
        """Return path for the persistent SQLite DB for this DDL."""
        return self._db_dir / f"{_hash(ddl)}.sqlite"

    def db_exists(self, ddl: str) -> bool:
        return self.db_path(ddl).exists()

    def copy_db(self, src: str, ddl: str) -> Path:
        """Copy a freshly-built SQLite DB into the cache."""
        dst = self.db_path(ddl)
        if not dst.exists():
            shutil.copy2(src, dst)
            logger.debug("DB cached → %s", dst.name)
        return dst

    # ── Generation cache ──────────────────────────────────────────────────

    def _load_gen(self) -> None:
        if self._gen_index is not None:
            return
        self._gen_index = {}
        if self._gen_file.exists():
            for line in self._gen_file.read_text().splitlines():
                if line.strip():
                    try:
                        d = json.loads(line)
                        self._gen_index[d["key"]] = d["generated_sql"]
                    except Exception:
                        pass
        logger.debug("Generation cache: %d entries loaded", len(self._gen_index))

    def gen_key(self, question: str, ddl: str) -> str:
        return _hash(question + ddl)

    def get_generated(self, question: str, ddl: str) -> str | None:
        self._load_gen()
        return self._gen_index.get(self.gen_key(question, ddl))

    def save_generated(self, question: str, ddl: str, sql: str) -> None:
        self._load_gen()
        key = self.gen_key(question, ddl)
        if key not in self._gen_index:
            self._gen_index[key] = sql
            with self._gen_file.open("a") as f:
                f.write(json.dumps({"key": key, "question": question[:200],
                                    "generated_sql": sql}) + "\n")

    # ── Execution cache ───────────────────────────────────────────────────

    def _load_exc(self) -> None:
        if self._exc_index is not None:
            return
        self._exc_index = {}
        if self._exc_file.exists():
            for line in self._exc_file.read_text().splitlines():
                if line.strip():
                    try:
                        d = json.loads(line)
                        self._exc_index[d["key"]] = d["result"]
                    except Exception:
                        pass
        logger.debug("Execution cache: %d entries loaded", len(self._exc_index))

    def exec_key(self, db_path: str, sql: str) -> str:
        # Use DB file hash + SQL hash for portability
        try:
            db_hash = _hash(Path(db_path).read_bytes().hex()[:256])
        except Exception:
            db_hash = _hash(db_path)
        return _hash(db_hash + sql)

    def get_exec(self, db_path: str, sql: str) -> dict | None:
        self._load_exc()
        return self._exc_index.get(self.exec_key(db_path, sql))

    def save_exec(self, db_path: str, sql: str, result: dict) -> None:
        self._load_exc()
        key = self.exec_key(db_path, sql)
        if key not in self._exc_index:
            self._exc_index[key] = result
            with self._exc_file.open("a") as f:
                f.write(json.dumps({"key": key, "result": result}) + "\n")

    # ── Full case results ─────────────────────────────────────────────────

    def _load_cases(self) -> None:
        if self._cas_index is not None:
            return
        self._cas_index = {}
        if self._cas_file.exists():
            for line in self._cas_file.read_text().splitlines():
                if line.strip():
                    try:
                        d = json.loads(line)
                        self._cas_index[d["case_id"]] = d
                    except Exception:
                        pass
        logger.debug("Case cache: %d entries loaded", len(self._cas_index))

    def get_case(self, case_id: str) -> dict | None:
        self._load_cases()
        return self._cas_index.get(case_id)

    def save_case(self, case: dict) -> None:
        self._load_cases()
        cid = case["case_id"]
        if cid not in self._cas_index:
            self._cas_index[cid] = case
            with self._cas_file.open("a") as f:
                f.write(json.dumps(case) + "\n")

    def load_all_cases(self) -> list[dict]:
        self._load_cases()
        return list(self._cas_index.values())

    def case_count(self) -> int:
        self._load_cases()
        return len(self._cas_index)

    # ── Convenience stats ─────────────────────────────────────────────────

    def stats(self) -> dict:
        self._load_gen(); self._load_exc(); self._load_cases()
        return {
            "run_tag": self.run_tag,
            "dbs_cached":         len(list(self._db_dir.glob("*.sqlite"))),
            "generations_cached": len(self._gen_index or {}),
            "executions_cached":  len(self._exc_index or {}),
            "cases_cached":       len(self._cas_index or {}),
        }
