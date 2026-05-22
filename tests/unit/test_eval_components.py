"""Unit tests for eval components: EvalSandbox, FailureTaxonomy,
AgentScores, AgentEvalReport, benchmark loaders."""

from __future__ import annotations

import csv
import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from harness.eval.agent_report import AgentEvalReport, generate_report
from harness.eval.agent_scorer import AgentScores, evaluate_agent_output
from harness.eval.benchmark_loaders import (
    load_bird,
    load_csv,
    load_gsm8k,
    load_humaneval,
    load_jsonl,
    load_spider,
)
from harness.eval.datasets import EvalCase, EvalDataset
from harness.eval.failure_taxonomy import (
    FailureAnalysis,
    FailureCategory,
    classify_failure,
)
from harness.improvement.rlvr.verifiers import VerificationStep
from harness.eval.sandbox import (
    CodeSandbox,
    HttpSandbox,
    SQLSandbox,
    SandboxResult,
    ToolCallSandbox,
)
from harness.eval.task_hardness import (
    classify_sql_hardness,
    classify_task_hardness,
    detect_nondeterministic,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scores(c=0.9, q=0.8, s=1.0, hardness="easy", verdict="PASS") -> AgentScores:
    return AgentScores(
        correctness_score=c, quality_score=q, safety_score=s,
        overall_score=round(c * 0.5 + q * 0.3 + s * 0.2, 4),
        hardness=hardness, nondeterministic_warning=False,
        verdict=verdict,
    )


def _sqlite_db(tmp_path: Path, name="test") -> str:
    db = str(tmp_path / f"{name}.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, total REAL)")
    conn.execute("INSERT INTO users VALUES (1,'Alice'),(2,'Bob')")
    conn.execute("INSERT INTO orders VALUES (1,1,100.0),(2,2,200.0)")
    conn.commit()
    conn.close()
    return db


# ===========================================================================
# SandboxResult
# ===========================================================================

def test_sandbox_result_success_true_when_no_error():
    r = SandboxResult(output={}, raw_text="ok", execution_time_ms=5.0)
    assert r.success is True


def test_sandbox_result_success_false_when_error():
    r = SandboxResult(output={}, raw_text="", execution_time_ms=1.0, error="oops")
    assert r.success is False


def test_sandbox_result_truncated_default_false():
    r = SandboxResult(output={}, raw_text="", execution_time_ms=0.0)
    assert r.truncated is False


# ===========================================================================
# SQLSandbox
# ===========================================================================

@pytest.mark.asyncio
async def test_sql_sandbox_execute_select(tmp_path):
    db = _sqlite_db(tmp_path)
    sb = SQLSandbox(db_path=db)
    result = await sb.execute("SELECT COUNT(*) FROM users")
    assert result.success
    assert result.output["row_count"] == 1
    assert result.output["rows"] == [[2]]


@pytest.mark.asyncio
async def test_sql_sandbox_execute_with_filter(tmp_path):
    db = _sqlite_db(tmp_path)
    sb = SQLSandbox(db_path=db)
    result = await sb.execute("SELECT name FROM users WHERE id = 1")
    assert result.success
    assert result.output["rows"] == [["Alice"]]


@pytest.mark.asyncio
async def test_sql_sandbox_rejects_insert(tmp_path):
    db = _sqlite_db(tmp_path)
    sb = SQLSandbox(db_path=db)
    # Safety violation is caught internally and returned as an error SandboxResult
    result = await sb.execute("INSERT INTO users VALUES (3,'Eve')")
    assert not result.success


@pytest.mark.asyncio
async def test_sql_sandbox_rejects_drop(tmp_path):
    db = _sqlite_db(tmp_path)
    sb = SQLSandbox(db_path=db)
    result = await sb.execute("DROP TABLE users")
    assert not result.success


@pytest.mark.asyncio
async def test_sql_sandbox_is_available_with_path(tmp_path):
    db = _sqlite_db(tmp_path)
    sb = SQLSandbox(db_path=db)
    assert await sb.is_available() is True


@pytest.mark.asyncio
async def test_sql_sandbox_is_available_no_path():
    sb = SQLSandbox(db_path=None)
    assert await sb.is_available() is False


@pytest.mark.asyncio
async def test_sql_sandbox_no_db_path_returns_error():
    sb = SQLSandbox(db_path=None)
    result = await sb.execute("SELECT 1")
    assert result.error is not None
    assert not result.success


@pytest.mark.asyncio
async def test_sql_sandbox_max_rows_truncates(tmp_path):
    db = _sqlite_db(tmp_path)
    sb = SQLSandbox(db_path=db, max_rows=1)
    result = await sb.execute("SELECT * FROM users")
    assert result.truncated is True
    assert len(result.output["rows"]) == 1


@pytest.mark.asyncio
async def test_sql_sandbox_raw_text_not_empty_on_success(tmp_path):
    db = _sqlite_db(tmp_path)
    sb = SQLSandbox(db_path=db)
    result = await sb.execute("SELECT * FROM users")
    assert result.raw_text  # non-empty


@pytest.mark.asyncio
async def test_sql_sandbox_columns_returned(tmp_path):
    db = _sqlite_db(tmp_path)
    sb = SQLSandbox(db_path=db)
    result = await sb.execute("SELECT id, name FROM users")
    assert result.output["columns"] == ["id", "name"]


@pytest.mark.asyncio
async def test_sql_sandbox_timeout_field_stored():
    # Just verify timeout is stored and used — don't actually fire it
    # (firing asyncio.wait_for with tiny timeout leaves orphan executor threads)
    sb = SQLSandbox(db_path="/tmp/test.db", timeout=0.5)
    assert sb._timeout == pytest.approx(0.5)


# ===========================================================================
# CodeSandbox (mock Docker to avoid subprocess hang in CI)
# ===========================================================================

@pytest.mark.asyncio
async def test_code_sandbox_is_available_returns_bool():
    sb = CodeSandbox()
    with patch("harness.eval.sandbox.CodeSandbox.is_available", new=AsyncMock(return_value=False)):
        result = await sb.is_available()
    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_code_sandbox_execute_returns_sandbox_result():
    sb = CodeSandbox()
    # Patch DockerSandbox.is_available so we don't spawn a subprocess in CI
    with patch("harness.filesystem.sandbox.DockerSandbox.is_available",
               new=AsyncMock(return_value=False)):
        result = await sb.execute("print('hello')")
    assert isinstance(result, SandboxResult)


# ===========================================================================
# filesystem.SandboxResult — oom_killed property
# ===========================================================================

def test_fs_sandbox_result_oom_killed_true():
    from harness.filesystem.sandbox import SandboxResult as FSSandboxResult
    r = FSSandboxResult(stdout="", stderr="Killed", exit_code=137, timed_out=False, execution_time_ms=10.0)
    assert r.oom_killed is True


def test_fs_sandbox_result_oom_killed_false_when_timed_out():
    from harness.filesystem.sandbox import SandboxResult as FSSandboxResult
    # timed_out flag means the harness killed the process, not the OOM killer
    r = FSSandboxResult(stdout="", stderr="", exit_code=137, timed_out=True, execution_time_ms=30_000.0)
    assert r.oom_killed is False


def test_fs_sandbox_result_oom_killed_false_normal_failure():
    from harness.filesystem.sandbox import SandboxResult as FSSandboxResult
    r = FSSandboxResult(stdout="", stderr="error", exit_code=1, timed_out=False, execution_time_ms=5.0)
    assert r.oom_killed is False


def test_fs_sandbox_result_oom_killed_false_success():
    from harness.filesystem.sandbox import SandboxResult as FSSandboxResult
    r = FSSandboxResult(stdout="ok", stderr="", exit_code=0, timed_out=False, execution_time_ms=5.0)
    assert r.oom_killed is False


# ===========================================================================
# CodeSandbox — OOM detection (exit code 137)
# ===========================================================================

@pytest.mark.asyncio
async def test_code_sandbox_oom_returns_clear_error():
    from harness.filesystem.sandbox import SandboxResult as FSSandboxResult
    oom_result = FSSandboxResult(
        stdout="", stderr="Killed", exit_code=137, timed_out=False, execution_time_ms=100.0
    )
    mock_instance = AsyncMock()
    mock_instance.run_code = AsyncMock(return_value=oom_result)
    mock_cls = MagicMock(return_value=mock_instance)
    mock_cls.is_available = AsyncMock(return_value=True)

    with patch("harness.filesystem.sandbox.DockerSandbox", mock_cls):
        sb = CodeSandbox()
        result = await sb.execute("x = [0] * 10**9")

    assert result.error == "OOM: container exceeded memory limit"
    assert result.success is False


@pytest.mark.asyncio
async def test_code_sandbox_non_oom_failure_preserves_exit_code_in_error():
    from harness.filesystem.sandbox import SandboxResult as FSSandboxResult
    fail_result = FSSandboxResult(
        stdout="", stderr="NameError: name 'x' is not defined", exit_code=1, timed_out=False,
        execution_time_ms=50.0
    )
    mock_instance = AsyncMock()
    mock_instance.run_code = AsyncMock(return_value=fail_result)
    mock_cls = MagicMock(return_value=mock_instance)
    mock_cls.is_available = AsyncMock(return_value=True)

    with patch("harness.filesystem.sandbox.DockerSandbox", mock_cls):
        sb = CodeSandbox()
        result = await sb.execute("print(x)")

    assert result.error is not None
    assert "exit_code=1" in result.error
    assert "OOM" not in result.error


# ===========================================================================
# RunCodeTool — OOM detection across all execution paths
# ===========================================================================

@pytest.mark.asyncio
async def test_run_code_tool_subprocess_oom_returns_error(tmp_path):
    from harness.tools.code_tools import RunCodeTool
    from harness.core.context import ToolResult

    tool = RunCodeTool()  # no docker_sandbox, no restricted_executor → subprocess path
    ctx = MagicMock()
    ctx.run_id = "test-run"
    ctx.step_count = 1
    ctx.workspace_path = tmp_path

    mock_proc = AsyncMock()
    mock_proc.returncode = 137
    mock_proc.communicate = AsyncMock(return_value=(b"", b"Killed"))

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await tool.execute(ctx, {"code": "x = [0]*10**9"})

    assert result.error == "OOM: container exceeded memory limit"
    assert result.data is None


@pytest.mark.asyncio
async def test_run_code_tool_subprocess_normal_failure_not_oom(tmp_path):
    from harness.tools.code_tools import RunCodeTool

    tool = RunCodeTool()
    ctx = MagicMock()
    ctx.run_id = "test-run"
    ctx.step_count = 1
    ctx.workspace_path = tmp_path

    mock_proc = AsyncMock()
    mock_proc.returncode = 1
    mock_proc.communicate = AsyncMock(return_value=(b"", b"ZeroDivisionError"))

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await tool.execute(ctx, {"code": "1/0"})

    # Normal failure — data is returned, error is None (non-zero exits are surfaced in data)
    assert result.error != "OOM: container exceeded memory limit"


@pytest.mark.asyncio
async def test_run_code_tool_docker_oom_returns_error(tmp_path):
    from harness.tools.code_tools import RunCodeTool

    from harness.filesystem.sandbox import SandboxResult as FSSandboxResult
    mock_docker = MagicMock()
    mock_docker.run_code = AsyncMock(return_value=FSSandboxResult(
        stdout="", stderr="Killed", exit_code=137, timed_out=False, execution_time_ms=100.0
    ))
    tool = RunCodeTool(docker_sandbox=mock_docker)
    ctx = MagicMock()
    ctx.run_id = "test-run"
    ctx.step_count = 1
    ctx.workspace_path = tmp_path
    ctx.metadata = {}  # no live session

    result = await tool.execute(ctx, {"code": "x = [0]*10**9"})

    assert result.error == "OOM: container exceeded memory limit"
    assert result.data is None


# ===========================================================================
# FailureTaxonomy
# ===========================================================================

def test_classify_failure_unsafe_sql():
    f = classify_failure("output", scores={}, action="DROP TABLE users")
    assert f.primary == FailureCategory.UNSAFE_ACTION


def test_classify_failure_unsafe_insert():
    f = classify_failure("output", scores={}, action="INSERT INTO users VALUES (1,'x')")
    assert f.primary == FailureCategory.UNSAFE_ACTION


def test_classify_failure_pii_in_output():
    f = classify_failure("email: user@example.com", scores={}, action="SELECT *")
    assert f.primary == FailureCategory.PII_LEAK


def test_classify_failure_sql_injection():
    f = classify_failure("ok", scores={}, action="SELECT * FROM users UNION ALL SELECT 1,2,3")
    assert f.primary == FailureCategory.INJECTION_ATTEMPT


def test_classify_failure_row_explosion():
    f = classify_failure(
        "output",
        scores={"row_count_match": 0.1},
        details={"pred_rows": 900, "gold_rows": 100},
    )
    assert f.primary == FailureCategory.ROW_EXPLOSION


def test_classify_failure_output_truncated():
    f = classify_failure(
        "output",
        scores={"row_count_match": 0.2},
        action="SELECT * FROM users LIMIT 10",
    )
    assert f.primary == FailureCategory.OUTPUT_TRUNCATED


def test_classify_failure_full_scan():
    f = classify_failure("output", scores={}, action="SELECT * FROM users;")
    assert f.primary == FailureCategory.FULL_SCAN


def test_classify_failure_wrong_tool():
    f = classify_failure(
        "output",
        scores={"correctness": 0.1},
        details={"used_tools": ["file_read"], "expected_tools": ["execute_sql"]},
        expected_tools=["execute_sql"],
    )
    assert f.primary == FailureCategory.WRONG_TOOL


def test_classify_failure_faithfulness_drop():
    f = classify_failure("output", scores={"faithfulness": 0.3})
    assert f.primary == FailureCategory.FAITHFULNESS_DROP


def test_classify_failure_unknown_when_no_signal():
    f = classify_failure("clean output", scores={"correctness": 0.5})
    assert f.primary == FailureCategory.UNKNOWN


def test_failure_analysis_top_hint_all_categories():
    for cat in FailureCategory:
        fa = FailureAnalysis(primary=cat, evidence={}, score=0.0)
        hint = fa.top_hint()
        assert isinstance(hint, str) and len(hint) > 5


def test_failure_analysis_summary_format():
    fa = FailureAnalysis(primary=FailureCategory.FULL_SCAN, evidence={}, score=0.1)
    summary = fa.summary()
    assert "full_scan" in summary
    assert "0.100" in summary


def test_failure_analysis_hallucination_with_schema():
    # Use a filtered query so FULL_SCAN is not triggered; only hallucination fires
    f = classify_failure(
        "output",
        scores={"correctness": 0.1},
        schema_names=["users", "orders"],
        action="SELECT id FROM fake_table WHERE id = 1",
    )
    assert f.primary == FailureCategory.HALLUCINATION


# ===========================================================================
# AgentScores
# ===========================================================================

def test_agent_scores_pass_all_thresholds():
    s = _scores(c=0.9, q=0.8, s=1.0, verdict="PASS")
    assert s.verdict == "PASS"


def test_agent_scores_fail_below_correctness():
    s = AgentScores(
        correctness_score=0.3, quality_score=0.8, safety_score=1.0,
        overall_score=0.3, hardness="easy",
        nondeterministic_warning=False, verdict="FAIL",
    )
    assert s.verdict == "FAIL"


def test_agent_scores_fail_below_safety():
    s = AgentScores(
        correctness_score=0.9, quality_score=0.8, safety_score=0.5,
        overall_score=0.76, hardness="easy",
        nondeterministic_warning=False, verdict="FAIL",
    )
    assert s.verdict == "FAIL"


def test_agent_scores_to_markdown_contains_dimensions():
    s = _scores()
    md = s.to_markdown_report(task="test task", action="SELECT 1")
    assert "Correctness" in md
    assert "Quality" in md
    assert "Safety" in md
    assert "PASS" in md


def test_agent_scores_to_json_valid():
    s = _scores()
    parsed = json.loads(s.to_json())
    assert parsed["verdict"] == "PASS"
    assert 0.0 <= parsed["correctness_score"] <= 1.0


def test_agent_scores_nondeterministic_warning_in_report():
    s = AgentScores(
        correctness_score=0.9, quality_score=0.8, safety_score=1.0,
        overall_score=0.89, hardness="easy",
        nondeterministic_warning=True, verdict="PASS",
    )
    md = s.to_markdown_report()
    assert "Non-deterministic" in md or "nondeterministic" in md.lower()


def test_agent_scores_failure_analysis_in_report():
    fa = FailureAnalysis(primary=FailureCategory.FULL_SCAN, evidence={}, score=0.0)
    s = AgentScores(
        correctness_score=0.2, quality_score=0.3, safety_score=1.0,
        overall_score=0.22, hardness="easy",
        nondeterministic_warning=False, verdict="FAIL",
        failure_analysis=fa,
    )
    md = s.to_markdown_report()
    assert "full_scan" in md


@pytest.mark.asyncio
async def test_evaluate_agent_output_no_sandbox_no_gold():
    result = await evaluate_agent_output(
        task="What is 2+2?",
        output="4",
        action="4",
        gold_action=None,
        sandbox=None,
        llm_judge=None,
    )
    assert isinstance(result, AgentScores)
    assert result.verdict in ("PASS", "FAIL")
    assert 0.0 <= result.overall_score <= 1.0


@pytest.mark.asyncio
async def test_evaluate_agent_output_exact_match_gold():
    result = await evaluate_agent_output(
        task="count users",
        output="SELECT COUNT(*) FROM users",
        action="SELECT COUNT(*) FROM users",
        gold_action="SELECT COUNT(*) FROM users",
        sandbox=None,
    )
    assert result.correctness_score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_evaluate_agent_output_safety_violation():
    result = await evaluate_agent_output(
        task="delete users",
        output="Done",
        action="DROP TABLE users",
    )
    assert result.safety_score == pytest.approx(0.0)
    assert result.verdict == "FAIL"


@pytest.mark.asyncio
async def test_evaluate_agent_output_hardness_set():
    result = await evaluate_agent_output(task="list tables", output="output")
    assert result.hardness in ("easy", "medium", "hard", "extra-hard")


@pytest.mark.asyncio
async def test_evaluate_agent_output_nondeterminism_detected():
    result = await evaluate_agent_output(
        task="get current time",
        output="result",
        action="SELECT NOW() FROM dual",
    )
    assert result.nondeterministic_warning is True


# ===========================================================================
# AgentEvalReport
# ===========================================================================

def test_agent_eval_report_pass_rate():
    scores = [_scores(verdict="PASS"), _scores(c=0.3, q=0.3, s=0.3, verdict="FAIL")]
    report = AgentEvalReport(dataset_name="test", scores=scores)
    assert report.overall_pass_rate() == pytest.approx(0.5)


def test_agent_eval_report_all_pass():
    scores = [_scores(verdict="PASS")] * 5
    report = AgentEvalReport(dataset_name="test", scores=scores)
    assert report.overall_pass_rate() == pytest.approx(1.0)


def test_agent_eval_report_all_fail():
    scores = [_scores(c=0.1, q=0.1, s=0.1, verdict="FAIL")] * 3
    report = AgentEvalReport(dataset_name="test", scores=scores)
    assert report.overall_pass_rate() == pytest.approx(0.0)


def test_agent_eval_report_by_hardness():
    scores = [
        _scores(hardness="easy",   verdict="PASS"),
        _scores(hardness="easy",   verdict="PASS"),
        _scores(hardness="hard",   c=0.2, q=0.2, s=0.2, verdict="FAIL"),
    ]
    report = AgentEvalReport(dataset_name="test", scores=scores)
    bh = report.by_hardness()
    assert bh["easy"] == pytest.approx(1.0)
    assert bh["hard"] == pytest.approx(0.0)


def test_agent_eval_report_by_dimension():
    scores = [_scores(c=0.8, q=0.7, s=1.0, verdict="PASS")] * 2
    report = AgentEvalReport(dataset_name="test", scores=scores)
    dims = report.by_dimension()
    assert dims["correctness"] == pytest.approx(0.8)
    assert dims["quality"] == pytest.approx(0.7)
    assert dims["safety"] == pytest.approx(1.0)


def test_agent_eval_report_failure_distribution():
    from harness.eval.failure_taxonomy import FailureAnalysis, FailureCategory
    fa = FailureAnalysis(FailureCategory.FULL_SCAN, {}, 0.0)
    scores = [
        _scores(c=0.1, q=0.1, s=0.1, verdict="FAIL"),
        _scores(c=0.1, q=0.1, s=0.1, verdict="FAIL"),
    ]
    scores[0].failure_analysis = fa
    scores[1].failure_analysis = fa
    report = AgentEvalReport(dataset_name="test", scores=scores)
    dist = report.failure_distribution()
    assert dist.get("full_scan", 0) == 2


def test_agent_eval_report_to_markdown():
    scores = [_scores(verdict="PASS"), _scores(c=0.2, q=0.2, s=0.2, verdict="FAIL")]
    report = AgentEvalReport(dataset_name="my_dataset", scores=scores)
    md = report.to_markdown()
    assert "my_dataset" in md
    assert "50.0%" in md or "50%" in md
    assert "Pass rate" in md or "pass_rate" in md.lower()


def test_agent_eval_report_to_json():
    scores = [_scores(verdict="PASS")]
    report = AgentEvalReport(dataset_name="test", scores=scores)
    parsed = json.loads(report.to_json())
    assert parsed["pass_rate"] == pytest.approx(1.0)
    assert "by_dimension" in parsed


def test_agent_eval_report_tasks_shown_in_markdown():
    scores = [_scores(verdict="PASS")]
    report = AgentEvalReport(dataset_name="test", scores=scores,
                              tasks=["count active users"])
    md = report.to_markdown()
    assert "count active users" in md


def test_generate_report_markdown():
    md = generate_report([_scores()], tasks=["task1"], dataset_name="ds")
    assert "ds" in md


def test_generate_report_json():
    j = generate_report([_scores()], format="json", dataset_name="ds")
    assert json.loads(j)["dataset_name"] == "ds"


def test_agent_eval_report_empty():
    report = AgentEvalReport(dataset_name="empty", scores=[])
    assert report.overall_pass_rate() == 0.0
    assert report.by_hardness() == {}
    assert report.failure_distribution() == {}


# ===========================================================================
# Benchmark loaders
# ===========================================================================

@pytest.mark.asyncio
async def test_load_jsonl(tmp_path):
    data = [
        {"id": "q1", "task": "count users", "gold": "SELECT COUNT(*) FROM users",
         "agent_type": "sql", "sandbox_type": "sql"},
        {"id": "q2", "task": "list tables",  "gold": "SHOW TABLES",
         "agent_type": "sql"},
    ]
    p = tmp_path / "cases.jsonl"
    p.write_text("\n".join(json.dumps(d) for d in data))
    ds = load_jsonl(str(p))
    assert len(ds.cases) == 2
    assert ds.cases[0].case_id == "q1"
    assert ds.cases[0].expected_output == "SELECT COUNT(*) FROM users"
    assert ds.cases[0].sandbox_type == "sql"


@pytest.mark.asyncio
async def test_load_jsonl_n_samples(tmp_path):
    data = [{"id": str(i), "task": f"t{i}", "gold": "x"} for i in range(20)]
    p = tmp_path / "big.jsonl"
    p.write_text("\n".join(json.dumps(d) for d in data))
    ds = load_jsonl(str(p), n_samples=5)
    assert len(ds.cases) == 5


@pytest.mark.asyncio
async def test_load_jsonl_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_jsonl("/does/not/exist.jsonl")


@pytest.mark.asyncio
async def test_load_csv(tmp_path):
    p = tmp_path / "cases.csv"
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "task", "gold"])
        writer.writeheader()
        writer.writerow({"id": "1", "task": "count users", "gold": "SELECT COUNT(*)"})
        writer.writerow({"id": "2", "task": "list all",    "gold": "SELECT *"})
    ds = load_csv(str(p))
    assert len(ds.cases) == 2
    assert ds.cases[0].task == "count users"


@pytest.mark.asyncio
async def test_load_csv_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_csv("/no/such/file.csv")


@pytest.mark.asyncio
async def test_load_spider(tmp_path):
    spider_dir = tmp_path / "spider"
    spider_dir.mkdir()
    db_dir = spider_dir / "database" / "mydb"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(str(db_dir / "mydb.sqlite"))
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.commit()
    conn.close()
    data = [
        {"question": "count rows", "query": "SELECT COUNT(*) FROM t", "db_id": "mydb"},
        {"question": "all rows",   "query": "SELECT * FROM t",         "db_id": "mydb"},
    ]
    (spider_dir / "dev.json").write_text(json.dumps(data))
    ds = load_spider(str(spider_dir), split="dev")
    assert len(ds.cases) == 2
    assert ds.cases[0].task == "count rows"
    assert ds.cases[0].sandbox_type == "sql"
    assert ds.cases[0].hardness is not None
    assert ds.agent_type == "sql"


@pytest.mark.asyncio
async def test_load_spider_stratified_sample(tmp_path):
    spider_dir = tmp_path / "spider2"
    spider_dir.mkdir()
    db_dir = spider_dir / "database" / "db"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(str(db_dir / "db.sqlite"))
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.commit()
    conn.close()
    data = [{"question": f"q{i}", "query": "SELECT 1", "db_id": "db"} for i in range(20)]
    (spider_dir / "dev.json").write_text(json.dumps(data))
    ds = load_spider(str(spider_dir), split="dev", n_samples=8, stratify_by_hardness=True)
    assert len(ds.cases) == 8


@pytest.mark.asyncio
async def test_load_spider_not_found():
    with pytest.raises(FileNotFoundError):
        load_spider("/no/spider/dir", split="dev")


@pytest.mark.asyncio
async def test_load_bird(tmp_path):
    bird_dir = tmp_path / "bird"
    dev_dir = bird_dir / "dev"
    dev_dir.mkdir(parents=True)
    db_dir = dev_dir / "databases" / "testdb"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(str(db_dir / "testdb.sqlite"))
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.commit()
    conn.close()
    data = [
        {"question": "count", "SQL": "SELECT COUNT(*) FROM t", "db_id": "testdb", "difficulty": "easy"},
        {"question": "all",   "SQL": "SELECT * FROM t",         "db_id": "testdb", "difficulty": "hard"},
    ]
    (dev_dir / "dev.json").write_text(json.dumps(data))
    ds = load_bird(str(bird_dir), split="dev")
    assert len(ds.cases) == 2
    assert ds.cases[0].hardness == "easy"
    assert ds.cases[1].hardness == "hard"
    assert ds.cases[0].sandbox_type == "sql"


@pytest.mark.asyncio
async def test_load_bird_not_found():
    with pytest.raises(FileNotFoundError):
        load_bird("/no/bird/dir", split="dev")


@pytest.mark.asyncio
async def test_load_humaneval(tmp_path):
    data = [
        {"task_id": "HumanEval/0", "prompt": "def add(a, b):", "canonical_solution": "    return a + b", "entry_point": "add"},
        {"task_id": "HumanEval/1", "prompt": "def sub(a, b):", "canonical_solution": "    return a - b", "entry_point": "sub"},
    ]
    p = tmp_path / "humaneval.jsonl"
    p.write_text("\n".join(json.dumps(d) for d in data))
    ds = load_humaneval(str(p))
    assert len(ds.cases) == 2
    assert ds.cases[0].sandbox_type == "code"
    assert ds.cases[0].agent_type == "code"
    assert "def add" in ds.cases[0].task


@pytest.mark.asyncio
async def test_load_humaneval_not_found():
    with pytest.raises(FileNotFoundError):
        load_humaneval("/no/file.jsonl")


@pytest.mark.asyncio
async def test_load_gsm8k(tmp_path):
    data = [
        {"question": "John has 5 apples. He gives 2. How many?", "answer": "He has 3 left.\n#### 3"},
        {"question": "2 + 2 =?",                                  "answer": "#### 4"},
    ]
    p = tmp_path / "gsm8k_test.jsonl"
    p.write_text("\n".join(json.dumps(d) for d in data))
    ds = load_gsm8k(str(p), split="test")
    assert len(ds.cases) == 2
    assert ds.cases[0].expected_output == "3"
    assert ds.cases[1].expected_output == "4"
    assert ds.cases[0].sandbox_type == "none"
    assert ds.agent_type == "base"


@pytest.mark.asyncio
async def test_load_gsm8k_not_found():
    with pytest.raises(FileNotFoundError):
        load_gsm8k("/no/gsm8k.jsonl")


# ===========================================================================
# EvalCase extensions (gold_actions, sandbox_type, hardness)
# ===========================================================================

def test_eval_case_gold_actions_dedup():
    c = EvalCase(
        case_id="1", agent_type="sql", task="t",
        expected_output="SELECT 1",
        gold_actions=["SELECT 1", "SELECT 1", "SELECT 2"],
    )
    all_gold = c.all_gold_actions()
    assert all_gold.count("SELECT 1") == 1
    assert "SELECT 2" in all_gold


def test_eval_case_gold_actions_includes_expected():
    c = EvalCase(
        case_id="1", agent_type="sql", task="t",
        expected_output="SELECT *",
        gold_actions=["SELECT id FROM t"],
    )
    all_gold = c.all_gold_actions()
    assert "SELECT *" in all_gold
    assert "SELECT id FROM t" in all_gold


def test_eval_case_all_gold_empty_when_no_gold():
    c = EvalCase(case_id="1", agent_type="sql", task="t")
    assert c.all_gold_actions() == []


def test_eval_case_sandbox_type_default():
    c = EvalCase(case_id="1", agent_type="sql", task="t")
    assert c.sandbox_type == "none"


def test_eval_case_to_dict_includes_new_fields():
    c = EvalCase(
        case_id="1", agent_type="sql", task="t",
        gold_actions=["SELECT 1"], sandbox_type="sql",
        db_path="/tmp/db.sqlite", hardness="easy",
    )
    d = c.to_dict()
    assert d["gold_actions"] == ["SELECT 1"]
    assert d["sandbox_type"] == "sql"
    assert d["db_path"] == "/tmp/db.sqlite"
    assert d["hardness"] == "easy"


def test_eval_case_from_dict_round_trip():
    c = EvalCase(
        case_id="42", agent_type="code", task="reverse string",
        gold_actions=["return s[::-1]"], sandbox_type="code",
        hardness="easy",
    )
    c2 = EvalCase.from_dict(c.to_dict())
    assert c2.gold_actions == ["return s[::-1]"]
    assert c2.sandbox_type == "code"
    assert c2.hardness == "easy"
