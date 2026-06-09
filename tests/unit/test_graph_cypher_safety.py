"""Regression tests for Cypher label/relationship-type injection (graph.py item 6)."""

from __future__ import annotations

import pytest

from harness.core.errors import HarnessError
from harness.memory.graph import Neo4jGraphMemory, _safe_label


# ---------------------------------------------------------------------------
# _safe_label helper
# ---------------------------------------------------------------------------

def test_safe_label_accepts_plain_identifier():
    assert _safe_label("Entity", "label") == "Entity"
    assert _safe_label("USED_BY_QUERY", "relationship type") == "USED_BY_QUERY"
    assert _safe_label("Query_1", "label") == "Query_1"


@pytest.mark.parametrize(
    "bad",
    [
        "Entity) DELETE n //",          # break out of MERGE
        "Foo`bar",                       # backtick escape
        "Label WITH space",              # spaces
        "Label-dash",                    # hyphen
        "",                              # empty
        "Label;DROP",                    # statement separator
    ],
)
def test_safe_label_rejects_injection(bad):
    with pytest.raises(HarnessError):
        _safe_label(bad, "label")


# ---------------------------------------------------------------------------
# Neo4jGraphMemory.add_node / add_edge use the whitelist
# ---------------------------------------------------------------------------

class _CapturingNeo4j(Neo4jGraphMemory):
    """Stub that records the Cypher instead of hitting a real DB."""

    def __init__(self) -> None:
        super().__init__()
        self.queries: list[str] = []

    async def _run_query(self, cypher, params=None):  # type: ignore[override]
        self.queries.append(cypher)
        return []


@pytest.mark.asyncio
async def test_add_node_rejects_malicious_type():
    mem = _CapturingNeo4j()
    with pytest.raises(HarnessError):
        await mem.add_node(id="x", type="Entity) DETACH DELETE n //", props={})
    assert mem.queries == []  # nothing was executed


@pytest.mark.asyncio
async def test_add_edge_rejects_malicious_type():
    mem = _CapturingNeo4j()
    with pytest.raises(HarnessError):
        await mem.add_edge(src="a", tgt="b", type="REL]->() DELETE x //")
    assert mem.queries == []


@pytest.mark.asyncio
async def test_add_node_accepts_clean_type_and_interpolates():
    mem = _CapturingNeo4j()
    await mem.add_node(id="x", type="Entity", props={"name": "x"})
    assert mem.queries and "MERGE (n:Entity {id: $id})" in mem.queries[0]


@pytest.mark.asyncio
async def test_add_edge_accepts_clean_type_and_interpolates():
    mem = _CapturingNeo4j()
    await mem.add_edge(src="a", tgt="b", type="USED_BY_QUERY")
    assert mem.queries and "[r:USED_BY_QUERY]" in mem.queries[0]
