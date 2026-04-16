"""Structured memory fields — title, facts, concepts round-trip + Block D render."""
import json

import pytest
import pytest_asyncio
from sqlalchemy import select

from orchestrator.core.memory import MemoryStore, _json_load_list
from orchestrator.core.planner import Intent, Plan
from orchestrator.core.prompt_assembler import PromptAssembler, _format_memory_line
from orchestrator.db.models import SemanticMemory


class _FakeEmbedder:
    def embed(self, texts):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


@pytest_asyncio.fixture
async def store(session_maker):
    s = MemoryStore.__new__(MemoryStore)
    s.engine = session_maker.kw["bind"]
    s.session_maker = session_maker
    s.embedding_dim = 4
    s._embedder = _FakeEmbedder()
    s._extractor = None
    s._llm_extract_in_background = False
    s._vector_enabled = False
    yield s


@pytest.mark.asyncio
async def test_structured_fields_round_trip(store, session_maker):
    await store.store_memory(
        "Met with Anja at Bosch about pricing",
        title="Meeting 2026-04-16 with Anja",
        facts=["Pricing discussed", "Close-date Q3"],
        concepts=["Anja Weber", "Bosch"],
    )
    async with session_maker() as s:
        row = (await s.execute(select(SemanticMemory))).scalar_one()
    assert row.title == "Meeting 2026-04-16 with Anja"
    assert "Pricing discussed" in row.facts_json
    assert "Bosch" in row.concepts_json


@pytest.mark.asyncio
async def test_recall_hybrid_returns_structured_fields(store):
    await store.store_memory(
        "body content",
        title="Short title",
        facts=["fact A", "fact B"],
        concepts=["concept X"],
    )
    results = await store.recall_hybrid("anything", limit=5)
    assert results[0]["title"] == "Short title"
    assert results[0]["facts"] == ["fact A", "fact B"]
    assert results[0]["concepts"] == ["concept X"]


@pytest.mark.asyncio
async def test_unstructured_memory_still_works(store):
    """Back-compat: old callers that don't pass structured args get empty lists."""
    await store.store_memory("just content")
    results = await store.recall_hybrid("anything", limit=5)
    assert results[0]["title"] == ""
    assert results[0]["facts"] == []
    assert results[0]["concepts"] == []


# ---- Block D formatter -----------------------------------------------


def test_format_memory_line_uses_title_and_facts_when_both_present():
    line = _format_memory_line({
        "source": "meeting",
        "title": "Meeting 2026-04-16 with Anja",
        "facts": ["Pricing discussed", "Close-date Q3", "Legal review by EOM"],
        "content": "Long transcript...",
    })
    assert "[meeting]" in line
    assert "Meeting 2026-04-16 with Anja" in line
    assert "Pricing discussed" in line
    # 3 facts max — 4th should be dropped
    assert "Legal review" in line


def test_format_memory_line_falls_back_to_content_without_title():
    line = _format_memory_line({"source": "voice", "content": "random thought"})
    assert line.strip() == "- [voice] random thought"


def test_format_memory_line_with_title_only():
    line = _format_memory_line({
        "source": "mem", "title": "Summary",
        "content": "original long content to truncate",
        "facts": [],
    })
    assert "Summary" in line
    assert "original long content" in line


# ---- PromptAssembler integration -------------------------------------


def test_block_d_uses_structured_memories():
    asm = PromptAssembler()
    p = asm.assemble(
        facts=[], plan=Plan(intent=Intent.QUERY),
        memories=[{
            "source": "meeting",
            "title": "Bosch prep call",
            "facts": ["Anja agreed on timeline"],
            "content": "full transcript",
        }],
    )
    assert "Bosch prep call" in p.block_d
    assert "Anja agreed on timeline" in p.block_d


# ---- JSON helper -----------------------------------------------------


def test_json_load_list_tolerant():
    assert _json_load_list("") == []
    assert _json_load_list(None) == []
    assert _json_load_list("not json") == []
    assert _json_load_list('{"not": "list"}') == []
    assert _json_load_list('["a", "b"]') == ["a", "b"]
