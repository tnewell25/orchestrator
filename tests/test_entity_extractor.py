"""EntityExtractor — substring path correctness and LLM JSON parsing.

The LLM client is mocked (no network). Real LLM behavior is tested by
integration suites against the live API.
"""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.core.constants import EdgeKind, EntityType
from orchestrator.core.entity_extractor import (
    EntityExtractor,
    EntityIndex,
    _parse_llm_response,
)
from orchestrator.core.graph import EntityRef, GraphStore
from orchestrator.db.models import Company, Contact, Deal


# -----------------------------
# Substring index tests
# -----------------------------


def test_index_lookup_whole_word_only():
    idx = EntityIndex()
    idx.add("Markus", EntityRef(EntityType.CONTACT, "c1"))

    matches = idx.lookup_substring("Met with Markus today.")
    assert len(matches) == 1
    assert matches[0].ref.id == "c1"

    # "Mark" inside "Markus" — should NOT match (word boundary)
    idx2 = EntityIndex()
    idx2.add("Mark", EntityRef(EntityType.CONTACT, "c2"))
    no_match = idx2.lookup_substring("Met Markus")
    assert no_match == []


def test_index_lookup_case_insensitive():
    idx = EntityIndex()
    idx.add("Bosch", EntityRef(EntityType.COMPANY, "b1"))

    matches = idx.lookup_substring("bosch is going well")
    assert len(matches) == 1


def test_index_skips_stopwords():
    idx = EntityIndex()
    idx.add("the", EntityRef(EntityType.CONTACT, "x"))  # bogus
    idx.add("Anja", EntityRef(EntityType.CONTACT, "a"))

    matches = idx.lookup_substring("the meeting with Anja was good")
    assert len(matches) == 1
    assert matches[0].ref.id == "a"


def test_ambiguous_name_returns_all_with_reduced_confidence():
    idx = EntityIndex()
    idx.add("Markus", EntityRef(EntityType.CONTACT, "markus-1"))
    idx.add("Markus", EntityRef(EntityType.CONTACT, "markus-2"))

    matches = idx.lookup_substring("Markus called")
    assert len(matches) == 2
    for m in matches:
        assert m.confidence < 0.95  # reduced because ambiguous


# -----------------------------
# Extractor write-path
# -----------------------------


@pytest.mark.asyncio
async def test_extract_substring_creates_edges(session_maker):
    """Seed a contact + run extraction → edge written to graph."""
    async with session_maker() as s:
        s.add(Contact(id="c-anja", name="Anja Weber"))
        s.add(Contact(id="c-markus", name="Markus Schulz"))
        s.add(Company(id="comp-bosch", name="Bosch"))
        await s.commit()

    graph = GraphStore(session_maker)
    ext = EntityExtractor(session_maker, graph, anthropic_client=None)
    await ext.refresh_index()

    source = EntityRef(EntityType.MEMORY, "m1")
    result = await ext.extract(
        "Met with Anja Weber at Bosch — she introduced me to Markus.",
        source,
    )

    refs_to = {e[1] for e in result.edges}
    assert EntityRef(EntityType.CONTACT, "c-anja") in refs_to
    assert EntityRef(EntityType.COMPANY, "comp-bosch") in refs_to
    assert EntityRef(EntityType.CONTACT, "c-markus") in refs_to

    # Verify edges actually persisted
    edges = await graph.neighbors(source)
    assert len(edges) == 3


@pytest.mark.asyncio
async def test_extract_idempotent_reinforces(session_maker):
    """Running the same extraction twice should NOT create duplicate edges."""
    async with session_maker() as s:
        s.add(Contact(id="c-anja", name="Anja"))
        await s.commit()

    graph = GraphStore(session_maker)
    ext = EntityExtractor(session_maker, graph, anthropic_client=None)
    await ext.refresh_index()

    source = EntityRef(EntityType.MEMORY, "m-dup")
    text = "Anja called today"

    await ext.extract(text, source)
    await ext.extract(text, source)

    edges = await graph.neighbors(source)
    assert len(edges) == 1
    assert edges[0].reinforcement_count == 2


@pytest.mark.asyncio
async def test_first_name_alias_added(session_maker):
    async with session_maker() as s:
        s.add(Contact(id="c1", name="Anja Weber"))
        await s.commit()

    graph = GraphStore(session_maker)
    ext = EntityExtractor(session_maker, graph, anthropic_client=None)
    await ext.refresh_index()

    # First-name only should still match
    source = EntityRef(EntityType.MEMORY, "m1")
    result = await ext.extract("Anja was great", source)
    refs = {e[1] for e in result.edges}
    assert EntityRef(EntityType.CONTACT, "c1") in refs


# -----------------------------
# LLM extractor (mocked)
# -----------------------------


@pytest.mark.asyncio
async def test_llm_extractor_parses_response_and_writes_edges(session_maker):
    async with session_maker() as s:
        s.add(Contact(id="c1", name="Anja"))
        await s.commit()

    graph = GraphStore(session_maker)

    # Mock anthropic client returning a JSON array
    payload = json.dumps([
        {"kind": "Contact", "id": "c1", "name": "Anja", "matched": "she", "confidence": 0.85},
        {"kind": "Company", "id": "", "name": "Acme Corp", "matched": "Acme", "confidence": 0.7},
    ])
    fake_block = MagicMock()
    fake_block.type = "text"
    fake_block.text = payload
    fake_resp = MagicMock()
    fake_resp.content = [fake_block]

    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=fake_resp)

    ext = EntityExtractor(session_maker, graph, anthropic_client=client)
    await ext.refresh_index()

    source = EntityRef(EntityType.MEMORY, "m1")
    result = await ext.extract_llm("She mentioned Acme.", source)

    # Should have one linked edge (Anja) and one is_new candidate (Acme)
    assert len(result.edges) == 1  # only the existing-id link is written
    assert result.edges[0][1] == EntityRef(EntityType.CONTACT, "c1")
    assert any(e.is_new and e.proposed_name == "Acme Corp" for e in result.entities)


def test_parse_llm_response_handles_markdown_fences():
    raw = "```json\n[{\"kind\":\"Contact\",\"id\":\"x\",\"name\":\"X\",\"matched\":\"X\"}]\n```"
    parsed = _parse_llm_response(raw)
    assert len(parsed) == 1
    assert parsed[0]["kind"] == "Contact"


def test_parse_llm_response_extracts_array_from_prose():
    raw = "Sure, here's the JSON: [{\"kind\": \"Deal\", \"id\": \"d1\", \"name\": \"x\"}] hope this helps"
    parsed = _parse_llm_response(raw)
    assert parsed and parsed[0]["kind"] == "Deal"


def test_parse_llm_response_returns_empty_on_garbage():
    assert _parse_llm_response("") == []
    assert _parse_llm_response("not json at all") == []
