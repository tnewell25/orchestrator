"""Hybrid retrieval — recency × reinforcement × graph proximity.

Pgvector path is integration-tested elsewhere; these tests use the sqlite
fallback (synthetic similarity = 0.5) so we can verify the SCORING math
deterministically. The graph proximity boost is the headline feature.
"""
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from orchestrator.core.constants import EdgeKind, EntityType, GRAPH_PROXIMITY_BONUS
from orchestrator.core.graph import EntityRef, GraphStore
from orchestrator.core.memory import (
    MemoryStore,
    _hybrid_score,
    _recency_factor,
    _reinforcement_boost,
)
from orchestrator.db.models import Base, SemanticMemory


class _FakeEmbedder:
    """Returns a fixed 4-dim vector regardless of input — fine because the
    sqlite fallback path doesn't actually compute similarity."""
    def embed(self, texts):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


@pytest_asyncio.fixture
async def memory_store(session_maker):
    """A real MemoryStore wired to in-memory sqlite, with the embedder mocked."""
    # Use the same engine so we share state with the session_maker fixture.
    # Construct via __new__ to skip the URL parsing / engine creation in __init__.
    store = MemoryStore.__new__(MemoryStore)
    store.engine = session_maker.kw["bind"]
    store.session_maker = session_maker
    store.embedding_dim = 4
    store._embedder = _FakeEmbedder()
    store._extractor = None
    store._llm_extract_in_background = False
    store._vector_enabled = False  # forces sqlite fallback path
    yield store


# -----------------------------
# Pure scoring functions
# -----------------------------


def test_recency_factor_is_1_for_now():
    now = datetime.now(timezone.utc)
    assert _recency_factor(now, now) == pytest.approx(1.0)


def test_recency_factor_decays_with_age():
    now = datetime.now(timezone.utc)
    # At one half-life (14 days), the decaying portion halves: 0.4 + 0.6 * 0.5 = 0.7
    one_half_life_ago = now - timedelta(days=14)
    assert _recency_factor(one_half_life_ago, now) == pytest.approx(0.7)


def test_recency_factor_floors_at_04():
    now = datetime.now(timezone.utc)
    ancient = now - timedelta(days=365 * 5)
    assert _recency_factor(ancient, now) == pytest.approx(0.4, abs=0.01)


def test_reinforcement_boost_zero_for_count_1():
    assert _reinforcement_boost(1) == pytest.approx(0.0)


def test_reinforcement_boost_grows_logarithmically():
    boost_10 = _reinforcement_boost(10)
    boost_100 = _reinforcement_boost(100)
    # log(100)/log(10) == 2, so boost_100 = 2 × boost_10
    assert boost_100 == pytest.approx(2 * boost_10, rel=0.01)


def test_hybrid_score_proximity_bonus():
    near = _hybrid_score(0.5, 1.0, 1, proximity=1.0)
    far = _hybrid_score(0.5, 1.0, 1, proximity=0.0)
    assert near - far == pytest.approx(GRAPH_PROXIMITY_BONUS)


# -----------------------------
# recall_for_entity (graph-only path)
# -----------------------------


@pytest.mark.asyncio
async def test_recall_for_entity_returns_proximate_memories(session_maker, memory_store):
    """Memories edge-linked to focus entity should surface; unrelated ones don't."""
    graph = GraphStore(session_maker)
    memory_store.attach_graph(graph)

    # Seed two memories — one linked to a deal, one not
    linked_id = await memory_store.store_memory("Bosch deal moving forward", source="meeting")
    unlinked_id = await memory_store.store_memory("Random thought about lunch", source="conversation")

    deal_ref = EntityRef(EntityType.DEAL, "deal-bosch")
    await graph.add_edge(
        EntityRef(EntityType.MEMORY, linked_id), deal_ref, EdgeKind.MENTIONS
    )

    results = await memory_store.recall_for_entity(deal_ref, limit=10)
    ids = {r["id"] for r in results}
    assert linked_id in ids
    assert unlinked_id not in ids


@pytest.mark.asyncio
async def test_recall_for_entity_returns_empty_when_no_graph(session_maker):
    store = MemoryStore.__new__(MemoryStore)
    store.session_maker = session_maker
    store._embedder = _FakeEmbedder()
    store._vector_enabled = False
    # Note: NOT calling attach_graph
    results = await store.recall_for_entity(
        EntityRef(EntityType.DEAL, "x"), limit=5,
    )
    assert results == []


# -----------------------------
# recall_hybrid (vector + scoring)
# -----------------------------


@pytest.mark.asyncio
async def test_recall_hybrid_boosts_proximate_memory(session_maker, memory_store):
    """Two memories with identical (synthetic) similarity — the one linked to
    focus_ref should rank ahead."""
    graph = GraphStore(session_maker)
    memory_store.attach_graph(graph)

    proximate_id = await memory_store.store_memory("Memory near focus", source="meeting")
    distant_id = await memory_store.store_memory("Memory far from focus", source="conversation")

    focus = EntityRef(EntityType.DEAL, "d-focus")
    await graph.add_edge(
        EntityRef(EntityType.MEMORY, proximate_id), focus, EdgeKind.MENTIONS
    )

    results = await memory_store.recall_hybrid(query="anything", focus_ref=focus, limit=2)
    assert len(results) == 2
    assert results[0]["id"] == proximate_id
    assert results[0]["proximity"] == 1.0
    assert results[1]["proximity"] == 0.0


@pytest.mark.asyncio
async def test_recall_hybrid_recent_memory_outranks_old(session_maker, memory_store):
    """All else equal, newer reinforced_at wins."""
    # Insert two memories then directly mutate timestamps to simulate age
    new_id = await memory_store.store_memory("Recent memory")
    old_id = await memory_store.store_memory("Old memory")

    long_ago = datetime.now(timezone.utc) - timedelta(days=200)
    async with session_maker() as s:
        old = await s.get(SemanticMemory, old_id)
        old.timestamp = long_ago
        old.last_reinforced_at = long_ago
        await s.commit()

    results = await memory_store.recall_hybrid(query="anything", limit=2)
    # Sort by score: new_id should be first
    assert results[0]["id"] == new_id
    assert results[0]["recency"] > results[1]["recency"]


@pytest.mark.asyncio
async def test_recall_hybrid_reinforcement_boosts_score(session_maker, memory_store):
    a_id = await memory_store.store_memory("A")
    b_id = await memory_store.store_memory("B")

    async with session_maker() as s:
        a = await s.get(SemanticMemory, a_id)
        a.reinforcement_count = 50
        await s.commit()

    results = await memory_store.recall_hybrid(query="anything", limit=2)
    a_score = next(r["score"] for r in results if r["id"] == a_id)
    b_score = next(r["score"] for r in results if r["id"] == b_id)
    assert a_score > b_score


@pytest.mark.asyncio
async def test_recall_backwards_compatible_without_focus(session_maker, memory_store):
    """The original recall(query, limit) signature still works without focus_ref."""
    await memory_store.store_memory("test memory")
    results = await memory_store.recall("query", limit=5)
    assert isinstance(results, list)
    assert len(results) >= 1
    # Result schema includes the new score field
    assert "score" in results[0]
