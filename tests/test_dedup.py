"""Content-hash dedup on memory.remember() — reinforces instead of duplicating."""
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import func, select

from orchestrator.core.memory import MemoryStore, _content_hash
from orchestrator.db.models import SemanticMemory


class _FakeEmbedder:
    def __init__(self):
        self.call_count = 0
    def embed(self, texts):
        self.call_count += len(list(texts) if not isinstance(texts, list) else texts)
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


def test_content_hash_is_stable_and_case_insensitive():
    assert _content_hash("Hello World") == _content_hash("hello world")
    assert _content_hash("  Foo  ") == _content_hash("foo")
    assert _content_hash("a") != _content_hash("b")
    assert len(_content_hash("anything")) == 16


@pytest.mark.asyncio
async def test_identical_content_within_window_reinforces_instead_of_inserting(store, session_maker):
    first_id = await store.store_memory("Markus prefers email")
    second_id = await store.store_memory("Markus prefers email")
    third_id = await store.store_memory("  MARKUS PREFERS EMAIL  ")  # normalized → same hash

    assert first_id == second_id == third_id

    async with session_maker() as s:
        count = (await s.execute(select(func.count()).select_from(SemanticMemory))).scalar()
        row = await s.get(SemanticMemory, first_id)
    assert count == 1
    assert row.reinforcement_count == 3


@pytest.mark.asyncio
async def test_different_content_inserts_new_row(store, session_maker):
    a = await store.store_memory("Markus prefers email")
    b = await store.store_memory("Anja prefers Slack")
    assert a != b
    async with session_maker() as s:
        count = (await s.execute(select(func.count()).select_from(SemanticMemory))).scalar()
    assert count == 2


@pytest.mark.asyncio
async def test_dedup_skips_embedding_call_on_hit(store):
    await store.store_memory("something unique")
    embed_count_after_first = store._embedder.call_count

    # Second call with the same content should be a dedup hit and skip embed
    await store.store_memory("something unique")
    assert store._embedder.call_count == embed_count_after_first


@pytest.mark.asyncio
async def test_dedup_expires_after_window(store, session_maker):
    """Content older than 60s is no longer deduped — a legitimate fresh mention
    of the same fact creates a new row (still reinforcement_count=1)."""
    first_id = await store.store_memory("stale fact")

    async with session_maker() as s:
        row = await s.get(SemanticMemory, first_id)
        row.last_reinforced_at = datetime.now(timezone.utc) - timedelta(seconds=120)
        row.timestamp = datetime.now(timezone.utc) - timedelta(seconds=120)
        await s.commit()

    second_id = await store.store_memory("stale fact")
    assert second_id != first_id

    async with session_maker() as s:
        count = (await s.execute(select(func.count()).select_from(SemanticMemory))).scalar()
    assert count == 2


@pytest.mark.asyncio
async def test_content_hash_populated_on_new_insert(store, session_maker):
    new_id = await store.store_memory("indexed content")
    async with session_maker() as s:
        row = await s.get(SemanticMemory, new_id)
    assert row.content_hash == _content_hash("indexed content")
