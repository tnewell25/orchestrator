"""Compactor — threshold-driven summarization, mark-as-compacted, brief retrieval."""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.core.compactor import Compactor
from orchestrator.db.models import Conversation, SessionBrief


def _fake_anthropic_client(summary_text: str = "Summary of older turns."):
    """Minimal Anthropic client that returns a canned summary."""
    block = MagicMock(type="text", text=summary_text)
    resp = MagicMock(content=[block])
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=resp)
    return client


async def _seed_messages(session_maker, session_id: str, n: int):
    base = datetime.now(timezone.utc) - timedelta(minutes=n)
    async with session_maker() as s:
        for i in range(n):
            s.add(Conversation(
                id=f"conv-{i}",
                session_id=session_id,
                role="user" if i % 2 == 0 else "assistant",
                content=f"message {i}",
                timestamp=base + timedelta(seconds=i),
            ))
        await s.commit()


@pytest.mark.asyncio
async def test_compactor_skips_below_threshold(session_maker):
    await _seed_messages(session_maker, "s1", n=10)
    compactor = Compactor(
        session_maker, _fake_anthropic_client(),
        compact_threshold=40, keep_recent=15,
    )
    brief = await compactor.maybe_compact("s1")
    assert brief is None


@pytest.mark.asyncio
async def test_compactor_compacts_older_keeps_recent(session_maker):
    await _seed_messages(session_maker, "s1", n=50)
    compactor = Compactor(
        session_maker, _fake_anthropic_client("rolled-up summary"),
        compact_threshold=40, keep_recent=15,
    )

    brief = await compactor.maybe_compact("s1")
    assert brief is not None
    assert brief.summary == "rolled-up summary"
    assert brief.rows_compacted == 35  # 50 - 15

    # The 35 oldest rows should be marked compacted_into
    from sqlalchemy import select, func
    async with session_maker() as s:
        compacted = (await s.execute(
            select(func.count()).select_from(Conversation).where(
                Conversation.session_id == "s1",
                Conversation.compacted_into.is_not(None),
            )
        )).scalar()
        active = (await s.execute(
            select(func.count()).select_from(Conversation).where(
                Conversation.session_id == "s1",
                Conversation.compacted_into.is_(None),
            )
        )).scalar()
    assert compacted == 35
    assert active == 15


@pytest.mark.asyncio
async def test_compactor_idempotent_on_no_growth(session_maker):
    await _seed_messages(session_maker, "s1", n=50)
    compactor = Compactor(
        session_maker, _fake_anthropic_client(),
        compact_threshold=40, keep_recent=15,
    )

    first = await compactor.maybe_compact("s1")
    assert first is not None
    # After compaction only 15 rows are uncompacted; second call should be no-op
    second = await compactor.maybe_compact("s1")
    assert second is None


@pytest.mark.asyncio
async def test_compactor_handles_llm_failure_gracefully(session_maker):
    await _seed_messages(session_maker, "s1", n=50)
    bad_client = MagicMock()
    bad_client.messages = MagicMock()
    bad_client.messages.create = AsyncMock(side_effect=RuntimeError("api down"))

    compactor = Compactor(session_maker, bad_client, compact_threshold=40, keep_recent=15)
    brief = await compactor.maybe_compact("s1")
    assert brief is None  # fail-soft, no crash, no rows marked


@pytest.mark.asyncio
async def test_compactor_skips_without_client(session_maker):
    await _seed_messages(session_maker, "s1", n=50)
    compactor = Compactor(session_maker, anthropic_client=None,
                          compact_threshold=40, keep_recent=15)
    brief = await compactor.maybe_compact("s1")
    assert brief is None


# ---- MemoryStore integration --------------------------------------


@pytest.mark.asyncio
async def test_get_conversation_excludes_compacted_rows(session_maker):
    await _seed_messages(session_maker, "s1", n=50)
    compactor = Compactor(
        session_maker, _fake_anthropic_client("brief"),
        compact_threshold=40, keep_recent=15,
    )
    await compactor.maybe_compact("s1")

    # Build a minimal MemoryStore-like wrapper that exercises the SAME query
    from orchestrator.core.memory import MemoryStore
    store = MemoryStore.__new__(MemoryStore)
    store.session_maker = session_maker

    rows = await store.get_conversation("s1", limit=100)
    assert len(rows) == 15  # only active window


@pytest.mark.asyncio
async def test_get_latest_session_brief_returns_summary(session_maker):
    await _seed_messages(session_maker, "s1", n=50)
    compactor = Compactor(
        session_maker, _fake_anthropic_client("the summary"),
        compact_threshold=40, keep_recent=15,
    )
    await compactor.maybe_compact("s1")

    from orchestrator.core.memory import MemoryStore
    store = MemoryStore.__new__(MemoryStore)
    store.session_maker = session_maker

    summary = await store.get_latest_session_brief("s1")
    assert summary == "the summary"


@pytest.mark.asyncio
async def test_get_latest_session_brief_empty_when_uncompacted(session_maker):
    await _seed_messages(session_maker, "s1", n=10)

    from orchestrator.core.memory import MemoryStore
    store = MemoryStore.__new__(MemoryStore)
    store.session_maker = session_maker

    assert await store.get_latest_session_brief("s1") == ""
