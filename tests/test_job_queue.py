"""JobQueue — CLAIM-CONFIRM pattern, stuck recovery, retry-then-abandon."""
import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from orchestrator.core.job_queue import JobQueue, MAX_ATTEMPTS
from orchestrator.db.models import BackgroundJob


@pytest.mark.asyncio
async def test_enqueue_and_claim_roundtrip(session_maker):
    q = JobQueue(session_maker)
    j = await q.enqueue("compaction", {"session_id": "s1"})
    assert j.status == "pending"

    claimed = await q.claim("compaction")
    assert claimed is not None
    assert claimed.id == j.id
    assert claimed.status == "processing"


@pytest.mark.asyncio
async def test_claim_returns_none_when_empty(session_maker):
    q = JobQueue(session_maker)
    assert await q.claim() is None


@pytest.mark.asyncio
async def test_claim_filters_by_type(session_maker):
    q = JobQueue(session_maker)
    await q.enqueue("compaction", {})
    await q.enqueue("extraction", {})

    only_extraction = await q.claim("extraction")
    assert only_extraction.job_type == "extraction"


@pytest.mark.asyncio
async def test_confirm_marks_completed(session_maker):
    q = JobQueue(session_maker)
    j = await q.enqueue("x")
    c = await q.claim()
    await q.confirm(c.id)

    async with session_maker() as s:
        row = await s.get(BackgroundJob, j.id)
    assert row.status == "completed"


@pytest.mark.asyncio
async def test_fail_increments_attempts_and_resets_to_pending(session_maker):
    q = JobQueue(session_maker)
    j = await q.enqueue("x")
    await q.claim()
    failed = await q.fail(j.id, "boom")
    assert failed.attempts == 1
    assert failed.status == "pending"   # eligible for retry
    assert "boom" in failed.last_error


@pytest.mark.asyncio
async def test_fail_abandons_after_max_attempts(session_maker):
    q = JobQueue(session_maker)
    j = await q.enqueue("x")
    for _ in range(MAX_ATTEMPTS):
        await q.claim()
        await q.fail(j.id, "boom")
    async with session_maker() as s:
        row = await s.get(BackgroundJob, j.id)
    assert row.status == "abandoned"


@pytest.mark.asyncio
async def test_recover_stuck_resets_old_processing_rows(session_maker):
    q = JobQueue(session_maker)
    j = await q.enqueue("x")
    await q.claim()   # now processing

    # Simulate stuck: updated_at way in the past
    async with session_maker() as s:
        row = await s.get(BackgroundJob, j.id)
        row.updated_at = datetime.now(timezone.utc) - timedelta(hours=1)
        await s.commit()

    reset = await q.recover_stuck()
    assert reset == 1

    async with session_maker() as s:
        row = await s.get(BackgroundJob, j.id)
    assert row.status == "pending"
    assert row.attempts == 1


@pytest.mark.asyncio
async def test_recover_stuck_abandons_after_retry_budget(session_maker):
    q = JobQueue(session_maker)
    j = await q.enqueue("x")
    async with session_maker() as s:
        row = await s.get(BackgroundJob, j.id)
        row.status = "processing"
        row.attempts = MAX_ATTEMPTS - 1  # already used retries
        row.updated_at = datetime.now(timezone.utc) - timedelta(hours=1)
        await s.commit()

    await q.recover_stuck()
    async with session_maker() as s:
        row = await s.get(BackgroundJob, j.id)
    assert row.status == "abandoned"


@pytest.mark.asyncio
async def test_payload_of_tolerates_garbage(session_maker):
    q = JobQueue(session_maker)
    j = await q.enqueue("x", {"a": 1})
    assert (await q.payload_of(j)) == {"a": 1}

    async with session_maker() as s:
        row = await s.get(BackgroundJob, j.id)
        row.payload = "not json"
        await s.commit()
    assert (await q.payload_of(row)) == {}
