"""Usage logging — AuditLog captures token cost per messages.create round-trip."""
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from orchestrator.core.audit import AuditLogger
from orchestrator.db.models import AuditLog


class _FakeUsage:
    """Mirrors the shape of anthropic.types.Usage."""
    def __init__(self, input=0, output=0, cache_read=0, cache_creation=0):
        self.input_tokens = input
        self.output_tokens = output
        self.cache_read_input_tokens = cache_read
        self.cache_creation_input_tokens = cache_creation


@pytest.mark.asyncio
async def test_log_usage_writes_turn_row(session_maker):
    audit = AuditLogger(session_maker)
    usage = _FakeUsage(input=1000, output=200, cache_read=500, cache_creation=50)

    await audit.log_usage("s1", "claude-haiku-4-5", usage, duration_ms=123, iteration=0)

    async with session_maker() as s:
        rows = (await s.execute(select(AuditLog))).scalars().all()
    assert len(rows) == 1
    row = rows[0]
    assert row.tool_name == "_turn"
    assert row.model == "claude-haiku-4-5"
    assert row.input_tokens == 1000
    assert row.output_tokens == 200
    assert row.cache_read_tokens == 500
    assert row.cache_creation_tokens == 50
    assert row.duration_ms == 123


@pytest.mark.asyncio
async def test_log_usage_handles_partial_data(session_maker):
    """Sonnet may return different field availability; missing fields default to 0."""
    audit = AuditLogger(session_maker)

    class PartialUsage:
        input_tokens = 100
        output_tokens = 50
        # no cache fields

    await audit.log_usage("s2", "claude-sonnet-4-5", PartialUsage(), duration_ms=10)
    async with session_maker() as s:
        row = (await s.execute(select(AuditLog))).scalar_one()
    assert row.input_tokens == 100
    assert row.cache_read_tokens == 0
    assert row.cache_creation_tokens == 0


@pytest.mark.asyncio
async def test_log_usage_accepts_dict_like_usage(session_maker):
    audit = AuditLogger(session_maker)
    await audit.log_usage("s3", "m", {
        "input_tokens": 500, "output_tokens": 100,
        "cache_read_input_tokens": 300, "cache_creation_input_tokens": 20,
    })
    async with session_maker() as s:
        row = (await s.execute(select(AuditLog))).scalar_one()
    assert row.input_tokens == 500
    assert row.cache_read_tokens == 300


@pytest.mark.asyncio
async def test_log_usage_none_input_safe(session_maker):
    """usage=None shouldn't crash."""
    audit = AuditLogger(session_maker)
    await audit.log_usage("s4", "m", None)
    async with session_maker() as s:
        row = (await s.execute(select(AuditLog))).scalar_one()
    assert row.input_tokens == 0


@pytest.mark.asyncio
async def test_tool_call_audit_unchanged(session_maker):
    """Existing per-tool log() path still works and has zero token fields."""
    audit = AuditLogger(session_maker)
    await audit.log(
        tool_name="deal-find", args={"q": "x"},
        result_status="ok", result_summary="found 3",
        session_id="s5", duration_ms=45,
    )
    async with session_maker() as s:
        row = (await s.execute(select(AuditLog))).scalar_one()
    assert row.tool_name == "deal-find"
    assert row.input_tokens == 0  # not populated for tool rows
    assert row.output_tokens == 0
