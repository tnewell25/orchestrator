"""Date/time anchor in prompt + timezone-aware reminder parsing.

Regression coverage for the user-reported bug where the agent thought it
was a prior year (training cutoff) and reminders never fired at wall-clock
time the user expected.
"""
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import select

from orchestrator.core.planner import Intent, Plan
from orchestrator.core.prompt_assembler import (
    PromptAssembler,
    build_datetime_header,
)
from orchestrator.core.reminder_service import (
    STALE_CUTOFF,
    ReminderService,
)
from orchestrator.db.models import Reminder
from orchestrator.skills.reminder_skill import ReminderSkill, _parse_when


# ---- Datetime header ---------------------------------------------


def test_datetime_header_includes_today_in_utc():
    header = build_datetime_header("UTC")
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert today_str in header
    assert "Current UTC" in header
    assert "do NOT infer the date" in header


def test_datetime_header_adds_local_time_for_non_utc():
    header = build_datetime_header("America/New_York")
    assert "User local time" in header
    assert "America/New_York" in header


def test_datetime_header_tolerates_bad_timezone():
    """Unknown timezone shouldn't crash — falls back to UTC-only."""
    header = build_datetime_header("Mars/Olympus_Mons")
    assert "Current UTC" in header
    assert "User local time" not in header


# ---- Block D includes the anchor --------------------------------


def test_block_d_always_starts_with_datetime_header():
    asm = PromptAssembler(user_timezone="UTC")
    p = asm.assemble(facts=[], memories=[], plan=Plan(intent=Intent.QUERY))
    assert p.block_d.startswith("CURRENT DATE/TIME")


def test_block_d_datetime_appears_even_when_ambiguous_intent():
    """AMBIGUOUS used to produce an empty Block D — must still inject datetime."""
    asm = PromptAssembler(user_timezone="UTC")
    p = asm.assemble(facts=[], memories=[], plan=Plan(intent=Intent.AMBIGUOUS))
    assert "CURRENT DATE/TIME" in p.block_d


def test_assembler_propagates_timezone_to_header():
    asm = PromptAssembler(user_timezone="Europe/Berlin")
    p = asm.assemble(facts=[], memories=[], plan=Plan(intent=Intent.QUERY))
    assert "Europe/Berlin" in p.block_d


# ---- Reminder parsing respects timezone --------------------------


def test_parse_when_respects_user_timezone():
    """Parsing '9am' in EST should give UTC 13:00 or 14:00 (depending on DST).
    Before fix, it parsed as UTC 9am — 4-5 hours too early for EST users."""
    dt = _parse_when("9am tomorrow", user_timezone="America/New_York")
    assert dt is not None
    # Should NOT be 9am UTC — that would be 4-5am EST which isn't what the user said
    assert dt.hour in (13, 14)  # 9am EST = 14:00 UTC (standard) or 13:00 UTC (DST)


def test_parse_when_utc_default_unchanged():
    dt = _parse_when("9am tomorrow", user_timezone="UTC")
    assert dt is not None
    assert dt.hour == 9


def test_parse_when_empty_returns_none():
    assert _parse_when("", user_timezone="UTC") is None


# ---- Stale-reminder guard ----------------------------------------


@pytest_asyncio.fixture
async def reminder_service(session_maker):
    # No telegram bot — _deliver will log the warning path, which is fine.
    svc = ReminderService(session_maker, telegram_bot=None, agent=None)
    yield svc


@pytest.mark.asyncio
async def test_stale_reminder_marked_skipped_not_delivered(reminder_service, session_maker):
    """A reminder set for years in the past (agent hallucination) should be
    marked stale, not fire a flood of belated pings."""
    async with session_maker() as s:
        # trigger_at 2 years ago
        ancient = datetime.now(timezone.utc) - timedelta(days=730)
        s.add(Reminder(
            id="stale-1", trigger_at=ancient, message="Old reminder",
            status="pending", kind="custom",
        ))
        # Also a legit reminder from 1 hour ago that should still fire
        recent_past = datetime.now(timezone.utc) - timedelta(hours=1)
        s.add(Reminder(
            id="recent-1", trigger_at=recent_past, message="Recent reminder",
            status="pending", kind="custom", target_chat_id="c1",
        ))
        await s.commit()

    await reminder_service._tick()

    async with session_maker() as s:
        stale = await s.get(Reminder, "stale-1")
        recent = await s.get(Reminder, "recent-1")
    assert stale.status == "stale"
    # Recent either went to sent or failed (no bot wired). Key: NOT stale.
    assert recent.status != "stale"


@pytest.mark.asyncio
async def test_stale_cutoff_boundary(reminder_service, session_maker):
    """Reminder exactly at boundary + a bit (stale) vs just inside (not stale)."""
    async with session_maker() as s:
        # Just past the boundary → stale
        over = datetime.now(timezone.utc) - STALE_CUTOFF - timedelta(minutes=5)
        s.add(Reminder(id="over", trigger_at=over, message="x", status="pending", kind="custom"))
        # Inside the window → should be processed, not stale
        under = datetime.now(timezone.utc) - STALE_CUTOFF + timedelta(minutes=5)
        s.add(Reminder(id="under", trigger_at=under, message="y", status="pending", kind="custom"))
        await s.commit()

    await reminder_service._tick()

    async with session_maker() as s:
        assert (await s.get(Reminder, "over")).status == "stale"
        assert (await s.get(Reminder, "under")).status != "stale"
