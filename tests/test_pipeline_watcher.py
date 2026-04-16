"""PipelineWatcher — snapshot collection, LLM ranking, dispatch to dispatcher."""
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select

from orchestrator.core.events import EventBus, EventType
from orchestrator.core.pipeline_watcher import (
    PipelineWatcher,
    WatcherSnapshot,
    _parse_items,
)
from orchestrator.core.rule_engine import ActionDispatcher
from orchestrator.db.models import ActionItem, Deal, EmailTrack, Meeting, Reminder


def _fake_client(items_json: str):
    block = MagicMock(type="text", text=items_json)
    resp = MagicMock(content=[block])
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=resp)
    return client


# ---- Snapshot collection ------------------------------------------


@pytest.mark.asyncio
async def test_collect_snapshots_includes_meddic_gaps(session_maker):
    long_ago = datetime.now(timezone.utc) - timedelta(days=20)
    async with session_maker() as s:
        s.add(Deal(id="d1", name="Bosch", stage="proposal", value_usd=500_000))
        s.add(Meeting(id="m1", deal_id="d1", date=long_ago))
        s.add(ActionItem(id="a1", deal_id="d1", description="follow up", status="open"))
        await s.commit()

    watcher = PipelineWatcher(session_maker, anthropic_client=None)
    snaps = await watcher.collect_snapshots()
    assert len(snaps) == 1
    snap = snaps[0]
    assert snap.id == "d1"
    assert snap.value_usd == 500_000
    assert "economic_buyer" in snap.meddic_gaps
    assert snap.open_action_count == 1
    assert snap.days_since_last_meeting >= 20


@pytest.mark.asyncio
async def test_collect_snapshots_skips_closed_deals(session_maker):
    async with session_maker() as s:
        s.add(Deal(id="d1", name="Won", stage="closed_won"))
        s.add(Deal(id="d2", name="Active", stage="proposal"))
        await s.commit()

    watcher = PipelineWatcher(session_maker, anthropic_client=None)
    snaps = await watcher.collect_snapshots()
    assert len(snaps) == 1
    assert snaps[0].name == "Active"


# ---- LLM ranking + dispatch ---------------------------------------


@pytest.mark.asyncio
async def test_run_returns_parsed_items(session_maker):
    async with session_maker() as s:
        s.add(Deal(id="d1", name="Bosch", stage="proposal", value_usd=500_000))
        await s.commit()

    payload = json.dumps([
        {"deal_id": "d1", "deal_name": "Bosch", "priority": 1,
         "headline": "Re-engage Bosch — silent for 20d", "why": "Top value at risk"},
    ])
    watcher = PipelineWatcher(session_maker, anthropic_client=_fake_client(payload))
    result = await watcher.run()
    assert len(result.items) == 1
    assert result.items[0].deal_id == "d1"
    assert result.items[0].priority == 1


@pytest.mark.asyncio
async def test_run_and_dispatch_creates_reminders(session_maker):
    async with session_maker() as s:
        s.add(Deal(id="d1", name="Bosch", stage="proposal", value_usd=500_000))
        await s.commit()

    payload = json.dumps([
        {"deal_id": "d1", "deal_name": "Bosch", "priority": 1,
         "headline": "Re-engage Bosch", "why": "value at risk"},
    ])
    bus = EventBus()
    disp = ActionDispatcher(session_maker, bus)
    watcher = PipelineWatcher(
        session_maker, anthropic_client=_fake_client(payload), dispatcher=disp,
    )

    result = await watcher.run_and_dispatch()
    assert len(result.items) == 1

    async with session_maker() as s:
        rems = (await s.execute(select(Reminder))).scalars().all()
    assert len(rems) == 1
    assert "#1" in rems[0].message
    assert "Re-engage Bosch" in rems[0].message
    assert rems[0].related_deal_id == "d1"


@pytest.mark.asyncio
async def test_attach_to_bus_fires_on_daily_sweep(session_maker):
    async with session_maker() as s:
        s.add(Deal(id="d1", name="X", stage="qualified"))
        await s.commit()

    payload = json.dumps([
        {"deal_id": "d1", "deal_name": "X", "priority": 2, "headline": "h", "why": "w"},
    ])
    bus = EventBus()
    disp = ActionDispatcher(session_maker, bus)
    watcher = PipelineWatcher(
        session_maker, anthropic_client=_fake_client(payload), dispatcher=disp,
    )
    watcher.attach_to_bus(bus)

    await bus.publish(EventType.DAILY_SWEEP, payload={})

    async with session_maker() as s:
        rems = (await s.execute(select(Reminder))).scalars().all()
    assert len(rems) == 1


# ---- LLM response parsing ----------------------------------------


def test_parse_items_handles_markdown_fence():
    raw = "```json\n[{\"deal_id\":\"d1\",\"deal_name\":\"X\",\"priority\":1,\"headline\":\"h\",\"why\":\"w\"}]\n```"
    items = _parse_items(raw)
    assert len(items) == 1


def test_parse_items_returns_empty_on_garbage():
    assert _parse_items("nothing here") == []


def test_parse_items_sorts_by_priority():
    raw = json.dumps([
        {"deal_id": "a", "deal_name": "A", "priority": 3, "headline": "h", "why": "w"},
        {"deal_id": "b", "deal_name": "B", "priority": 1, "headline": "h", "why": "w"},
        {"deal_id": "c", "deal_name": "C", "priority": 2, "headline": "h", "why": "w"},
    ])
    items = _parse_items(raw)
    assert [i.deal_id for i in items] == ["b", "c", "a"]
