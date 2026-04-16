"""Event bus + rule engine + action dispatcher.

Each built-in rule is exercised end-to-end: seed DB → publish event → assert
that the right Reminder rows materialize."""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from orchestrator.core.events import Event, EventBus, EventType
from orchestrator.core.rule_engine import (
    ActionDispatcher,
    CreateReminder,
    PublishEvent,
    Rule,
    RuleEngine,
    rule_bid_deadline_t7d,
    rule_email_received_triage,
    rule_no_reply_email,
    rule_stalled_deal,
)
from orchestrator.db.models import Bid, Deal, EmailTrack, Meeting, Reminder


# ---- EventBus basics ----------------------------------------------


@pytest.mark.asyncio
async def test_bus_dispatches_to_subscriber():
    bus = EventBus()
    received = []

    async def handler(event):
        received.append(event)

    bus.subscribe("foo", handler)
    await bus.publish("foo", {"x": 1}, source="test")
    assert len(received) == 1
    assert received[0].payload == {"x": 1}
    assert received[0].source == "test"


@pytest.mark.asyncio
async def test_bus_handler_exception_does_not_kill_others():
    bus = EventBus()

    async def bad(event): raise RuntimeError("boom")
    async def good(event): good.called = True

    bus.subscribe("e", bad)
    bus.subscribe("e", good)
    await bus.publish("e")
    assert getattr(good, "called", False) is True


@pytest.mark.asyncio
async def test_bus_wildcard_receives_all():
    bus = EventBus()
    log = []

    async def all_handler(event):
        log.append(event.type)

    bus.subscribe_all(all_handler)
    await bus.publish("a")
    await bus.publish("b")
    assert log == ["a", "b"]


# ---- Built-in rules ------------------------------------------------


@pytest.mark.asyncio
async def test_stalled_deal_creates_reminder(session_maker):
    long_ago = datetime.now(timezone.utc) - timedelta(days=30)
    async with session_maker() as s:
        s.add(Deal(id="d1", name="Bosch Forge", stage="proposal"))
        s.add(Meeting(id="m1", deal_id="d1", date=long_ago))
        await s.commit()

    actions = await rule_stalled_deal(
        Event(EventType.DAILY_SWEEP, payload={"stalled_deal_days": 14, "chat_id": "c"}),
        session_maker,
    )
    assert len(actions) == 1
    assert isinstance(actions[0], CreateReminder)
    assert "Bosch Forge" in actions[0].message
    assert actions[0].related_deal_id == "d1"


@pytest.mark.asyncio
async def test_stalled_deal_skips_recent_meeting(session_maker):
    recent = datetime.now(timezone.utc) - timedelta(days=2)
    async with session_maker() as s:
        s.add(Deal(id="d1", name="Active Deal", stage="proposal"))
        s.add(Meeting(id="m1", deal_id="d1", date=recent))
        await s.commit()

    actions = await rule_stalled_deal(
        Event(EventType.DAILY_SWEEP, payload={"stalled_deal_days": 14}),
        session_maker,
    )
    assert actions == []


@pytest.mark.asyncio
async def test_no_reply_email_creates_reminder(session_maker):
    week_ago = datetime.now(timezone.utc) - timedelta(days=7)
    async with session_maker() as s:
        s.add(EmailTrack(
            id="e1", to_address="anja@bosch.com", subject="Pricing follow-up",
            sent_at=week_ago, status="awaiting_reply", nudge_after_days=5,
        ))
        await s.commit()

    actions = await rule_no_reply_email(
        Event(EventType.DAILY_SWEEP),
        session_maker,
    )
    assert len(actions) == 1
    assert "Pricing follow-up" in actions[0].message
    assert "anja@bosch.com" in actions[0].message


@pytest.mark.asyncio
async def test_bid_deadline_schedules_three_reminders(session_maker):
    deadline = datetime.now(timezone.utc) + timedelta(days=10)
    async with session_maker() as s:
        s.add(Bid(id="b1", name="Honeywell RFP-23", submission_deadline=deadline))
        await s.commit()

    actions = await rule_bid_deadline_t7d(
        Event(EventType.BID_CREATED, payload={"bid_id": "b1", "chat_id": "c"}),
        session_maker,
    )
    # 10 days out: T-7d (in 3d), T-3d (in 7d), T-1d (in 9d) — all three should fire
    assert len(actions) == 3
    assert all(isinstance(a, CreateReminder) for a in actions)
    assert all(a.related_bid_id == "b1" for a in actions)


@pytest.mark.asyncio
async def test_bid_deadline_skips_past_offsets(session_maker):
    """Deadline in 2d → only T-1d fires; T-7d and T-3d are already past."""
    deadline = datetime.now(timezone.utc) + timedelta(days=2)
    async with session_maker() as s:
        s.add(Bid(id="b1", name="Last-minute RFP", submission_deadline=deadline))
        await s.commit()

    actions = await rule_bid_deadline_t7d(
        Event(EventType.BID_CREATED, payload={"bid_id": "b1"}),
        session_maker,
    )
    assert len(actions) == 1


@pytest.mark.asyncio
async def test_email_received_chains_triage_event(session_maker):
    actions = await rule_email_received_triage(
        Event(EventType.EMAIL_RECEIVED, payload={"history_id": "abc"}),
        session_maker,
    )
    assert len(actions) == 1
    assert isinstance(actions[0], PublishEvent)
    assert actions[0].type == "email.triage_requested"


# ---- ActionDispatcher --------------------------------------------


@pytest.mark.asyncio
async def test_dispatcher_writes_reminder(session_maker):
    bus = EventBus()
    disp = ActionDispatcher(session_maker, bus)
    trig = datetime.now(timezone.utc) + timedelta(hours=1)

    await disp.dispatch(CreateReminder(
        message="test reminder", trigger_at=trig, related_deal_id="d1",
    ))

    async with session_maker() as s:
        rows = (await s.execute(select(Reminder))).scalars().all()
    assert len(rows) == 1
    assert rows[0].message == "test reminder"


@pytest.mark.asyncio
async def test_dispatcher_dedupes_within_24h(session_maker):
    bus = EventBus()
    disp = ActionDispatcher(session_maker, bus)
    trig = datetime.now(timezone.utc) + timedelta(hours=1)

    await disp.dispatch(CreateReminder(message="dup test", trigger_at=trig))
    await disp.dispatch(CreateReminder(message="dup test", trigger_at=trig))

    async with session_maker() as s:
        rows = (await s.execute(select(Reminder))).scalars().all()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_dispatcher_publishes_chained_events(session_maker):
    bus = EventBus()
    disp = ActionDispatcher(session_maker, bus)
    received = []

    async def h(e): received.append(e)
    bus.subscribe("chained", h)

    await disp.dispatch(PublishEvent(type="chained", payload={"x": 9}))
    assert len(received) == 1
    assert received[0].payload == {"x": 9}


# ---- End-to-end via RuleEngine ----------------------------------


@pytest.mark.asyncio
async def test_rule_engine_full_pipeline(session_maker):
    """publish DAILY_SWEEP → stalled_deal rule fires → reminder written."""
    long_ago = datetime.now(timezone.utc) - timedelta(days=30)
    async with session_maker() as s:
        s.add(Deal(id="d1", name="StalledCo", stage="qualified"))
        s.add(Meeting(id="m1", deal_id="d1", date=long_ago))
        await s.commit()

    bus = EventBus()
    disp = ActionDispatcher(session_maker, bus)
    engine = RuleEngine(bus, session_maker, disp)
    engine.register_builtins()

    await bus.publish(EventType.DAILY_SWEEP, payload={"stalled_deal_days": 14})

    async with session_maker() as s:
        reminders = (await s.execute(select(Reminder))).scalars().all()
    assert len(reminders) == 1
    assert "StalledCo" in reminders[0].message
