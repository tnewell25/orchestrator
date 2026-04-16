"""Rule engine — declarative reactions to events.

Built-in rules cover stalled deals, no-reply emails, bid deadlines, MEDDIC
gaps. Custom rules can be added at runtime via `engine.register(rule)`.

A rule = predicate(event) → list[Action]. The engine subscribes one handler
per registered rule per event type. Actions are dispatched by the
ActionDispatcher which writes Reminders / publishes Notifications.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from ..db.models import Bid, Deal, EmailTrack, Meeting, Reminder
from .events import Event, EventBus, EventType

logger = logging.getLogger(__name__)


def _utc(dt: datetime | None) -> datetime | None:
    """Normalize naive (sqlite) and aware (postgres) datetimes to UTC-aware
    so comparisons don't crash."""
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ---- Action types ---------------------------------------------------


@dataclass
class CreateReminder:
    """Output action: a Reminder row should be inserted."""
    message: str
    trigger_at: datetime
    target_chat_id: str = ""
    related_deal_id: str | None = None
    related_bid_id: str | None = None
    related_contact_id: str | None = None
    related_meeting_id: str | None = None
    kind: str = "custom"


@dataclass
class PublishEvent:
    """Output action: chain another event onto the bus."""
    type: str
    payload: dict = field(default_factory=dict)


Action = CreateReminder | PublishEvent


# ---- Rule definitions ------------------------------------------------


@dataclass
class Rule:
    name: str
    event_types: tuple[str, ...]
    handler: Callable[[Event, async_sessionmaker], Awaitable[list[Action]]]
    description: str = ""


# ---- Built-in rules -------------------------------------------------
# Each rule is an async function taking (event, session_maker) and returning
# a list of Actions. Handlers must NOT mutate state directly — that's the
# dispatcher's job, so rules stay testable in isolation.


async def rule_stalled_deal(event: Event, sm: async_sessionmaker) -> list[Action]:
    """When a daily sweep fires, surface deals with no meeting in N days."""
    if event.type != EventType.DAILY_SWEEP:
        return []
    threshold_days = int(event.payload.get("stalled_deal_days", 14))
    cutoff = datetime.now(timezone.utc) - timedelta(days=threshold_days)
    actions: list[Action] = []
    chat_id = event.payload.get("chat_id", "")
    async with sm() as s:
        deals = (
            await s.execute(
                select(Deal).where(Deal.stage.in_(["qualified", "proposal", "negotiation"]))
            )
        ).scalars().all()
        for d in deals:
            last_meeting = (
                await s.execute(
                    select(Meeting)
                    .where(Meeting.deal_id == d.id)
                    .order_by(Meeting.date.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            last_dt = _utc(last_meeting.date) if last_meeting else None
            if last_dt is None or last_dt < cutoff:
                actions.append(CreateReminder(
                    message=f"Stalled deal: {d.name} — no meeting in {threshold_days}+ days",
                    trigger_at=datetime.now(timezone.utc) + timedelta(minutes=1),
                    target_chat_id=chat_id,
                    related_deal_id=d.id,
                    kind="commitment",
                ))
    return actions


async def rule_no_reply_email(event: Event, sm: async_sessionmaker) -> list[Action]:
    """Daily sweep: surface sent emails awaiting reply past nudge threshold."""
    if event.type != EventType.DAILY_SWEEP:
        return []
    chat_id = event.payload.get("chat_id", "")
    actions: list[Action] = []
    now = datetime.now(timezone.utc)
    async with sm() as s:
        rows = (
            await s.execute(
                select(EmailTrack).where(EmailTrack.status == "awaiting_reply")
            )
        ).scalars().all()
        for et in rows:
            sent = _utc(et.sent_at)
            if sent is None:
                continue
            age_days = (now - sent).days
            if age_days >= (et.nudge_after_days or 5):
                actions.append(CreateReminder(
                    message=f"No reply in {age_days}d: {et.subject or '(no subject)'} → {et.to_address}",
                    trigger_at=now + timedelta(minutes=1),
                    target_chat_id=chat_id,
                    related_deal_id=et.related_deal_id,
                    related_contact_id=et.related_contact_id,
                    kind="custom",
                ))
    return actions


async def rule_bid_deadline_t7d(event: Event, sm: async_sessionmaker) -> list[Action]:
    """When a bid is created, schedule T-7d / T-3d / T-1d reminders.

    Listens to bid.created so it's purely event-driven (no daily sweep needed)."""
    if event.type != EventType.BID_CREATED:
        return []
    bid_id = event.payload.get("bid_id")
    chat_id = event.payload.get("chat_id", "")
    if not bid_id:
        return []
    async with sm() as s:
        bid = await s.get(Bid, bid_id)
        if not bid or bid.submission_deadline is None:
            return []

    actions: list[Action] = []
    deadline = _utc(bid.submission_deadline)
    now = datetime.now(timezone.utc)
    for offset_days, label in [(7, "T-7d"), (3, "T-3d"), (1, "T-1d")]:
        trigger = deadline - timedelta(days=offset_days)
        if trigger > now:
            actions.append(CreateReminder(
                message=f"{label} bid deadline: {bid.name}",
                trigger_at=trigger,
                target_chat_id=chat_id,
                related_bid_id=bid_id,
                kind="bid_deadline",
            ))
    return actions


async def rule_email_received_triage(event: Event, sm: async_sessionmaker) -> list[Action]:
    """When new mail arrives via webhook, chain a triage event so the inbox
    triager can score + draft reply (handled elsewhere)."""
    if event.type != EventType.EMAIL_RECEIVED:
        return []
    return [PublishEvent(type="email.triage_requested", payload=event.payload)]


# ---- Engine + dispatcher --------------------------------------------


class RuleEngine:
    """Holds registered rules and bridges EventBus → Actions → dispatcher."""

    def __init__(
        self,
        bus: EventBus,
        session_maker: async_sessionmaker,
        dispatcher: "ActionDispatcher",
    ):
        self.bus = bus
        self.sm = session_maker
        self.dispatcher = dispatcher
        self._rules: list[Rule] = []

    def register(self, rule: Rule) -> None:
        self._rules.append(rule)
        for et in rule.event_types:
            self.bus.subscribe(et, self._make_handler(rule))

    def register_builtins(self) -> None:
        """One-call shortcut to register the standard rule library."""
        self.register(Rule(
            name="stalled_deal",
            event_types=(EventType.DAILY_SWEEP,),
            handler=rule_stalled_deal,
            description="Surface deals with no meeting in N days",
        ))
        self.register(Rule(
            name="no_reply_email",
            event_types=(EventType.DAILY_SWEEP,),
            handler=rule_no_reply_email,
            description="Nudge on emails awaiting reply past threshold",
        ))
        self.register(Rule(
            name="bid_deadline_schedule",
            event_types=(EventType.BID_CREATED,),
            handler=rule_bid_deadline_t7d,
            description="Schedule T-7d/T-3d/T-1d reminders on bid creation",
        ))
        self.register(Rule(
            name="email_received_triage",
            event_types=(EventType.EMAIL_RECEIVED,),
            handler=rule_email_received_triage,
            description="Chain a triage event when new mail arrives",
        ))

    def _make_handler(self, rule: Rule):
        async def handler(event: Event):
            try:
                actions = await rule.handler(event, self.sm)
            except Exception as e:
                logger.exception("Rule %s crashed on %s: %s", rule.name, event.type, e)
                return
            if not actions:
                return
            for action in actions:
                await self.dispatcher.dispatch(action)
        return handler


class ActionDispatcher:
    """Executes Actions emitted by rules. Reminder writes hit the DB; chained
    events go back to the bus."""

    def __init__(self, session_maker: async_sessionmaker, bus: EventBus):
        self.sm = session_maker
        self.bus = bus

    async def dispatch(self, action: Action) -> None:
        if isinstance(action, CreateReminder):
            await self._create_reminder(action)
        elif isinstance(action, PublishEvent):
            await self.bus.publish(action.type, action.payload)
        else:
            logger.warning("Unknown action type: %r", type(action))

    async def _create_reminder(self, a: CreateReminder) -> Reminder:
        async with self.sm() as session:
            # De-duplication: don't insert near-identical reminder fired in last 24h
            cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
            existing = (
                await session.execute(
                    select(Reminder).where(
                        Reminder.message == a.message,
                        Reminder.created_at >= cutoff,
                    )
                )
            ).scalar_one_or_none()
            if existing:
                return existing
            r = Reminder(
                message=a.message,
                trigger_at=a.trigger_at,
                target_chat_id=a.target_chat_id,
                related_deal_id=a.related_deal_id,
                related_bid_id=a.related_bid_id,
                related_contact_id=a.related_contact_id,
                related_meeting_id=a.related_meeting_id,
                kind=a.kind,
            )
            session.add(r)
            await session.commit()
            await session.refresh(r)
            return r


__all__ = [
    "Action",
    "ActionDispatcher",
    "CreateReminder",
    "PublishEvent",
    "Rule",
    "RuleEngine",
    "rule_stalled_deal",
    "rule_no_reply_email",
    "rule_bid_deadline_t7d",
    "rule_email_received_triage",
]
