"""Event bus + Event types for the rule engine.

In-process async pub/sub. Skills/services publish events; rules subscribe and
emit Actions. Async-only — handlers may await DB or LLM calls.

Event flow:
  skill.publish(EventType.DEAL_UPDATED, payload) →
    bus.publish() →
      registered handlers fire concurrently (asyncio.gather) →
        rule_engine evaluates rules subscribed to that type →
          emits Action(s) →
            dispatcher creates Reminder/Notification rows
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


# ---- Event types ----------------------------------------------------


class EventType:
    # CRM mutations
    DEAL_CREATED = "deal.created"
    DEAL_UPDATED = "deal.updated"
    DEAL_STAGE_CHANGED = "deal.stage_changed"
    CONTACT_CREATED = "contact.created"
    CONTACT_UPDATED = "contact.updated"
    MEETING_LOGGED = "meeting.logged"
    BID_CREATED = "bid.created"
    BID_DEADLINE_APPROACHING = "bid.deadline_approaching"

    # Email
    EMAIL_SENT = "email.sent"
    EMAIL_RECEIVED = "email.received"
    EMAIL_NO_REPLY = "email.no_reply"

    # Calendar
    CALENDAR_EVENT_ADDED = "calendar.event_added"
    CALENDAR_EVENT_UPDATED = "calendar.event_updated"

    # Time-based (fired by scheduler ticks)
    DAILY_SWEEP = "time.daily_sweep"
    HOURLY_SWEEP = "time.hourly_sweep"

    # Detected conditions (fired by rule engine itself or background services)
    DEAL_STALLED = "deal.stalled"
    MEDDIC_GAP_FOUND = "meddic.gap_found"

    # Cost-aware — fired by Agent when token usage crosses thresholds.
    # Rules can react by forcing compaction, downgrading model, or notifying user.
    TURN_COST_HIGH = "cost.turn_high"
    SESSION_COST_EXCEEDED = "cost.session_exceeded"


@dataclass
class Event:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = ""    # skill or service that emitted it


# ---- Bus -----------------------------------------------------------


Handler = Callable[[Event], Awaitable[None]]


class EventBus:
    """Async pub/sub. Handlers run concurrently per event."""

    def __init__(self):
        self._handlers: dict[str, list[Handler]] = {}
        self._wildcard: list[Handler] = []   # subscribers to ALL events (audit, tests)

    def subscribe(self, event_type: str, handler: Handler) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def subscribe_all(self, handler: Handler) -> None:
        self._wildcard.append(handler)

    def unsubscribe(self, event_type: str, handler: Handler) -> bool:
        lst = self._handlers.get(event_type)
        if lst and handler in lst:
            lst.remove(handler)
            return True
        return False

    async def publish(self, event_type: str, payload: dict | None = None, source: str = "") -> None:
        event = Event(type=event_type, payload=payload or {}, source=source)
        targets = list(self._handlers.get(event_type, [])) + list(self._wildcard)
        if not targets:
            return
        await asyncio.gather(
            *(_safe(handler, event) for handler in targets),
            return_exceptions=False,
        )


async def _safe(handler: Handler, event: Event) -> None:
    """Wrap a handler so one bad subscriber can't crash the others."""
    try:
        await handler(event)
    except Exception as e:
        logger.exception("EventBus handler crashed on %s: %s", event.type, e)


__all__ = ["Event", "EventBus", "EventType", "Handler"]
