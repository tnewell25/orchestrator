"""PipelineWatcher — first true sub-agent.

Runs nightly (or on-demand) over the active pipeline and produces a ranked
notification queue: "5 things to look at." Backed by Haiku for cost; the
output flows through the rule engine's CreateReminder action so it lands
in the same Reminder table the user already trusts.

Why a sub-agent and not a static rule? Because ranking requires synthesis —
weighing MEDDIC gaps against age, value, and stage. That's classic LLM work,
not a hardcoded predicate.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from ..db.models import ActionItem, Deal, EmailTrack, Meeting
from .events import Event, EventBus, EventType
from .rule_engine import ActionDispatcher, CreateReminder

logger = logging.getLogger(__name__)


WATCHER_SYSTEM = """You are the Pipeline Watcher — a sub-agent that nightly
audits a senior sales engineer's deal pipeline.

You receive a JSON payload describing every active deal: stage, value, last
meeting age, MEDDIC gaps, open commitments, recent emails. You rank the
top 5 items the user should look at TOMORROW MORNING.

Output ONLY a JSON array of items. Each item:
  {
    "deal_id": "<uuid>",
    "deal_name": "<name>",
    "priority": 1-5 (1 = highest),
    "headline": "<one-line summary, ≤80 chars, action-oriented>",
    "why": "<one-sentence rationale>"
  }

Ranking heuristics:
- Imminent value at risk (large $ × stale + missing critical MEDDIC) ranks higher
- Stage 'negotiation' with no recent contact > stage 'qualified'
- Don't include deals with everything healthy
- 5 items max — fewer is fine if pipeline is in good shape"""


@dataclass
class WatcherSnapshot:
    """Per-deal context fed to the LLM."""
    id: str
    name: str
    stage: str
    value_usd: float
    days_since_last_meeting: int | None
    meddic_gaps: list[str]
    open_action_count: int
    recent_email_count: int


@dataclass
class WatcherItem:
    deal_id: str
    deal_name: str
    priority: int
    headline: str
    why: str


@dataclass
class WatcherResult:
    snapshots: list[WatcherSnapshot] = field(default_factory=list)
    items: list[WatcherItem] = field(default_factory=list)


class PipelineWatcher:
    def __init__(
        self,
        session_maker: async_sessionmaker,
        anthropic_client,
        dispatcher: ActionDispatcher | None = None,
        fast_model: str = "claude-haiku-4-5-20251001",
        default_chat_id: str = "",
    ):
        self.sm = session_maker
        self.client = anthropic_client
        self.dispatcher = dispatcher
        self.fast_model = fast_model
        self.default_chat_id = default_chat_id

    def attach_to_bus(self, bus: EventBus) -> None:
        """Subscribe to DAILY_SWEEP. The rule engine handles the simple polls;
        the watcher synthesizes the ranked digest."""
        async def handler(event: Event):
            try:
                await self.run_and_dispatch()
            except Exception as e:
                logger.exception("PipelineWatcher tick failed: %s", e)
        bus.subscribe(EventType.DAILY_SWEEP, handler)

    async def collect_snapshots(self) -> list[WatcherSnapshot]:
        now = datetime.now(timezone.utc)
        snapshots: list[WatcherSnapshot] = []

        async with self.sm() as session:
            deals = (
                await session.execute(
                    select(Deal).where(Deal.stage.in_([
                        "qualified", "proposal", "negotiation",
                    ]))
                )
            ).scalars().all()

            for d in deals:
                last_meeting = (
                    await session.execute(
                        select(Meeting)
                        .where(Meeting.deal_id == d.id)
                        .order_by(Meeting.date.desc())
                        .limit(1)
                    )
                ).scalar_one_or_none()
                days_stale = None
                if last_meeting and last_meeting.date:
                    mdt = last_meeting.date if last_meeting.date.tzinfo else last_meeting.date.replace(tzinfo=timezone.utc)
                    days_stale = max(0, (now - mdt).days)

                gaps: list[str] = []
                if not d.economic_buyer_id:
                    gaps.append("economic_buyer")
                if not d.champion_id:
                    gaps.append("champion")
                if not d.metrics:
                    gaps.append("metrics")
                if not d.decision_criteria:
                    gaps.append("decision_criteria")

                open_actions = (
                    await session.execute(
                        select(ActionItem).where(
                            ActionItem.deal_id == d.id,
                            ActionItem.status == "open",
                        )
                    )
                ).scalars().all()

                recent_email_count = 0
                rows = (
                    await session.execute(
                        select(EmailTrack).where(EmailTrack.related_deal_id == d.id)
                    )
                ).scalars().all()
                week_ago = now - timedelta(days=7)
                for r in rows:
                    sent = r.sent_at if r.sent_at and r.sent_at.tzinfo else (
                        r.sent_at.replace(tzinfo=timezone.utc) if r.sent_at else None
                    )
                    if sent and sent >= week_ago:
                        recent_email_count += 1

                snapshots.append(WatcherSnapshot(
                    id=d.id, name=d.name, stage=d.stage,
                    value_usd=d.value_usd or 0.0,
                    days_since_last_meeting=days_stale,
                    meddic_gaps=gaps,
                    open_action_count=len(open_actions),
                    recent_email_count=recent_email_count,
                ))
        return snapshots

    async def run(self) -> WatcherResult:
        snapshots = await self.collect_snapshots()
        result = WatcherResult(snapshots=snapshots)
        if not snapshots:
            return result

        if not self.client:
            return result

        payload = json.dumps([_snapshot_to_dict(s) for s in snapshots])
        try:
            resp = await self.client.messages.create(
                model=self.fast_model,
                max_tokens=800,
                system=WATCHER_SYSTEM,
                messages=[{"role": "user", "content": payload}],
            )
            raw = "".join(b.text for b in resp.content if b.type == "text").strip()
        except Exception as e:
            logger.warning("PipelineWatcher LLM call failed: %s", e)
            return result

        result.items = _parse_items(raw)
        return result

    async def run_and_dispatch(self) -> WatcherResult:
        result = await self.run()
        if not result.items or self.dispatcher is None:
            return result

        now = datetime.now(timezone.utc) + timedelta(minutes=1)
        for item in result.items[:5]:
            await self.dispatcher.dispatch(CreateReminder(
                message=f"#{item.priority} {item.headline} — {item.why}",
                trigger_at=now,
                target_chat_id=self.default_chat_id,
                related_deal_id=item.deal_id,
                kind="commitment",
            ))
        return result


# ---- helpers ---------------------------------------------------------


def _snapshot_to_dict(s: WatcherSnapshot) -> dict:
    return {
        "deal_id": s.id, "deal_name": s.name, "stage": s.stage,
        "value_usd": s.value_usd,
        "days_since_last_meeting": s.days_since_last_meeting,
        "meddic_gaps": s.meddic_gaps,
        "open_action_count": s.open_action_count,
        "recent_email_count": s.recent_email_count,
    }


def _parse_items(raw: str) -> list[WatcherItem]:
    if not raw:
        return []
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if "\n" in cleaned:
            cleaned = cleaned.split("\n", 1)[1]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    if not isinstance(data, list):
        return []

    items: list[WatcherItem] = []
    for d in data:
        if not isinstance(d, dict):
            continue
        try:
            items.append(WatcherItem(
                deal_id=str(d.get("deal_id") or ""),
                deal_name=str(d.get("deal_name") or ""),
                priority=int(d.get("priority") or 5),
                headline=str(d.get("headline") or "")[:120],
                why=str(d.get("why") or "")[:200],
            ))
        except (ValueError, TypeError):
            continue
    items.sort(key=lambda i: i.priority)
    return items


__all__ = ["PipelineWatcher", "WatcherItem", "WatcherResult", "WatcherSnapshot"]
