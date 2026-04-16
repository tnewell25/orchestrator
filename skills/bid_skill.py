"""BidSkill — RFP/bid tracking with automatic deadline reminders.

When a Bid is created with a submission_deadline, this skill auto-schedules
three reminders: T-7 days, T-3 days, T-1 day. Q&A deadlines get T-2 days and
T-1 day. Reminders are persisted so they survive restart.
"""
from datetime import datetime, timedelta, timezone

import dateparser
from sqlalchemy import select

from ..core.skill_base import Skill, tool
from ..db.models import Bid, Reminder


_STAGES = {"evaluating", "in_progress", "submitted", "won", "lost", "withdrawn"}


def _parse_dt(s: str, user_timezone: str = "UTC") -> datetime | None:
    if not s:
        return None
    dt = dateparser.parse(
        s,
        settings={
            "PREFER_DATES_FROM": "future",
            "RETURN_AS_TIMEZONE_AWARE": True,
            "TIMEZONE": user_timezone,
            "TO_TIMEZONE": "UTC",
        },
    )
    if dt and dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class BidSkill(Skill):
    name = "bid"
    description = "Manage RFP/bid deadlines with automatic countdown reminders."

    def __init__(self, session_maker, default_chat_id: str = "", user_timezone: str = "UTC"):
        super().__init__()
        self.session_maker = session_maker
        self.default_chat_id = default_chat_id
        self.user_timezone = user_timezone

    async def _schedule_deadline_reminders(
        self,
        session,
        bid: Bid,
        target_dt: datetime,
        label: str,
        offsets_days: list[int],
    ):
        """Create reminders at each offset before target_dt. Skips offsets in the past."""
        now = datetime.now(timezone.utc)
        for days in offsets_days:
            trigger = target_dt - timedelta(days=days)
            if trigger <= now:
                continue
            msg = (
                f"{label} for '{bid.name}' in {days} day{'s' if days != 1 else ''} "
                f"({target_dt.strftime('%a %b %d %H:%M UTC')})"
            )
            session.add(
                Reminder(
                    trigger_at=trigger,
                    message=msg,
                    target_chat_id=self.default_chat_id or None,
                    related_bid_id=bid.id,
                    related_deal_id=bid.deal_id,
                    kind="bid_deadline",
                )
            )

    @tool(
        "Create a bid/RFP. submission_deadline and qa_deadline accept natural language "
        "('2026-05-15 17:00', 'next Friday 5pm', 'in 2 weeks'). On creation, reminders "
        "are auto-scheduled at T-7d/T-3d/T-1d for submission, T-2d/T-1d for Q&A."
    )
    async def create(
        self,
        name: str,
        submission_deadline: str,
        company_id: str = "",
        deal_id: str = "",
        value_usd: float = 0.0,
        qa_deadline: str = "",
        rfp_url: str = "",
        deliverables: str = "",
        notes: str = "",
    ) -> dict:
        sub_dt = _parse_dt(submission_deadline, self.user_timezone)
        if not sub_dt:
            return {"error": f"Could not parse submission_deadline '{submission_deadline}'"}
        qa_dt = _parse_dt(qa_deadline, self.user_timezone) if qa_deadline else None

        async with self.session_maker() as s:
            b = Bid(
                name=name,
                company_id=company_id or None,
                deal_id=deal_id or None,
                value_usd=value_usd,
                submission_deadline=sub_dt,
                qa_deadline=qa_dt,
                rfp_url=rfp_url,
                deliverables=deliverables,
                notes=notes,
            )
            s.add(b)
            await s.flush()

            await self._schedule_deadline_reminders(
                s, b, sub_dt, "SUBMISSION DEADLINE", [7, 3, 1]
            )
            if qa_dt:
                await self._schedule_deadline_reminders(
                    s, b, qa_dt, "Q&A window closes", [2, 1]
                )

            await s.commit()
            await s.refresh(b)
            return {
                "id": b.id,
                "name": b.name,
                "submission_deadline": b.submission_deadline.isoformat(),
                "days_until_deadline": (b.submission_deadline - datetime.now(timezone.utc)).days,
                "reminders_scheduled": "T-7d/T-3d/T-1d for submission"
                + (", T-2d/T-1d for Q&A" if qa_dt else ""),
            }

    @tool("List open bids (not submitted/won/lost/withdrawn), sorted by nearest deadline.")
    async def list_open(self) -> list[dict]:
        async with self.session_maker() as s:
            result = await s.execute(
                select(Bid)
                .where(Bid.stage.in_({"evaluating", "in_progress"}))
                .order_by(Bid.submission_deadline.asc().nullslast())
            )
            now = datetime.now(timezone.utc)
            return [
                {
                    "id": r.id,
                    "name": r.name,
                    "stage": r.stage,
                    "value_usd": r.value_usd,
                    "submission_deadline": r.submission_deadline.isoformat() if r.submission_deadline else None,
                    "days_remaining": (r.submission_deadline - now).days if r.submission_deadline else None,
                    "deal_id": r.deal_id,
                    "company_id": r.company_id,
                }
                for r in result.scalars().all()
            ]

    @tool("Update a bid's fields. For stage, valid values: evaluating, in_progress, submitted, won, lost, withdrawn.")
    async def update(
        self,
        bid_id: str,
        stage: str = "",
        value_usd: float = -1.0,
        submission_deadline: str = "",
        qa_deadline: str = "",
        deliverables: str = "",
        notes: str = "",
    ) -> dict:
        async with self.session_maker() as s:
            b = await s.get(Bid, bid_id)
            if not b:
                return {"error": f"Bid {bid_id} not found"}
            if stage:
                if stage not in _STAGES:
                    return {"error": f"Invalid stage '{stage}'. Valid: {sorted(_STAGES)}"}
                b.stage = stage
            if value_usd >= 0:
                b.value_usd = value_usd
            if submission_deadline:
                dt = _parse_dt(submission_deadline, self.user_timezone)
                if dt:
                    b.submission_deadline = dt
            if qa_deadline:
                dt = _parse_dt(qa_deadline, self.user_timezone)
                if dt:
                    b.qa_deadline = dt
            if deliverables:
                b.deliverables = deliverables
            if notes:
                b.notes = (b.notes + "\n" if b.notes else "") + notes
            await s.commit()
            return {"id": b.id, "name": b.name, "stage": b.stage, "updated": True}

    @tool(
        "Mark a bid as submitted. Also cancels any remaining pre-submission reminders. "
        "Use when the bid has been sent off."
    )
    async def mark_submitted(self, bid_id: str) -> dict:
        async with self.session_maker() as s:
            b = await s.get(Bid, bid_id)
            if not b:
                return {"error": f"Bid {bid_id} not found"}
            b.stage = "submitted"

            # Cancel pending deadline reminders for this bid
            result = await s.execute(
                select(Reminder).where(
                    Reminder.related_bid_id == bid_id,
                    Reminder.status == "pending",
                    Reminder.kind == "bid_deadline",
                )
            )
            cancelled = 0
            for r in result.scalars().all():
                r.status = "cancelled"
                cancelled += 1
            await s.commit()
            return {"id": b.id, "stage": "submitted", "reminders_cancelled": cancelled}
