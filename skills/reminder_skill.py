"""ReminderSkill — proactive time-based prompts.

The agent calls these tools whenever the user says something like:
  - "remind me in 2 hours to call Markus"
  - "ping me 30 min before the Bosch meeting"
  - "follow up with Anja next Tuesday"

Natural-language times are parsed via dateparser (handles "in 2 hours",
"tomorrow 3pm", "next Tuesday at 9am", "2026-04-20 14:00", etc).
"""
from datetime import datetime, timedelta, timezone

import dateparser
from sqlalchemy import select

from ..core.skill_base import Skill, tool
from ..db.models import Meeting, Reminder


def _parse_when(when: str) -> datetime | None:
    """Parse natural-language time string to UTC datetime."""
    if not when:
        return None
    dt = dateparser.parse(
        when,
        settings={
            "PREFER_DATES_FROM": "future",
            "RETURN_AS_TIMEZONE_AWARE": True,
            "TIMEZONE": "UTC",
            "TO_TIMEZONE": "UTC",
        },
    )
    if dt and dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class ReminderSkill(Skill):
    name = "reminder"
    description = "Set, list, cancel, and snooze proactive reminders."

    def __init__(self, session_maker, default_chat_id: str = ""):
        super().__init__()
        self.session_maker = session_maker
        self.default_chat_id = default_chat_id

    @tool(
        "Set a reminder. `when` accepts natural language: 'in 2 hours', 'tomorrow 3pm', "
        "'next Tuesday 9am', or ISO 'YYYY-MM-DD HH:MM'. `message` is what to remind "
        "about. Optionally link to a deal/contact/meeting/bid via their ids for context. "
        "kind: custom | pre_meeting | bid_deadline | commitment."
    )
    async def set(
        self,
        when: str,
        message: str,
        deal_id: str = "",
        contact_id: str = "",
        meeting_id: str = "",
        bid_id: str = "",
        kind: str = "custom",
    ) -> dict:
        trigger_at = _parse_when(when)
        if not trigger_at:
            return {"error": f"Could not parse time '{when}'. Try 'in 2 hours' or 'tomorrow 3pm'."}
        if trigger_at <= datetime.now(timezone.utc):
            return {"error": f"Parsed time {trigger_at} is in the past."}

        async with self.session_maker() as s:
            r = Reminder(
                trigger_at=trigger_at,
                message=message,
                target_chat_id=self.default_chat_id or None,
                related_deal_id=deal_id or None,
                related_contact_id=contact_id or None,
                related_meeting_id=meeting_id or None,
                related_bid_id=bid_id or None,
                kind=kind,
            )
            s.add(r)
            await s.commit()
            await s.refresh(r)
            return {
                "id": r.id,
                "trigger_at": r.trigger_at.isoformat(),
                "message": r.message,
                "kind": r.kind,
            }

    @tool(
        "Schedule a pre-meeting brief reminder. `meeting_id` is required. "
        "`minutes_before` is how early to fire (default 30). When this reminder "
        "fires, the agent auto-generates a full context brief (last meetings, "
        "open actions, personal notes on attendees)."
    )
    async def set_pre_meeting(
        self,
        meeting_id: str,
        minutes_before: int = 30,
        custom_message: str = "",
    ) -> dict:
        async with self.session_maker() as s:
            m = await s.get(Meeting, meeting_id)
            if not m:
                return {"error": f"Meeting {meeting_id} not found"}
            if not m.date:
                return {"error": "Meeting has no date set"}

            trigger = m.date - timedelta(minutes=minutes_before)
            if trigger <= datetime.now(timezone.utc):
                return {"error": f"Pre-meeting trigger {trigger} is in the past"}

            msg = custom_message or f"Meeting in {minutes_before} min: {m.summary[:100] or 'untitled'}"
            r = Reminder(
                trigger_at=trigger,
                message=msg,
                target_chat_id=self.default_chat_id or None,
                related_meeting_id=meeting_id,
                related_deal_id=m.deal_id,
                kind="pre_meeting",
            )
            s.add(r)
            await s.commit()
            await s.refresh(r)
            return {
                "id": r.id,
                "trigger_at": r.trigger_at.isoformat(),
                "meeting_id": meeting_id,
                "kind": "pre_meeting",
            }

    @tool("List pending reminders, optionally filtered by deal_id or the next N hours.")
    async def list_pending(self, deal_id: str = "", within_hours: int = 0) -> list[dict]:
        async with self.session_maker() as s:
            q = select(Reminder).where(Reminder.status == "pending")
            if deal_id:
                q = q.where(Reminder.related_deal_id == deal_id)
            if within_hours > 0:
                cutoff = datetime.now(timezone.utc) + timedelta(hours=within_hours)
                q = q.where(Reminder.trigger_at <= cutoff)
            q = q.order_by(Reminder.trigger_at.asc())
            result = await s.execute(q)
            return [
                {
                    "id": r.id,
                    "trigger_at": r.trigger_at.isoformat(),
                    "message": r.message,
                    "kind": r.kind,
                    "deal_id": r.related_deal_id,
                }
                for r in result.scalars().all()
            ]

    @tool("Cancel a pending reminder.")
    async def cancel(self, reminder_id: str) -> dict:
        async with self.session_maker() as s:
            r = await s.get(Reminder, reminder_id)
            if not r:
                return {"error": f"Reminder {reminder_id} not found"}
            r.status = "cancelled"
            await s.commit()
            return {"id": r.id, "status": "cancelled"}

    @tool("Snooze a reminder — reschedule to a new `when` (natural-language time).")
    async def snooze(self, reminder_id: str, when: str) -> dict:
        new_trigger = _parse_when(when)
        if not new_trigger:
            return {"error": f"Could not parse time '{when}'"}
        async with self.session_maker() as s:
            r = await s.get(Reminder, reminder_id)
            if not r:
                return {"error": f"Reminder {reminder_id} not found"}
            r.trigger_at = new_trigger
            r.status = "pending"
            await s.commit()
            return {"id": r.id, "new_trigger_at": new_trigger.isoformat()}
