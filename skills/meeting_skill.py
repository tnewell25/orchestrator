"""MeetingSkill — capture meeting summaries, decisions, action items.

Typical flow: user sends a voice note → telegram_bot transcribes via Whisper →
agent receives text with [VOICE NOTE]: prefix → agent calls meeting.log with
the extracted fields AND generates action items via task.create.
"""
from datetime import datetime, timezone

from sqlalchemy import select

from ..core.skill_base import Skill, tool
from ..db.models import Meeting


def _parse_dt(s: str):
    if not s:
        return datetime.now(timezone.utc)
    for fmt in (
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return datetime.now(timezone.utc)


class MeetingSkill(Skill):
    name = "meeting"
    description = "Log meetings with attendees, summary, decisions, and transcript."

    def __init__(self, session_maker):
        super().__init__()
        self.session_maker = session_maker

    @tool(
        "Log a meeting. attendees is a comma-separated list of names. "
        "summary should be 2-4 sentences of what happened. decisions lists "
        "anything concrete that was decided. transcript is the raw voice-note "
        "text if captured. date in 'YYYY-MM-DD HH:MM' (UTC); empty = now. "
        "After logging, separately create action items for any commitments "
        "mentioned, and update contact personal_notes for any personal details."
    )
    async def log(
        self,
        summary: str,
        deal_id: str = "",
        attendees: str = "",
        decisions: str = "",
        transcript: str = "",
        date: str = "",
    ) -> dict:
        async with self.session_maker() as s:
            m = Meeting(
                deal_id=deal_id or None,
                attendees=attendees,
                summary=summary,
                decisions=decisions,
                transcript=transcript,
                date=_parse_dt(date),
            )
            s.add(m)
            await s.commit()
            await s.refresh(m)
            return {
                "id": m.id,
                "date": str(m.date),
                "summary": m.summary,
            }

    @tool("Get recent meetings for a deal, newest first.")
    async def recent(self, deal_id: str, limit: int = 10) -> list[dict]:
        async with self.session_maker() as s:
            result = await s.execute(
                select(Meeting)
                .where(Meeting.deal_id == deal_id)
                .order_by(Meeting.date.desc())
                .limit(limit)
            )
            return [
                {
                    "id": r.id,
                    "date": str(r.date),
                    "attendees": r.attendees,
                    "summary": r.summary,
                    "decisions": r.decisions,
                }
                for r in result.scalars().all()
            ]
