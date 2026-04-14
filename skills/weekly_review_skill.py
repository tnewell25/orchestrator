"""WeeklyReviewSkill — Friday auto-draft: what happened, what's next, quota status."""
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, func

from ..core.skill_base import Skill, tool
from ..db.models import ActionItem, Deal, Meeting, WinLossRecord


class WeeklyReviewSkill(Skill):
    name = "weekly"
    description = "Generate weekly review summaries."

    def __init__(self, session_maker):
        super().__init__()
        self.session_maker = session_maker

    @tool(
        "Generate this week's review: meetings held, deals moved, wins/losses, "
        "open action items carrying into next week, next week's meeting count, "
        "pipeline snapshot. Returns structured data the agent formats into a narrative."
    )
    async def generate(self) -> dict:
        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(days=7)
        next_week = now + timedelta(days=7)

        async with self.session_maker() as s:
            # Meetings this week
            meetings = (
                await s.execute(
                    select(Meeting)
                    .where(Meeting.date >= week_ago, Meeting.date <= now)
                    .order_by(Meeting.date.desc())
                )
            ).scalars().all()

            # Deals that moved stage this week
            moved = (
                await s.execute(
                    select(Deal).where(Deal.updated_at >= week_ago)
                )
            ).scalars().all()

            # Wins/losses
            won = (
                await s.execute(
                    select(WinLossRecord)
                    .where(WinLossRecord.created_at >= week_ago, WinLossRecord.outcome == "won")
                )
            ).scalars().all()
            lost = (
                await s.execute(
                    select(WinLossRecord)
                    .where(WinLossRecord.created_at >= week_ago, WinLossRecord.outcome == "lost")
                )
            ).scalars().all()

            # Open actions carrying over
            open_actions = (
                await s.execute(
                    select(ActionItem).where(ActionItem.status == "open")
                )
            ).scalars().all()

            # Upcoming meetings
            upcoming = (
                await s.execute(
                    select(Meeting)
                    .where(Meeting.date > now, Meeting.date <= next_week)
                )
            ).scalars().all()

            # Pipeline snapshot
            by_stage = await s.execute(
                select(Deal.stage, func.count(Deal.id), func.sum(Deal.value_usd))
                .group_by(Deal.stage)
            )
            pipeline = {row[0]: {"count": row[1], "value": float(row[2] or 0)} for row in by_stage.all()}

            return {
                "window": {"from": str(week_ago), "to": str(now)},
                "meetings_this_week": [
                    {"id": m.id, "date": str(m.date), "summary": m.summary[:200]}
                    for m in meetings
                ],
                "deals_moved": [
                    {"id": d.id, "name": d.name, "stage": d.stage, "value_usd": d.value_usd}
                    for d in moved
                ],
                "wins": [
                    {"deal_id": w.deal_id, "value_usd": w.value_usd, "reason": w.primary_reason}
                    for w in won
                ],
                "losses": [
                    {"deal_id": l.deal_id, "value_usd": l.value_usd, "reason": l.primary_reason, "competitor": l.winning_competitor}
                    for l in lost
                ],
                "open_actions_total": len(open_actions),
                "overdue_actions": [
                    {"id": a.id, "description": a.description, "due_date": str(a.due_date)}
                    for a in open_actions
                    if a.due_date and a.due_date < now.date()
                ],
                "upcoming_meetings_next_week": len(upcoming),
                "pipeline_snapshot": pipeline,
            }
