"""BriefingSkill — daily brief + pipeline snapshot + pre-meeting prep."""
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import select, func

from ..core.skill_base import Skill, tool
from ..db.models import ActionItem, Deal, Meeting


class BriefingSkill(Skill):
    name = "briefing"
    description = "Generate daily briefs, pipeline snapshots, and meeting prep."

    def __init__(self, session_maker):
        super().__init__()
        self.session_maker = session_maker

    @tool(
        "Generate today's brief: overdue/due-today action items, deals closing "
        "this week, and recent meetings from the last 48 hours."
    )
    async def daily_brief(self) -> dict:
        today = datetime.now(timezone.utc).date()
        week_ahead = today + timedelta(days=7)
        yesterday = datetime.now(timezone.utc) - timedelta(days=2)

        async with self.session_maker() as s:
            overdue = (
                await s.execute(
                    select(ActionItem)
                    .where(
                        ActionItem.status == "open",
                        ActionItem.due_date <= today,
                    )
                    .order_by(ActionItem.due_date.asc())
                )
            ).scalars().all()

            closing_soon = (
                await s.execute(
                    select(Deal)
                    .where(
                        Deal.close_date >= today,
                        Deal.close_date <= week_ahead,
                        Deal.stage.notin_(["closed_won", "closed_lost"]),
                    )
                    .order_by(Deal.close_date.asc())
                )
            ).scalars().all()

            recent_meetings = (
                await s.execute(
                    select(Meeting)
                    .where(Meeting.date >= yesterday)
                    .order_by(Meeting.date.desc())
                )
            ).scalars().all()

            return {
                "date": str(today),
                "overdue_or_due_today": [
                    {
                        "id": a.id,
                        "description": a.description,
                        "due_date": str(a.due_date) if a.due_date else None,
                        "deal_id": a.deal_id,
                    }
                    for a in overdue
                ],
                "deals_closing_this_week": [
                    {
                        "id": d.id,
                        "name": d.name,
                        "stage": d.stage,
                        "value_usd": d.value_usd,
                        "close_date": str(d.close_date),
                        "next_step": d.next_step,
                    }
                    for d in closing_soon
                ],
                "recent_meetings": [
                    {
                        "id": m.id,
                        "date": str(m.date),
                        "summary": m.summary,
                    }
                    for m in recent_meetings
                ],
            }

    @tool("Snapshot of the pipeline: count and total value by stage.")
    async def pipeline_snapshot(self) -> dict:
        async with self.session_maker() as s:
            result = await s.execute(
                select(Deal.stage, func.count(Deal.id), func.sum(Deal.value_usd))
                .group_by(Deal.stage)
            )
            snapshot = {}
            for stage, count, total in result.all():
                snapshot[stage] = {"count": count, "total_value_usd": float(total or 0)}
            return snapshot
