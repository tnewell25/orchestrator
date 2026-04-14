"""TaskSkill — action items / follow-ups / commitments.

Stored as ActionItem in the DB to avoid name collision with Claude's
'TaskCreate' conversation concept.
"""
from datetime import date, datetime, timezone

from sqlalchemy import select

from ..core.skill_base import Skill, tool
from ..db.models import ActionItem


def _parse_date(s: str) -> date | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


class TaskSkill(Skill):
    name = "task"
    description = "Manage action items, follow-ups, and commitments."

    def __init__(self, session_maker):
        super().__init__()
        self.session_maker = session_maker

    @tool(
        "Create an action item / follow-up. Use this whenever the user commits "
        "to doing something, or an action is extracted from a meeting. "
        "due_date in YYYY-MM-DD. source should be: manual, meeting, email, or proactive."
    )
    async def create(
        self,
        description: str,
        due_date: str = "",
        deal_id: str = "",
        contact_id: str = "",
        source: str = "manual",
    ) -> dict:
        async with self.session_maker() as s:
            a = ActionItem(
                description=description,
                due_date=_parse_date(due_date),
                deal_id=deal_id or None,
                contact_id=contact_id or None,
                source=source,
            )
            s.add(a)
            await s.commit()
            await s.refresh(a)
            return {
                "id": a.id,
                "description": a.description,
                "due_date": str(a.due_date) if a.due_date else None,
            }

    @tool("List open action items, optionally filtered by deal_id or due before a given YYYY-MM-DD.")
    async def list_open(self, deal_id: str = "", due_before: str = "") -> list[dict]:
        async with self.session_maker() as s:
            q = select(ActionItem).where(ActionItem.status == "open")
            if deal_id:
                q = q.where(ActionItem.deal_id == deal_id)
            due = _parse_date(due_before)
            if due:
                q = q.where(ActionItem.due_date <= due)
            q = q.order_by(ActionItem.due_date.asc().nullslast())
            result = await s.execute(q)
            return [
                {
                    "id": r.id,
                    "description": r.description,
                    "due_date": str(r.due_date) if r.due_date else None,
                    "deal_id": r.deal_id,
                    "contact_id": r.contact_id,
                    "source": r.source,
                }
                for r in result.scalars().all()
            ]

    @tool("Mark an action item as done.")
    async def complete(self, action_id: str) -> dict:
        async with self.session_maker() as s:
            a = await s.get(ActionItem, action_id)
            if not a:
                return {"error": f"Action item {action_id} not found"}
            a.status = "done"
            a.completed_at = datetime.now(timezone.utc)
            await s.commit()
            return {"id": a.id, "status": "done"}

    @tool("Snooze an action item — sets status to snoozed and updates due_date to a new YYYY-MM-DD.")
    async def snooze(self, action_id: str, new_due_date: str) -> dict:
        async with self.session_maker() as s:
            a = await s.get(ActionItem, action_id)
            if not a:
                return {"error": f"Action item {action_id} not found"}
            a.status = "snoozed"
            a.due_date = _parse_date(new_due_date)
            await s.commit()
            return {"id": a.id, "status": "snoozed", "new_due_date": new_due_date}

    @tool("List action items due today or overdue.")
    async def list_today(self) -> list[dict]:
        today = datetime.now(timezone.utc).date()
        async with self.session_maker() as s:
            result = await s.execute(
                select(ActionItem)
                .where(
                    ActionItem.status == "open",
                    ActionItem.due_date <= today,
                )
                .order_by(ActionItem.due_date.asc())
            )
            return [
                {
                    "id": r.id,
                    "description": r.description,
                    "due_date": str(r.due_date) if r.due_date else None,
                    "deal_id": r.deal_id,
                    "overdue": r.due_date < today if r.due_date else False,
                }
                for r in result.scalars().all()
            ]
