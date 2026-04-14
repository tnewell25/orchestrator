"""DealSkill — pipeline management."""
from datetime import date, datetime

from sqlalchemy import select, or_

from ..core.skill_base import Skill, tool
from ..db.models import Deal, Meeting, ActionItem


_STAGES = {
    "prospect",
    "qualified",
    "proposal",
    "negotiation",
    "closed_won",
    "closed_lost",
}


def _parse_date(s: str) -> date | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


class DealSkill(Skill):
    name = "deal"
    description = "Manage pipeline deals (create, update stage, track value and close date)."

    def __init__(self, session_maker):
        super().__init__()
        self.session_maker = session_maker

    @tool(
        "Create a new deal. stage must be one of: prospect, qualified, "
        "proposal, negotiation, closed_won, closed_lost. close_date in YYYY-MM-DD. "
        "competitors is a comma-separated list of competing vendors."
    )
    async def create(
        self,
        name: str,
        company_id: str = "",
        stage: str = "prospect",
        value_usd: float = 0.0,
        close_date: str = "",
        competitors: str = "",
        next_step: str = "",
        notes: str = "",
    ) -> dict:
        if stage not in _STAGES:
            return {"error": f"Invalid stage '{stage}'. Valid: {sorted(_STAGES)}"}
        async with self.session_maker() as s:
            d = Deal(
                name=name,
                company_id=company_id or None,
                stage=stage,
                value_usd=value_usd,
                close_date=_parse_date(close_date),
                competitors=competitors,
                next_step=next_step,
                notes=notes,
            )
            s.add(d)
            await s.commit()
            await s.refresh(d)
            return {"id": d.id, "name": d.name, "stage": d.stage}

    @tool("Find deals by name or company id substring. Returns deal summaries.")
    async def find(self, query: str) -> list[dict]:
        async with self.session_maker() as s:
            q = f"%{query.lower()}%"
            result = await s.execute(
                select(Deal).where(
                    or_(Deal.name.ilike(q), Deal.notes.ilike(q))
                ).limit(20)
            )
            return [
                {
                    "id": r.id,
                    "name": r.name,
                    "stage": r.stage,
                    "value_usd": r.value_usd,
                    "close_date": str(r.close_date) if r.close_date else None,
                    "next_step": r.next_step,
                    "company_id": r.company_id,
                }
                for r in result.scalars().all()
            ]

    @tool("List all deals in a given stage (prospect, qualified, proposal, negotiation, closed_won, closed_lost).")
    async def list_by_stage(self, stage: str) -> list[dict]:
        if stage not in _STAGES:
            return [{"error": f"Invalid stage '{stage}'"}]
        async with self.session_maker() as s:
            result = await s.execute(
                select(Deal).where(Deal.stage == stage).order_by(Deal.updated_at.desc())
            )
            return [
                {
                    "id": r.id,
                    "name": r.name,
                    "value_usd": r.value_usd,
                    "close_date": str(r.close_date) if r.close_date else None,
                    "next_step": r.next_step,
                }
                for r in result.scalars().all()
            ]

    @tool("Update deal fields. Only provided fields change.")
    async def update(
        self,
        deal_id: str,
        name: str = "",
        stage: str = "",
        value_usd: float = -1.0,
        close_date: str = "",
        competitors: str = "",
        next_step: str = "",
        notes: str = "",
    ) -> dict:
        async with self.session_maker() as s:
            d = await s.get(Deal, deal_id)
            if not d:
                return {"error": f"Deal {deal_id} not found"}
            if name:
                d.name = name
            if stage:
                if stage not in _STAGES:
                    return {"error": f"Invalid stage '{stage}'"}
                d.stage = stage
            if value_usd >= 0:
                d.value_usd = value_usd
            if close_date:
                d.close_date = _parse_date(close_date)
            if competitors:
                d.competitors = competitors
            if next_step:
                d.next_step = next_step
            if notes:
                d.notes = (d.notes + "\n" if d.notes else "") + notes
            await s.commit()
            return {"id": d.id, "name": d.name, "stage": d.stage, "updated": True}

    @tool(
        "Get full pipeline context for a deal: meetings, open action items, notes. "
        "Use this when the user asks 'what's going on with X?' or before a meeting."
    )
    async def get_context(self, deal_id: str) -> dict:
        async with self.session_maker() as s:
            d = await s.get(Deal, deal_id)
            if not d:
                return {"error": f"Deal {deal_id} not found"}
            meetings = (
                await s.execute(
                    select(Meeting)
                    .where(Meeting.deal_id == deal_id)
                    .order_by(Meeting.date.desc())
                    .limit(10)
                )
            ).scalars().all()
            open_actions = (
                await s.execute(
                    select(ActionItem)
                    .where(ActionItem.deal_id == deal_id, ActionItem.status == "open")
                    .order_by(ActionItem.due_date.asc().nullslast())
                )
            ).scalars().all()
            return {
                "deal": {
                    "id": d.id,
                    "name": d.name,
                    "stage": d.stage,
                    "value_usd": d.value_usd,
                    "close_date": str(d.close_date) if d.close_date else None,
                    "competitors": d.competitors,
                    "next_step": d.next_step,
                    "notes": d.notes,
                },
                "recent_meetings": [
                    {
                        "id": m.id,
                        "date": str(m.date),
                        "attendees": m.attendees,
                        "summary": m.summary,
                        "decisions": m.decisions,
                    }
                    for m in meetings
                ],
                "open_action_items": [
                    {
                        "id": a.id,
                        "description": a.description,
                        "due_date": str(a.due_date) if a.due_date else None,
                    }
                    for a in open_actions
                ],
            }
