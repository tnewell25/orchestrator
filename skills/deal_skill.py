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
        "Get full pipeline context for a deal: MEDDIC status, meetings, open action "
        "items, bids, competitors. Use this when the user asks 'what's going on with X?' "
        "or before a meeting. The MEDDIC gaps are flagged so the agent can nudge the "
        "user to fill them in on the next call."
    )
    async def get_context(self, deal_id: str) -> dict:
        from ..db.models import Bid, Contact

        async with self.session_maker() as s:
            d = await s.get(Deal, deal_id)
            if not d:
                return {"error": f"Deal {deal_id} not found"}
            meetings = (
                await s.execute(
                    select(Meeting)
                    .where(Meeting.deal_id == deal_id)
                    .order_by(Meeting.date.desc())
                    .limit(5)
                )
            ).scalars().all()
            open_actions = (
                await s.execute(
                    select(ActionItem)
                    .where(ActionItem.deal_id == deal_id, ActionItem.status == "open")
                    .order_by(ActionItem.due_date.asc().nullslast())
                )
            ).scalars().all()
            bids = (
                await s.execute(
                    select(Bid).where(Bid.deal_id == deal_id).order_by(Bid.submission_deadline.asc().nullslast())
                )
            ).scalars().all()

            ec = await s.get(Contact, d.economic_buyer_id) if d.economic_buyer_id else None
            ch = await s.get(Contact, d.champion_id) if d.champion_id else None

            meddic_gaps = [
                field for field, val in [
                    ("metrics", d.metrics),
                    ("economic_buyer", d.economic_buyer_id),
                    ("decision_criteria", d.decision_criteria),
                    ("decision_process", d.decision_process),
                    ("paper_process", d.paper_process),
                    ("pain", d.pain),
                    ("champion", d.champion_id),
                    ("competitors", d.competitors),
                ] if not val
            ]

            return {
                "deal": {
                    "id": d.id,
                    "name": d.name,
                    "stage": d.stage,
                    "value_usd": d.value_usd,
                    "close_date": str(d.close_date) if d.close_date else None,
                    "next_step": d.next_step,
                    "notes": d.notes,
                },
                "meddic": {
                    "metrics": d.metrics or None,
                    "economic_buyer": {"id": ec.id, "name": ec.name, "title": ec.title} if ec else None,
                    "decision_criteria": d.decision_criteria or None,
                    "decision_process": d.decision_process or None,
                    "paper_process": d.paper_process or None,
                    "pain": d.pain or None,
                    "champion": {"id": ch.id, "name": ch.name, "title": ch.title, "personal_notes": ch.personal_notes} if ch else None,
                    "competitors": d.competitors or None,
                    "gaps": meddic_gaps,  # agent should nudge user to fill these
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
                "bids": [
                    {
                        "id": b.id,
                        "name": b.name,
                        "stage": b.stage,
                        "value_usd": b.value_usd,
                        "submission_deadline": b.submission_deadline.isoformat() if b.submission_deadline else None,
                    }
                    for b in bids
                ],
            }

    @tool(
        "Set MEDDIC stakeholders on a deal. Pass contact IDs. Either field optional — "
        "only what's provided is updated. After setting, reminders to verify these "
        "stakeholders should be set separately by the agent if this is new info."
    )
    async def set_stakeholders(
        self,
        deal_id: str,
        economic_buyer_id: str = "",
        champion_id: str = "",
    ) -> dict:
        async with self.session_maker() as s:
            d = await s.get(Deal, deal_id)
            if not d:
                return {"error": f"Deal {deal_id} not found"}
            if economic_buyer_id:
                d.economic_buyer_id = economic_buyer_id
            if champion_id:
                d.champion_id = champion_id
            await s.commit()
            return {"id": d.id, "updated": True}

    @tool(
        "Update any MEDDIC qualitative field. field must be one of: metrics, "
        "decision_criteria, decision_process, paper_process, pain."
    )
    async def set_meddic_field(self, deal_id: str, field: str, value: str) -> dict:
        valid = {"metrics", "decision_criteria", "decision_process", "paper_process", "pain"}
        if field not in valid:
            return {"error": f"Invalid field '{field}'. Valid: {sorted(valid)}"}
        async with self.session_maker() as s:
            d = await s.get(Deal, deal_id)
            if not d:
                return {"error": f"Deal {deal_id} not found"}
            setattr(d, field, value)
            await s.commit()
            return {"id": d.id, "field": field, "updated": True}
