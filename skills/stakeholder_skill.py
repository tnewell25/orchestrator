"""StakeholderSkill — 5-role map per deal (champion / EC / technical / blocker / coach / user).

Forces elite multi-threaded selling. A deal without a coverage map is a deal
at risk. Gap analysis surfaces exactly who to land next.
"""
from sqlalchemy import select

from ..core.skill_base import Skill, tool
from ..db.models import Contact, DealStakeholder


_ROLES = {"champion", "economic_buyer", "technical_buyer", "blocker", "coach", "user"}
_SENTIMENTS = {"supportive", "neutral", "opposed", "unknown"}
_INFLUENCES = {"low", "medium", "high"}


class StakeholderSkill(Skill):
    name = "stakeholder"
    description = "Map contacts to their role on a deal; analyze coverage."

    def __init__(self, session_maker):
        super().__init__()
        self.session_maker = session_maker

    @tool(
        "Assign a contact to a role on a deal. role must be one of: champion, "
        "economic_buyer, technical_buyer, blocker, coach, user. sentiment: "
        "supportive, neutral, opposed, unknown. influence: low, medium, high. "
        "If this contact+role already exists, fields are updated."
    )
    async def assign(
        self,
        deal_id: str,
        contact_id: str,
        role: str,
        sentiment: str = "unknown",
        influence: str = "medium",
        notes: str = "",
    ) -> dict:
        if role not in _ROLES:
            return {"error": f"Invalid role '{role}'. Valid: {sorted(_ROLES)}"}
        if sentiment not in _SENTIMENTS:
            return {"error": f"Invalid sentiment '{sentiment}'"}
        if influence not in _INFLUENCES:
            return {"error": f"Invalid influence '{influence}'"}

        async with self.session_maker() as s:
            existing = (
                await s.execute(
                    select(DealStakeholder).where(
                        DealStakeholder.deal_id == deal_id,
                        DealStakeholder.contact_id == contact_id,
                        DealStakeholder.role == role,
                    )
                )
            ).scalar_one_or_none()
            if existing:
                existing.sentiment = sentiment
                existing.influence = influence
                if notes:
                    existing.notes = (existing.notes + "\n" if existing.notes else "") + notes
            else:
                s.add(
                    DealStakeholder(
                        deal_id=deal_id,
                        contact_id=contact_id,
                        role=role,
                        sentiment=sentiment,
                        influence=influence,
                        notes=notes,
                    )
                )
            await s.commit()
            return {"deal_id": deal_id, "contact_id": contact_id, "role": role, "updated": True}

    @tool(
        "Get the full stakeholder map for a deal. Returns roles covered, gaps "
        "(roles with no contact), sentiment breakdown, and risks (blockers with "
        "high influence). Use to answer 'who's who on this deal?'."
    )
    async def coverage(self, deal_id: str) -> dict:
        async with self.session_maker() as s:
            result = await s.execute(
                select(DealStakeholder).where(DealStakeholder.deal_id == deal_id)
            )
            rows = list(result.scalars().all())

            # Resolve contacts
            mapped = []
            for r in rows:
                c = await s.get(Contact, r.contact_id)
                mapped.append(
                    {
                        "contact_id": r.contact_id,
                        "name": c.name if c else "(unknown)",
                        "title": c.title if c else "",
                        "role": r.role,
                        "sentiment": r.sentiment,
                        "influence": r.influence,
                        "notes": r.notes,
                    }
                )

            roles_covered = {m["role"] for m in mapped}
            critical = {"champion", "economic_buyer", "technical_buyer"}
            gaps = sorted(list(critical - roles_covered))

            risks = [m for m in mapped if m["role"] == "blocker" and m["influence"] == "high"]

            return {
                "deal_id": deal_id,
                "stakeholders": mapped,
                "roles_covered": sorted(list(roles_covered)),
                "critical_gaps": gaps,
                "risks": risks,
                "coverage_score": round(100 * len(critical & roles_covered) / len(critical)),
            }

    @tool("Remove a stakeholder assignment.")
    async def remove(self, deal_id: str, contact_id: str, role: str) -> dict:
        async with self.session_maker() as s:
            existing = (
                await s.execute(
                    select(DealStakeholder).where(
                        DealStakeholder.deal_id == deal_id,
                        DealStakeholder.contact_id == contact_id,
                        DealStakeholder.role == role,
                    )
                )
            ).scalar_one_or_none()
            if not existing:
                return {"error": "Not found"}
            await s.delete(existing)
            await s.commit()
            return {"removed": True}
