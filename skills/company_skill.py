"""CompanySkill — CRUD for companies (Bosch, Honeywell, etc)."""
from sqlalchemy import select, or_

from ..core.skill_base import Skill, tool
from ..db.models import Company


class CompanySkill(Skill):
    name = "company"
    description = "Manage companies in the pipeline (create, find, update)."

    def __init__(self, session_maker):
        super().__init__()
        self.session_maker = session_maker

    @tool("Create a new company record. Returns the company id and name.")
    async def create(
        self,
        name: str,
        industry: str = "",
        website: str = "",
        notes: str = "",
    ) -> dict:
        async with self.session_maker() as s:
            c = Company(name=name, industry=industry, website=website, notes=notes)
            s.add(c)
            await s.commit()
            await s.refresh(c)
            return {"id": c.id, "name": c.name}

    @tool(
        "Find companies by name or industry (case-insensitive substring match). "
        "Returns a list with id, name, industry, website, notes."
    )
    async def find(self, query: str) -> list[dict]:
        async with self.session_maker() as s:
            q = f"%{query.lower()}%"
            result = await s.execute(
                select(Company).where(
                    or_(
                        Company.name.ilike(q),
                        Company.industry.ilike(q),
                    )
                ).limit(20)
            )
            return [
                {
                    "id": r.id,
                    "name": r.name,
                    "industry": r.industry,
                    "website": r.website,
                    "notes": r.notes,
                }
                for r in result.scalars().all()
            ]

    @tool("Update a company's fields. Only provided fields are changed.")
    async def update(
        self,
        company_id: str,
        name: str = "",
        industry: str = "",
        website: str = "",
        notes: str = "",
    ) -> dict:
        async with self.session_maker() as s:
            c = await s.get(Company, company_id)
            if not c:
                return {"error": f"Company {company_id} not found"}
            if name:
                c.name = name
            if industry:
                c.industry = industry
            if website:
                c.website = website
            if notes:
                c.notes = notes
            await s.commit()
            return {"id": c.id, "name": c.name, "updated": True}

    @tool("Get full context for a company: its contacts, open deals, and notes.")
    async def get_context(self, company_id: str) -> dict:
        from ..db.models import Contact, Deal

        async with self.session_maker() as s:
            c = await s.get(Company, company_id)
            if not c:
                return {"error": f"Company {company_id} not found"}
            contacts = (
                await s.execute(select(Contact).where(Contact.company_id == company_id))
            ).scalars().all()
            deals = (
                await s.execute(select(Deal).where(Deal.company_id == company_id))
            ).scalars().all()
            return {
                "company": {"id": c.id, "name": c.name, "industry": c.industry, "notes": c.notes},
                "contacts": [
                    {"id": ct.id, "name": ct.name, "title": ct.title, "email": ct.email}
                    for ct in contacts
                ],
                "deals": [
                    {"id": d.id, "name": d.name, "stage": d.stage, "value_usd": d.value_usd, "next_step": d.next_step}
                    for d in deals
                ],
            }
