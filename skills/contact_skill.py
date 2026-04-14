"""ContactSkill — CRUD for people with a focus on relationship memory.

personal_notes is the killer field: kids, hobbies, interests. The agent
should proactively surface this before meetings.
"""
from datetime import datetime, timezone

from sqlalchemy import select, or_

from ..core.skill_base import Skill, tool
from ..db.models import Contact


class ContactSkill(Skill):
    name = "contact"
    description = "Manage people in the network with personal relationship notes."

    def __init__(self, session_maker):
        super().__init__()
        self.session_maker = session_maker

    @tool(
        "Create a new contact. company_id is optional. personal_notes is for "
        "relationship details (kids, hobbies, interests) — always populate this "
        "when the user mentions personal details about someone."
    )
    async def create(
        self,
        name: str,
        company_id: str = "",
        title: str = "",
        email: str = "",
        phone: str = "",
        linkedin: str = "",
        personal_notes: str = "",
    ) -> dict:
        async with self.session_maker() as s:
            c = Contact(
                name=name,
                company_id=company_id or None,
                title=title,
                email=email,
                phone=phone,
                linkedin=linkedin,
                personal_notes=personal_notes,
            )
            s.add(c)
            await s.commit()
            await s.refresh(c)
            return {"id": c.id, "name": c.name}

    @tool("Find contacts by name, email, or title substring.")
    async def find(self, query: str) -> list[dict]:
        async with self.session_maker() as s:
            q = f"%{query.lower()}%"
            result = await s.execute(
                select(Contact).where(
                    or_(
                        Contact.name.ilike(q),
                        Contact.email.ilike(q),
                        Contact.title.ilike(q),
                    )
                ).limit(20)
            )
            return [
                {
                    "id": r.id,
                    "name": r.name,
                    "title": r.title,
                    "email": r.email,
                    "phone": r.phone,
                    "company_id": r.company_id,
                    "personal_notes": r.personal_notes,
                    "last_touch": str(r.last_touch) if r.last_touch else None,
                }
                for r in result.scalars().all()
            ]

    @tool(
        "Update a contact. Only provided fields change. "
        "Use this to append personal_notes when the user shares new relationship detail."
    )
    async def update(
        self,
        contact_id: str,
        name: str = "",
        company_id: str = "",
        title: str = "",
        email: str = "",
        phone: str = "",
        linkedin: str = "",
        personal_notes: str = "",
    ) -> dict:
        async with self.session_maker() as s:
            c = await s.get(Contact, contact_id)
            if not c:
                return {"error": f"Contact {contact_id} not found"}
            if name:
                c.name = name
            if company_id:
                c.company_id = company_id
            if title:
                c.title = title
            if email:
                c.email = email
            if phone:
                c.phone = phone
            if linkedin:
                c.linkedin = linkedin
            if personal_notes:
                # Append rather than overwrite — accumulate relationship detail
                c.personal_notes = (
                    (c.personal_notes + "\n" if c.personal_notes else "")
                    + personal_notes
                )
            await s.commit()
            return {"id": c.id, "name": c.name, "updated": True}

    @tool("Mark a contact as touched today (updates last_touch timestamp).")
    async def mark_touched(self, contact_id: str) -> dict:
        async with self.session_maker() as s:
            c = await s.get(Contact, contact_id)
            if not c:
                return {"error": f"Contact {contact_id} not found"}
            c.last_touch = datetime.now(timezone.utc)
            await s.commit()
            return {"id": c.id, "last_touch": str(c.last_touch)}
