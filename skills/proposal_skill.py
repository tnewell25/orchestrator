"""ProposalSkill — save precedent sections + draft new proposals from them.

Flow:
  1. User saves past-proposal sections via save_precedent → vector-indexed.
  2. When drafting a new proposal, find_precedent pulls the best-matching
     sections (e.g. warranty clauses for similar scope).
  3. draft_proposal composes these with deal context into a markdown doc.
"""
from sqlalchemy import select

from ..core.skill_base import Skill, tool
from ..db.models import Deal, ProposalPrecedent


_SECTION_TYPES = {
    "intro", "scope", "approach", "deliverables", "timeline",
    "pricing", "assumptions", "warranty", "terms", "team", "case_study",
}


class ProposalSkill(Skill):
    name = "proposal"
    description = "Draft proposals from vector-indexed precedent sections."

    def __init__(self, session_maker, memory):
        super().__init__()
        self.session_maker = session_maker
        self.memory = memory

    @tool(
        "Save a proposal section as reusable precedent. section_type: intro, scope, "
        "approach, deliverables, timeline, pricing, assumptions, warranty, terms, "
        "team, case_study. tags comma-separated for context ('automotive,condition-monitoring'). "
        "Auto-vector-indexed so future drafts can pull it."
    )
    async def save_precedent(
        self,
        title: str,
        content: str,
        section_type: str = "",
        tags: str = "",
        source_deal_id: str = "",
    ) -> dict:
        if section_type and section_type not in _SECTION_TYPES:
            return {"error": f"Invalid section_type. Valid: {sorted(_SECTION_TYPES)}"}
        async with self.session_maker() as s:
            p = ProposalPrecedent(
                title=title,
                content=content,
                section_type=section_type,
                tags=tags,
                source_deal_id=source_deal_id or None,
            )
            s.add(p)
            await s.commit()
            await s.refresh(p)

        embed_text = f"{title} [{section_type}] {tags}\n{content}"
        await self.memory.store_vector("proposal_precedents", p.id, embed_text)
        return {"id": p.id, "title": p.title, "indexed": True}

    @tool(
        "Find the most relevant precedent sections for a given context via "
        "semantic search. Use when starting a new proposal."
    )
    async def find_precedent(
        self, context: str, section_type: str = "", limit: int = 3
    ) -> list[dict]:
        results = await self.memory.search_vector(
            table="proposal_precedents",
            text_col="content",
            query=context,
            limit=limit * 2 if section_type else limit,
            extra_cols=["title", "section_type", "tags"],
        )
        if section_type:
            results = [r for r in results if r.get("section_type") == section_type][:limit]
        return results

    @tool(
        "Draft a full proposal markdown by composing precedent + deal context. "
        "sections is a comma-separated list of section_types to include "
        "(default: intro,scope,approach,deliverables,timeline,pricing,warranty). "
        "The output is markdown the user can paste into Word/Google Docs or edit."
    )
    async def draft_proposal(
        self,
        deal_id: str,
        sections: str = "intro,scope,approach,deliverables,timeline,pricing,warranty",
    ) -> dict:
        async with self.session_maker() as s:
            d = await s.get(Deal, deal_id)
            if not d:
                return {"error": f"Deal {deal_id} not found"}

        section_list = [sec.strip() for sec in sections.split(",") if sec.strip()]
        context = (
            f"{d.name}. Pain: {d.pain}. Metrics: {d.metrics}. "
            f"Value: ${d.value_usd:,.0f}. Decision criteria: {d.decision_criteria}. "
            f"Notes: {d.notes}"
        )

        md_out = [f"# Proposal — {d.name}\n"]
        md_out.append(f"**Prepared for:** {d.company_id or 'Client'}\n")
        md_out.append(f"**Estimated value:** ${d.value_usd:,.0f}\n\n")

        for sec in section_list:
            precedents = await self.find_precedent(context, section_type=sec, limit=1)
            header = sec.replace("_", " ").title()
            md_out.append(f"## {header}\n")
            if precedents:
                md_out.append(precedents[0]["content"])
            else:
                md_out.append(f"_TODO: no precedent found for '{sec}' — draft from scratch._")
            md_out.append("\n")

        return {
            "deal_id": deal_id,
            "markdown": "\n".join(md_out),
            "sections_included": section_list,
        }
