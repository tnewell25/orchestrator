"""Parallel sub-agent fan-out for STRATEGY intent.

When the planner identifies a strategic request ("how do we win Bosch?",
"why are we losing to Siemens?"), we don't want the main Sonnet+thinking
call to wander through raw tables and piece the picture together. Instead
we fan out two cheap Haiku calls in parallel, each specialized:

  1. HistoricalLessonsAgent — reads WinLossRecord for similar deals, distills
     "what worked / what didn't / patterns to replicate or avoid."
  2. RelationshipMapAgent — reads DealStakeholder + Contact.personal_notes for
     the focus deal, distills "who's the champion, who's the blocker, what
     personal leverage exists."

Results get stitched into a STRATEGIC CONTEXT block that the main agent's
Block D includes. The main agent now spends its big-model thinking on
SYNTHESIS, not fact-gathering.

Cost vs no fan-out: +~2k Haiku tokens (cheap) but saves Sonnet from reading
+ reasoning over raw tables (~5-10k tokens saved on the expensive tier).
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from ..db.models import Contact, Deal, DealStakeholder, WinLossRecord
from .graph import EntityRef
from .planner import Intent, Plan

logger = logging.getLogger(__name__)


_LESSONS_SYSTEM = """You synthesize win/loss history into tactical lessons.

Given a list of closed deal records (win or loss), output 3-5 bullet lessons.
Each bullet: what pattern played out + actionable implication for similar
open deals. No preamble, no closing. If data is thin, say so in one line."""


_RELATIONSHIP_SYSTEM = """You map a deal's human landscape from stakeholder data.

Given stakeholders with roles, sentiment, influence + their personal notes,
output a crisp relationship brief:
  - champion: <name, why they're backing us>
  - blocker: <name, what they're worried about> (or "none known")
  - economic_buyer: <name, what they care about>
  - conversation ammo: <1-2 personal-note hooks>

Skip sections with no data. Keep total output under 150 words."""


@dataclass
class StrategyContext:
    lessons: str = ""
    relationships: str = ""

    def to_block(self) -> str:
        parts = []
        if self.lessons:
            parts.append("HISTORICAL LESSONS:\n" + self.lessons)
        if self.relationships:
            parts.append("RELATIONSHIP MAP:\n" + self.relationships)
        return "\n\n".join(parts)

    def is_empty(self) -> bool:
        return not (self.lessons or self.relationships)


class StrategyFanout:
    """Runs parallel sub-agents and stitches results."""

    def __init__(
        self,
        session_maker: async_sessionmaker,
        anthropic_client,
        fast_model: str = "claude-haiku-4-5-20251001",
        max_tokens_per_sub: int = 400,
    ):
        self.sm = session_maker
        self.client = anthropic_client
        self.fast_model = fast_model
        self.max_tokens_per_sub = max_tokens_per_sub

    def should_run(self, plan: Plan) -> bool:
        """Only STRATEGY intent with a known focus entity benefits from fan-out."""
        return (
            plan.intent == Intent.STRATEGY
            and plan.focus is not None
            and self.client is not None
        )

    async def gather(self, plan: Plan) -> StrategyContext:
        """Run both sub-agents in parallel, return synthesized context."""
        if not self.should_run(plan):
            return StrategyContext()

        focus = plan.focus
        lessons_task = asyncio.create_task(self._lessons_sub(focus))
        rel_task = asyncio.create_task(self._relationship_sub(focus))

        results = await asyncio.gather(lessons_task, rel_task, return_exceptions=True)
        lessons = results[0] if not isinstance(results[0], Exception) else ""
        relationships = results[1] if not isinstance(results[1], Exception) else ""
        return StrategyContext(lessons=lessons, relationships=relationships)

    # ---- Historical lessons ----

    async def _lessons_sub(self, focus: EntityRef) -> str:
        records = await self._fetch_winloss(focus)
        if not records:
            return ""
        payload = "\n".join(
            f"- {r.outcome.upper()} ({r.value_usd or 0:.0f}): "
            f"reason={r.primary_reason or 'unspecified'}; "
            f"worked={(r.what_worked or '')[:120]}; "
            f"didnt={(r.what_didnt or '')[:120]}; "
            f"lessons={(r.lessons or '')[:150]}"
            for r in records
        )
        return await self._call_haiku(_LESSONS_SYSTEM, payload)

    async def _fetch_winloss(self, focus: EntityRef) -> list[WinLossRecord]:
        """Pull win/loss records relevant to the focus.

        If focus is a Deal, pull that deal's record (if closed) + records
        from the same Company. If focus is a Company, all its deals.
        Capped at ~10 records to keep the Haiku prompt lean.
        """
        async with self.sm() as session:
            if focus.type == "Deal":
                deal = await session.get(Deal, focus.id)
                if deal is None:
                    return []
                company_id = deal.company_id
                deal_ids = [focus.id]
                if company_id:
                    rows = (
                        await session.execute(
                            select(Deal.id).where(Deal.company_id == company_id)
                        )
                    ).all()
                    deal_ids = list({r[0] for r in rows} | {focus.id})
            elif focus.type == "Company":
                rows = (
                    await session.execute(
                        select(Deal.id).where(Deal.company_id == focus.id)
                    )
                ).all()
                deal_ids = [r[0] for r in rows]
            else:
                return []

            if not deal_ids:
                return []

            records = (
                await session.execute(
                    select(WinLossRecord)
                    .where(WinLossRecord.deal_id.in_(deal_ids))
                    .order_by(WinLossRecord.created_at.desc())
                    .limit(10)
                )
            ).scalars().all()
            return list(records)

    # ---- Relationship map ----

    async def _relationship_sub(self, focus: EntityRef) -> str:
        stakeholders = await self._fetch_stakeholders(focus)
        if not stakeholders:
            return ""
        payload = "\n".join(
            f"- {s['role']}: {s['name']} (influence={s['influence']}, "
            f"sentiment={s['sentiment']}){' — ' + s['personal_notes'][:200] if s['personal_notes'] else ''}"
            for s in stakeholders
        )
        return await self._call_haiku(_RELATIONSHIP_SYSTEM, payload)

    async def _fetch_stakeholders(self, focus: EntityRef) -> list[dict]:
        """For Deal focus, return stakeholder rows joined with contact names +
        personal_notes. For Company, aggregate across all its deals."""
        async with self.sm() as session:
            if focus.type == "Deal":
                deal_ids = [focus.id]
            elif focus.type == "Company":
                rows = (
                    await session.execute(select(Deal.id).where(Deal.company_id == focus.id))
                ).all()
                deal_ids = [r[0] for r in rows]
            else:
                return []
            if not deal_ids:
                return []

            stakeholder_rows = (
                await session.execute(
                    select(DealStakeholder).where(DealStakeholder.deal_id.in_(deal_ids))
                )
            ).scalars().all()
            if not stakeholder_rows:
                return []

            contact_ids = {s.contact_id for s in stakeholder_rows}
            contacts = (
                await session.execute(
                    select(Contact).where(Contact.id.in_(contact_ids))
                )
            ).scalars().all()
            contact_map = {c.id: c for c in contacts}

            out = []
            for s in stakeholder_rows:
                c = contact_map.get(s.contact_id)
                if c is None:
                    continue
                out.append({
                    "role": s.role, "name": c.name,
                    "influence": s.influence or "unknown",
                    "sentiment": s.sentiment or "unknown",
                    "personal_notes": c.personal_notes or "",
                })
            return out

    # ---- LLM call ----

    async def _call_haiku(self, system_prompt: str, user_payload: str) -> str:
        try:
            resp = await self.client.messages.create(
                model=self.fast_model,
                max_tokens=self.max_tokens_per_sub,
                system=system_prompt,
                messages=[{"role": "user", "content": user_payload}],
            )
            return "".join(b.text for b in resp.content if b.type == "text").strip()
        except Exception as e:
            logger.warning("Strategy sub-agent LLM call failed: %s", e)
            return ""


__all__ = ["StrategyContext", "StrategyFanout"]
