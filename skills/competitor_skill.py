"""CompetitorSkill — battle cards + win/loss pattern mining.

Battle cards are vector-indexed so the agent can pull "how to win vs Siemens
MindSphere for automotive OEM" on-demand.
"""
from sqlalchemy import select, func

from ..core.skill_base import Skill, tool
from ..db.models import BattleCard, Competitor, WinLossRecord


class CompetitorSkill(Skill):
    name = "competitor"
    description = "Manage competitor profiles, battle cards, win/loss patterns."

    def __init__(self, session_maker, memory):
        super().__init__()
        self.session_maker = session_maker
        self.memory = memory  # for vector store_vector / search_vector

    @tool(
        "Create or update a competitor profile. aliases comma-separated "
        "(e.g. 'MindSphere,Siemens Industrial Edge'). Strengths/weaknesses/pricing "
        "notes build the top-level profile."
    )
    async def upsert_competitor(
        self,
        name: str,
        aliases: str = "",
        strengths: str = "",
        weaknesses: str = "",
        pricing_notes: str = "",
    ) -> dict:
        async with self.session_maker() as s:
            existing = (
                await s.execute(select(Competitor).where(Competitor.name == name))
            ).scalar_one_or_none()
            if existing:
                if aliases:
                    existing.aliases = aliases
                if strengths:
                    existing.strengths = strengths
                if weaknesses:
                    existing.weaknesses = weaknesses
                if pricing_notes:
                    existing.pricing_notes = pricing_notes
                await s.commit()
                return {"id": existing.id, "name": existing.name, "updated": True}
            c = Competitor(
                name=name,
                aliases=aliases,
                strengths=strengths,
                weaknesses=weaknesses,
                pricing_notes=pricing_notes,
            )
            s.add(c)
            await s.commit()
            await s.refresh(c)
            return {"id": c.id, "name": c.name, "created": True}

    @tool(
        "Add a battle card for beating a competitor in a specific situation. "
        "Situation is the context ('when EC is CFO-level', 'automotive OEMs in EU'). "
        "Content is the talking points / objection handling. The card is "
        "vector-indexed so it can be pulled by semantic search."
    )
    async def add_battle_card(
        self,
        competitor_name: str,
        situation: str,
        content: str,
    ) -> dict:
        async with self.session_maker() as s:
            comp = (
                await s.execute(select(Competitor).where(Competitor.name == competitor_name))
            ).scalar_one_or_none()
            if not comp:
                # Auto-create competitor
                comp = Competitor(name=competitor_name)
                s.add(comp)
                await s.flush()
            card = BattleCard(
                competitor_id=comp.id, situation=situation, content=content
            )
            s.add(card)
            await s.commit()
            await s.refresh(card)

        # Embed using memory helper
        embed_text = f"{competitor_name} | {situation}\n{content}"
        await self.memory.store_vector("battle_cards", card.id, embed_text)
        return {"id": card.id, "competitor": competitor_name, "indexed": True}

    @tool(
        "Find the most relevant battle cards for a given situation via semantic "
        "search. Returns top N cards with their content. Use when the user is "
        "prepping for a meeting and mentions a competitor."
    )
    async def find_battle_card(self, situation: str, limit: int = 3) -> list[dict]:
        results = await self.memory.search_vector(
            table="battle_cards",
            text_col="content",
            query=situation,
            limit=limit,
            extra_cols=["situation", "competitor_id"],
        )
        # Enrich with competitor name
        async with self.session_maker() as s:
            for r in results:
                if r.get("competitor_id"):
                    c = await s.get(Competitor, r["competitor_id"])
                    r["competitor_name"] = c.name if c else None
        return results

    @tool(
        "Record a win or loss post-mortem for a deal. outcome: 'won' or 'lost'. "
        "winning_competitor is who beat you (if lost). primary_reason, what_worked, "
        "what_didnt, lessons feed pattern mining. Run after every closed deal."
    )
    async def log_win_loss(
        self,
        deal_id: str,
        outcome: str,
        primary_reason: str,
        winning_competitor: str = "",
        what_worked: str = "",
        what_didnt: str = "",
        lessons: str = "",
        value_usd: float = 0.0,
    ) -> dict:
        if outcome not in {"won", "lost"}:
            return {"error": "outcome must be 'won' or 'lost'"}
        async with self.session_maker() as s:
            r = WinLossRecord(
                deal_id=deal_id,
                outcome=outcome,
                winning_competitor=winning_competitor,
                primary_reason=primary_reason,
                what_worked=what_worked,
                what_didnt=what_didnt,
                lessons=lessons,
                value_usd=value_usd,
            )
            s.add(r)
            await s.commit()
            await s.refresh(r)
            return {"id": r.id, "outcome": outcome}

    @tool("Competitor win/loss summary — counts + rate per competitor.")
    async def win_loss_summary(self) -> dict:
        async with self.session_maker() as s:
            result = await s.execute(
                select(
                    WinLossRecord.winning_competitor,
                    WinLossRecord.outcome,
                    func.count(WinLossRecord.id),
                    func.sum(WinLossRecord.value_usd),
                )
                .group_by(WinLossRecord.winning_competitor, WinLossRecord.outcome)
            )
            summary: dict = {}
            for comp, outcome, count, value in result.all():
                key = comp or "(none)"
                summary.setdefault(key, {"won": 0, "lost": 0, "value_won": 0.0, "value_lost": 0.0})
                summary[key][outcome] += count
                summary[key][f"value_{outcome}"] += float(value or 0)
            return summary
