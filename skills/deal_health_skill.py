"""DealHealthSkill — temperature scoring + stalled-deal intelligence.

Score = weighted blend of:
  - Recency: days since last meeting (newer = better, exp decay)
  - MEDDIC completeness (fraction of MEDDIC fields filled)
  - Stakeholder coverage (champion + EC + technical_buyer)
  - Stage progress (qualified+ beats prospect)
  - Activity volume (meetings last 30d)

Temperature 0-100. 70+ hot, 40-70 warm, <40 cold/stalled.
"""
from datetime import datetime, timedelta, timezone
from math import exp

from sqlalchemy import select, func

from ..core.skill_base import Skill, tool
from ..db.models import Deal, DealStakeholder, Meeting


_STAGE_WEIGHT = {
    "prospect": 0.3,
    "qualified": 0.5,
    "proposal": 0.7,
    "negotiation": 0.85,
    "closed_won": 1.0,
    "closed_lost": 0.0,
}


def _meddic_score(d: Deal) -> float:
    fields = [
        d.metrics, d.economic_buyer_id, d.decision_criteria, d.decision_process,
        d.paper_process, d.pain, d.champion_id, d.competitors,
    ]
    filled = sum(1 for f in fields if f)
    return filled / len(fields)


class DealHealthSkill(Skill):
    name = "dealhealth"
    description = "Score deal health, find stalled deals, analyze pipeline."

    def __init__(self, session_maker):
        super().__init__()
        self.session_maker = session_maker

    async def _compute_score(self, session, d: Deal) -> dict:
        now = datetime.now(timezone.utc)

        # Last meeting recency
        last_m = (
            await session.execute(
                select(func.max(Meeting.date)).where(Meeting.deal_id == d.id)
            )
        ).scalar_one_or_none()
        days_since = (now - last_m).days if last_m else 999
        recency = exp(-days_since / 30)  # 30-day half-life

        # Activity volume last 30d
        cutoff = now - timedelta(days=30)
        activity_count = (
            await session.execute(
                select(func.count(Meeting.id))
                .where(Meeting.deal_id == d.id, Meeting.date >= cutoff)
            )
        ).scalar_one() or 0
        activity = min(activity_count / 4, 1.0)  # 4+ meetings/mo = full

        # MEDDIC
        meddic = _meddic_score(d)

        # Stakeholder coverage
        stake_rows = (
            await session.execute(
                select(DealStakeholder.role).where(DealStakeholder.deal_id == d.id).distinct()
            )
        ).all()
        roles = {r[0] for r in stake_rows}
        if d.champion_id:
            roles.add("champion")
        if d.economic_buyer_id:
            roles.add("economic_buyer")
        critical = {"champion", "economic_buyer", "technical_buyer"}
        stake = len(roles & critical) / len(critical)

        # Stage
        stage_weight = _STAGE_WEIGHT.get(d.stage, 0.3)

        # Blend
        weights = {"recency": 0.25, "meddic": 0.20, "stake": 0.20, "activity": 0.15, "stage": 0.20}
        score = 100 * (
            weights["recency"] * recency
            + weights["meddic"] * meddic
            + weights["stake"] * stake
            + weights["activity"] * activity
            + weights["stage"] * stage_weight
        )

        return {
            "deal_id": d.id,
            "name": d.name,
            "stage": d.stage,
            "value_usd": d.value_usd,
            "temperature": round(score, 1),
            "components": {
                "recency": round(100 * recency, 1),
                "meddic_completeness": round(100 * meddic, 1),
                "stakeholder_coverage": round(100 * stake, 1),
                "activity_30d": activity_count,
                "stage_weight": round(100 * stage_weight, 1),
            },
            "days_since_last_meeting": days_since if last_m else None,
            "label": (
                "hot" if score >= 70
                else "warm" if score >= 40
                else "cold" if score >= 20
                else "stalled"
            ),
        }

    @tool("Score a single deal's health. Returns temperature 0-100 + component breakdown.")
    async def score(self, deal_id: str) -> dict:
        async with self.session_maker() as s:
            d = await s.get(Deal, deal_id)
            if not d:
                return {"error": f"Deal {deal_id} not found"}
            return await self._compute_score(s, d)

    @tool(
        "List stalled deals (no meeting in N days, default 14). Open deals only. "
        "Use for proactive re-engagement."
    )
    async def stalled(self, days_since_last: int = 14) -> list[dict]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_since_last)
        async with self.session_maker() as s:
            result = await s.execute(
                select(Deal).where(
                    Deal.stage.notin_({"closed_won", "closed_lost"})
                )
            )
            stalled = []
            for d in result.scalars().all():
                last_m = (
                    await s.execute(
                        select(func.max(Meeting.date)).where(Meeting.deal_id == d.id)
                    )
                ).scalar_one_or_none()
                if last_m and last_m < cutoff:
                    days = (datetime.now(timezone.utc) - last_m).days
                    stalled.append({
                        "deal_id": d.id, "name": d.name, "stage": d.stage,
                        "value_usd": d.value_usd, "days_stalled": days,
                        "last_meeting": str(last_m),
                    })
                elif not last_m:
                    stalled.append({
                        "deal_id": d.id, "name": d.name, "stage": d.stage,
                        "value_usd": d.value_usd, "days_stalled": None,
                        "last_meeting": None, "note": "no meetings logged",
                    })
            stalled.sort(key=lambda x: x["value_usd"], reverse=True)
            return stalled

    @tool(
        "Pipeline health overview: temperature distribution (hot/warm/cold/stalled) "
        "with counts and total value at each tier."
    )
    async def pipeline_health(self) -> dict:
        async with self.session_maker() as s:
            result = await s.execute(
                select(Deal).where(Deal.stage.notin_({"closed_won", "closed_lost"}))
            )
            buckets = {"hot": [], "warm": [], "cold": [], "stalled": []}
            for d in result.scalars().all():
                score = await self._compute_score(s, d)
                buckets[score["label"]].append(score)
            return {
                label: {
                    "count": len(items),
                    "total_value_usd": sum(i["value_usd"] for i in items),
                    "deals": [
                        {"id": i["deal_id"], "name": i["name"], "temp": i["temperature"]}
                        for i in items
                    ],
                }
                for label, items in buckets.items()
            }
