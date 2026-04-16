"""UsageSkill — expose token cost summaries to the agent as tools.

Mirrors the /usage HTTP endpoint so questions like 'what's my usage today?'
or 'how much have I spent this week?' resolve in-chat without the user
needing to know an endpoint URL.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select

from ..core.skill_base import Skill, tool
from ..db.models import AuditLog


# Approximate per-model USD per 1M tokens. Keep in sync with main.py PRICING.
PRICING = {
    "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00, "cache_read": 0.30, "cache_creation": 3.75},
    "claude-sonnet-4-5":          {"input": 3.00, "output": 15.00, "cache_read": 0.30, "cache_creation": 3.75},
    "claude-haiku-4-5-20251001":  {"input": 1.00, "output": 5.00,  "cache_read": 0.10, "cache_creation": 1.25},
    "claude-haiku-4-5":           {"input": 1.00, "output": 5.00,  "cache_read": 0.10, "cache_creation": 1.25},
}
_DEFAULT = {"input": 3.00, "output": 15.00, "cache_read": 0.30, "cache_creation": 3.75}


def _cost_for(model: str, input_tok: int, output_tok: int, cache_read: int, cache_creation: int) -> float:
    p = PRICING.get(model or "", _DEFAULT)
    return (
        input_tok * p["input"] / 1_000_000
        + output_tok * p["output"] / 1_000_000
        + cache_read * p["cache_read"] / 1_000_000
        + cache_creation * p["cache_creation"] / 1_000_000
    )


class UsageSkill(Skill):
    name = "usage"
    description = "Token + cost usage analytics from the audit log."

    def __init__(self, session_maker):
        super().__init__()
        self.session_maker = session_maker

    @tool(
        "Get aggregated token usage + estimated cost over the last `hours` "
        "(default 24). Use `hours=168` for a week, `hours=0` for all time. "
        "Returns per-model breakdown + cache hit ratio + total $."
    )
    async def summary(self, hours: int = 24) -> dict:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours) if hours > 0 else None

        async with self.session_maker() as s:
            q = select(
                AuditLog.model,
                func.count().label("turns"),
                func.sum(AuditLog.input_tokens).label("input_tokens"),
                func.sum(AuditLog.output_tokens).label("output_tokens"),
                func.sum(AuditLog.cache_read_tokens).label("cache_read_tokens"),
                func.sum(AuditLog.cache_creation_tokens).label("cache_creation_tokens"),
            ).where(AuditLog.tool_name == "_turn")
            if cutoff:
                q = q.where(AuditLog.timestamp >= cutoff)
            q = q.group_by(AuditLog.model)
            rows = (await s.execute(q)).all()

        by_model = {}
        total_turns = 0
        total_cost = 0.0
        total_input = 0
        total_output = 0
        total_cache_read = 0

        for r in rows:
            model = r.model or "(unknown)"
            i = int(r.input_tokens or 0)
            o = int(r.output_tokens or 0)
            cr = int(r.cache_read_tokens or 0)
            cc = int(r.cache_creation_tokens or 0)
            cost = _cost_for(model, i, o, cr, cc)
            denom = i + cr + cc
            hit = round(cr / denom, 3) if denom else 0.0

            by_model[model] = {
                "turns": int(r.turns or 0),
                "input_tokens": i, "output_tokens": o,
                "cache_read_tokens": cr, "cache_creation_tokens": cc,
                "cache_hit_ratio": hit,
                "estimated_cost_usd": round(cost, 4),
            }
            total_turns += int(r.turns or 0)
            total_cost += cost
            total_input += i
            total_output += o
            total_cache_read += cr

        # "What would this have cost without caching?" — helps quantify W5 savings.
        savings = 0.0
        for model, data in by_model.items():
            p = PRICING.get(model, _DEFAULT)
            savings += data["cache_read_tokens"] * (p["input"] - p["cache_read"]) / 1_000_000

        return {
            "window_hours": hours if hours > 0 else "all",
            "total_turns": total_turns,
            "total_estimated_cost_usd": round(total_cost, 4),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cache_read_tokens": total_cache_read,
            "cache_savings_vs_uncached_usd": round(savings, 4),
            "by_model": by_model,
            "note": "Estimated — authoritative bill at console.anthropic.com/usage",
        }

    @tool(
        "Get the last N token-usage rows (one per messages.create round-trip). "
        "Use when you want to see recent turn-by-turn cost rather than aggregates."
    )
    async def recent(self, limit: int = 10) -> list[dict]:
        async with self.session_maker() as s:
            q = (
                select(AuditLog)
                .where(AuditLog.tool_name == "_turn")
                .order_by(AuditLog.timestamp.desc())
                .limit(limit)
            )
            rows = (await s.execute(q)).scalars().all()

        out = []
        for r in rows:
            cost = _cost_for(
                r.model or "",
                r.input_tokens or 0, r.output_tokens or 0,
                r.cache_read_tokens or 0, r.cache_creation_tokens or 0,
            )
            out.append({
                "timestamp": str(r.timestamp),
                "model": r.model or "",
                "input_tokens": r.input_tokens or 0,
                "output_tokens": r.output_tokens or 0,
                "cache_read_tokens": r.cache_read_tokens or 0,
                "duration_ms": r.duration_ms or 0,
                "estimated_cost_usd": round(cost, 4),
            })
        return out
