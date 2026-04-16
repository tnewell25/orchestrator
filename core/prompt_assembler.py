"""Multi-block system prompt with separate cache_control breakpoints.

Anthropic supports up to 4 cache breakpoints per request. We split the system
prompt into 4 layers, each with its own caching cadence:

  A — Identity / rules / output style       (forever-stable, cached)
  B — User profile (top accounts, role)     (weekly-stable, cached)
  C — Daily context (open commitments, etc) (daily-stable, cached)
  D — Per-turn focus + memories + plan hint (volatile, NOT cached)

Cache hit rate goes from ~10% (single-block prompt) to >70% because A+B+C
prefixes match across consecutive turns. Tools block also stays cached.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Iterable

from sqlalchemy import select

from .constants import EntityType
from .graph import EntityRef, Subgraph
from .planner import Intent, Plan


# ----------------------------------------------------------------------
# Block A — identity and rules. NEVER changes per-conversation.
# ----------------------------------------------------------------------

BLOCK_A_TEMPLATE = """You are {agent_name}, the user's AI chief of staff.

The user is a senior sales engineer selling industrial/enterprise solutions to
large firms (Bosch, Honeywell, GE, Rockwell, Siemens, Emerson, etc). Sales
cycles are long (9-18 months). Every ball dropped loses a 6-7 figure deal.

Your mission: capture everything the user tells you, keep their pipeline and
commitments organized, and act proactively so they never forget a detail.

CORE RULES
- Be concise — responses are read on a phone between meetings.
- Lead with the result ("Logged meeting, reminder set for Fri 2pm") not process.
- For multi-step requests, do ALL steps — never ask permission mid-task.
- Use tools freely. Every fact the user mentions should be filed somewhere.
- Prefer tool-search if you need a capability not currently loaded.
- For complex queries, use graph-context_for to pull a full subgraph in ONE call
  instead of chaining find + get + list calls.

WHAT TO CAPTURE (always)
- Meetings → meeting-log (summary, attendees, decisions, transcript)
- Commitments the user makes → task-create with due_date
- Commitments the OTHER side makes → task-create, source='meeting'
- Personal details about contacts (kids, hobbies, hometown, recent promotion) →
  contact-update personal_notes. Gold for relationship building.
- Competitor mentions → deal-update competitors
- MEDDIC signals → deal-set_meddic_field / deal-set_stakeholders
- Bid/RFP with deadline → bid-create (auto-schedules T-7d/T-3d/T-1d)

PROACTIVE BEHAVIOR
- Meeting scheduled → offer pre-meeting brief reminder.
- Commitment made → reminder a day or two before due date.
- Surface MEDDIC gaps after deal-get_context: "Note: economic buyer unknown."
- Before a meeting, surface personal_notes on attendees as conversation ammo.

REMINDERS
- Natural-language times via reminder-set ("in 2 hours", "tomorrow 9am").
- For meetings, prefer reminder-set_pre_meeting (auto-generates brief).
- Calendar events with known contacts auto-get pre-meeting briefs — don't duplicate.

INBOX
- Email questions → emailtriage-rank_unread (importance-scored).
- After gmail-send / gmail-create_draft → also call emailtriage-track_sent.

DEAL INTELLIGENCE
- "How's Bosch?" → graph-context_for(Deal, ...) AND stakeholder-coverage AND dealhealth-score.
- Stalled deals + overdue commitments are pinged proactively — no need to ask.

NEVER
- Invent data. If unknown, say so or look it up via research-search.
- Ask permission to log something the user already told you.
- Output lists with more than 10 items on mobile — summarize and offer drill-down."""


# ----------------------------------------------------------------------
# Mode-specific hints (Block D variants based on planner intent)
# ----------------------------------------------------------------------

MODE_HINTS = {
    Intent.PREP: (
        "MODE: pre-meeting prep — emphasize relationship intel (personal_notes), "
        "stakeholder roles, last meeting summary, open commitments, and one "
        "tactical opening question."
    ),
    Intent.STRATEGY: (
        "MODE: strategic synthesis — surface MEDDIC gaps, competitor positioning, "
        "win/loss patterns from similar deals, and concrete next steps. Use "
        "extended thinking if the picture is messy."
    ),
    Intent.QUERY: (
        "MODE: status query — give: stage, value, next step, last meeting, open "
        "actions, MEDDIC gaps. Keep to 6-10 bullets."
    ),
    Intent.RESEARCH: (
        "MODE: external research — synthesize raw search results into a tight "
        "5-10 bullet brief. Don't paste raw URLs or boilerplate."
    ),
    Intent.CRUD: (
        "MODE: capture — file the data, then a one-line confirmation. No prose."
    ),
}


@dataclass
class AssembledPrompt:
    """Renders to a list of content blocks with cache_control markers, ready
    for the Anthropic SDK system= parameter."""
    block_a: str
    block_b: str = ""
    block_c: str = ""
    block_d: str = ""

    def to_anthropic_blocks(self) -> list[dict]:
        out = [{"type": "text", "text": self.block_a, "cache_control": {"type": "ephemeral"}}]
        if self.block_b:
            out.append({"type": "text", "text": self.block_b, "cache_control": {"type": "ephemeral"}})
        if self.block_c:
            out.append({"type": "text", "text": self.block_c, "cache_control": {"type": "ephemeral"}})
        if self.block_d:
            # Block D is volatile — no cache_control marker.
            out.append({"type": "text", "text": self.block_d})
        return out


class PromptAssembler:
    """Builds an AssembledPrompt for one agent turn."""

    def __init__(self, agent_name: str = "Orchestrator"):
        self.agent_name = agent_name
        self._block_a_cached = BLOCK_A_TEMPLATE.format(agent_name=agent_name)

    def assemble(
        self,
        facts: list[dict],
        memories: list[dict],
        plan: Plan,
        focus_subgraph: Subgraph | None = None,
        daily_context_lines: list[str] | None = None,
        session_brief: str = "",
    ) -> AssembledPrompt:
        return AssembledPrompt(
            block_a=self._block_a_cached,
            block_b=self._render_block_b(facts),
            block_c=self._render_block_c(daily_context_lines or []),
            block_d=self._render_block_d(plan, memories, focus_subgraph, session_brief),
        )

    def _render_block_b(self, facts: list[dict]) -> str:
        """User profile — read from facts table. Stable for ~a week."""
        if not facts:
            return ""
        lines = ["USER PROFILE (stable facts):"]
        # Group facts by category for readability
        by_category: dict[str, list[str]] = {}
        for f in facts:
            cat = f.get("category", "general")
            key = f.get("key", "")
            value = f.get("value", "")
            if not key or not value:
                continue
            by_category.setdefault(cat, []).append(f"  - {key}: {value}")
        for cat in sorted(by_category):
            lines.append(f"[{cat}]")
            lines.extend(by_category[cat])
        return "\n".join(lines)

    def _render_block_c(self, lines: list[str]) -> str:
        """Daily context — refreshed at most ~once per day per user."""
        if not lines:
            return ""
        return "TODAY:\n" + "\n".join(f"- {ln}" for ln in lines)

    def _render_block_d(
        self,
        plan: Plan,
        memories: list[dict],
        focus_subgraph: Subgraph | None,
        session_brief: str = "",
    ) -> str:
        """Per-turn volatile context — focus, memories, mode hint, and the
        compacted-history brief (when older turns have been rolled up)."""
        parts: list[str] = []

        if session_brief:
            parts.append("EARLIER CONVERSATION SUMMARY:\n" + session_brief)

        mode_hint = MODE_HINTS.get(plan.intent, "")
        if mode_hint:
            parts.append(mode_hint)

        if plan.focus is not None:
            parts.append(f"FOCUS: {plan.focus}")
            if plan.rationale:
                parts.append(f"WHY: {plan.rationale}")

        if focus_subgraph is not None and focus_subgraph.edges:
            edge_lines = []
            for e in focus_subgraph.edges[:20]:
                edge_lines.append(
                    f"  {e.from_ref} --[{e.kind}]--> {e.to_ref} "
                    f"(reinforced × {e.reinforcement_count})"
                )
            parts.append("FOCUS SUBGRAPH:\n" + "\n".join(edge_lines))

        if memories:
            mem_lines = [
                f"  - [{m.get('source', 'mem')}] {m['content']}"
                for m in memories[:8]
            ]
            parts.append("RELEVANT MEMORIES:\n" + "\n".join(mem_lines))

        if plan.suggested_tools:
            parts.append("PLANNER SUGGESTED: " + ", ".join(plan.suggested_tools))

        return "\n\n".join(parts)


# ----------------------------------------------------------------------
# Daily context builder — DB-backed, keyed by date so caching works.
# ----------------------------------------------------------------------

async def build_daily_context_lines(session_maker) -> list[str]:
    """Fetch the bullets that go in Block C: today's commitments + open MEDDIC gaps.

    Cheap query — runs once per day per cache flush, then served from cache.
    Returns plain strings; the assembler wraps them with TODAY: header.
    """
    from ..db.models import ActionItem, Deal, Reminder

    today = datetime.now(timezone.utc).date()
    cutoff_meeting_window = datetime.now(timezone.utc) + timedelta(hours=24)
    lines: list[str] = []

    async with session_maker() as session:
        # Open action items due today or overdue
        rows = (
            await session.execute(
                select(ActionItem)
                .where(
                    ActionItem.status == "open",
                    ActionItem.due_date.is_not(None),
                    ActionItem.due_date <= today,
                )
                .order_by(ActionItem.due_date)
                .limit(8)
            )
        ).scalars().all()
        for r in rows:
            lines.append(f"OPEN: {r.description} (due {r.due_date})")

        # Pending reminders firing soon
        rems = (
            await session.execute(
                select(Reminder)
                .where(
                    Reminder.status == "pending",
                    Reminder.trigger_at <= cutoff_meeting_window,
                )
                .order_by(Reminder.trigger_at)
                .limit(5)
            )
        ).scalars().all()
        for r in rems:
            lines.append(f"REMINDER: {r.message[:80]} ({r.trigger_at:%a %H:%M})")

        # Deals missing critical MEDDIC fields
        deals = (
            await session.execute(
                select(Deal)
                .where(
                    Deal.stage.in_(["qualified", "proposal", "negotiation"]),
                )
                .limit(5)
            )
        ).scalars().all()
        for d in deals:
            gaps = []
            if not d.economic_buyer_id:
                gaps.append("economic buyer")
            if not d.champion_id:
                gaps.append("champion")
            if not d.metrics:
                gaps.append("metrics")
            if gaps:
                lines.append(f"MEDDIC gap [{d.name}]: missing {', '.join(gaps)}")

    return lines


__all__ = [
    "AssembledPrompt",
    "PromptAssembler",
    "build_daily_context_lines",
    "MODE_HINTS",
]
