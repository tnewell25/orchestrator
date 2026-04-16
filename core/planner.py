"""Planner — pre-pass that classifies intent + identifies focus entity.

Why: today's agent goes straight from user message to tool-loop. With a 500ms
Haiku call up front, we can:
  1. Resolve the focus entity (Deal/Contact/Company) so memory.recall_hybrid
     gets a proximity boost on every memory connected to it.
  2. Tag intent so the system prompt assembler (W5) shows the right mode hints.
  3. Suggest a starting tool sketch so the main agent stops at the answer faster.

The planner's output is a HINT, not a contract. The agent can override.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Iterable

from .constants import EntityType
from .graph import EntityRef

logger = logging.getLogger(__name__)


# ---- Intent taxonomy ---------------------------------------------------

class Intent:
    CRUD = "CRUD"               # log/update/create — straightforward writes
    PREP = "PREP"               # pre-meeting brief, who-is-this person prep
    STRATEGY = "STRATEGY"       # game-plan, win-loss synthesis, MEDDIC gaps
    RESEARCH = "RESEARCH"       # external lookup (company, exec, competitor)
    QUERY = "QUERY"             # status/lookup ("what's going on with Bosch")
    AMBIGUOUS = "AMBIGUOUS"     # unclear — let the main agent ask clarifying

    ALL = (CRUD, PREP, STRATEGY, RESEARCH, QUERY, AMBIGUOUS)


@dataclass
class Plan:
    intent: str = Intent.AMBIGUOUS
    focus: EntityRef | None = None
    rationale: str = ""              # one-sentence thought
    suggested_tools: list[str] = field(default_factory=list)
    parallel_groups: list[list[str]] = field(default_factory=list)
    use_thinking: bool = False       # whether to enable extended thinking on the main pass

    def to_prompt_hint(self) -> str:
        """Compact text block injected into the agent's system prompt."""
        lines = [f"INTENT: {self.intent}"]
        if self.focus:
            lines.append(f"FOCUS: {self.focus}")
        if self.rationale:
            lines.append(f"WHY: {self.rationale}")
        if self.suggested_tools:
            lines.append("SUGGESTED TOOLS: " + ", ".join(self.suggested_tools))
        if self.parallel_groups and any(len(g) > 1 for g in self.parallel_groups):
            lines.append("PARALLELIZABLE: " + " | ".join(
                f"[{', '.join(g)}]" for g in self.parallel_groups if len(g) > 1
            ))
        return "\n".join(lines)


# ---- Planner ----------------------------------------------------------

_PLANNER_SYSTEM = """You are the planning pre-pass for a sales-engineer's AI chief of staff.

Read the user message + recent context, then emit ONE compact JSON object that
classifies what they want and identifies the focus entity (deal/contact/company)
if there is one.

Output exactly this schema (no prose, no markdown fences):
{
  "intent": "CRUD" | "PREP" | "STRATEGY" | "RESEARCH" | "QUERY" | "AMBIGUOUS",
  "focus": {"type": "Deal" | "Contact" | "Company", "name": "<canonical>"} | null,
  "rationale": "<one sentence>",
  "suggested_tools": ["tool-1", "tool-2"],
  "parallel_groups": [["tool-1", "tool-2"], ["tool-3"]],
  "use_thinking": false
}

Rules:
- CRUD: user is logging/updating/creating something concrete ("Met with Anja today")
- PREP: getting ready for an upcoming call/meeting ("prep me for tomorrow")
- STRATEGY: synthesis, game-plan, why-are-we-losing ("how do we win Bosch")
- RESEARCH: external lookup ("tell me about Honeywell Forge")
- QUERY: status check ("what's going on with the Bosch deal")
- AMBIGUOUS: unclear — main agent should ask
- focus is the PRIMARY entity. If user mentions "Anja from Bosch", focus is the
  more specific one — usually the deal or contact, not the company.
- parallel_groups: tools in the same inner list can run concurrently. Only
  group reads (find/get/list); never group writes.
- use_thinking: true ONLY for STRATEGY or hard ambiguous cases."""


class Planner:
    def __init__(
        self,
        anthropic_client,
        fast_model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 600,
    ):
        self.client = anthropic_client
        self.fast_model = fast_model
        self.max_tokens = max_tokens

    async def plan(
        self,
        user_message: str,
        recent_summary: str = "",
        known_entities_summary: str = "",
        available_tools: Iterable[str] = (),
        entity_resolver=None,
    ) -> Plan:
        """Build a Plan from the user message + light context.

        entity_resolver: optional callable(name, type) → EntityRef|None to map
        the planner's "focus.name" back to an actual DB id. If None or returns
        None, the plan still has the type+name but no concrete ref.
        """
        if not self.client:
            return Plan()  # Empty plan — main agent runs as before.

        prompt = _build_planner_prompt(
            user_message, recent_summary, known_entities_summary,
            list(available_tools),
        )
        try:
            resp = await self.client.messages.create(
                model=self.fast_model,
                max_tokens=self.max_tokens,
                system=_PLANNER_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = "".join(b.text for b in resp.content if b.type == "text").strip()
        except Exception as e:
            logger.warning("Planner LLM call failed: %s", e)
            return Plan()

        return await _parse_plan(raw, entity_resolver)


# ---- Helpers ----------------------------------------------------------

def _build_planner_prompt(
    user_message: str,
    recent_summary: str,
    known_entities_summary: str,
    available_tools: list[str],
) -> str:
    parts = []
    if recent_summary:
        parts.append(f"Recent context:\n{recent_summary}")
    if known_entities_summary:
        parts.append(f"Known entities (for focus resolution):\n{known_entities_summary}")
    if available_tools:
        parts.append(f"Available tools:\n{', '.join(available_tools)}")
    parts.append(f"User message:\n{user_message}")
    parts.append("Return the JSON plan.")
    return "\n\n".join(parts)


async def _parse_plan(raw: str, entity_resolver) -> Plan:
    if not raw:
        return Plan()
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if "\n" in cleaned:
            cleaned = cleaned.split("\n", 1)[1]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            return Plan()
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return Plan()

    if not isinstance(data, dict):
        return Plan()

    intent = data.get("intent") or Intent.AMBIGUOUS
    if intent not in Intent.ALL:
        intent = Intent.AMBIGUOUS

    focus_ref = None
    focus_obj = data.get("focus")
    if isinstance(focus_obj, dict) and focus_obj.get("type") and focus_obj.get("name"):
        ftype = focus_obj["type"]
        fname = (focus_obj.get("name") or "").strip()
        if ftype in EntityType.ALL and fname and entity_resolver is not None:
            try:
                resolved = await entity_resolver(fname, ftype)
                if isinstance(resolved, EntityRef):
                    focus_ref = resolved
            except Exception as e:
                logger.warning("Focus resolver failed for %s/%s: %s", ftype, fname, e)

    suggested = data.get("suggested_tools") or []
    if not isinstance(suggested, list):
        suggested = []
    suggested = [s for s in suggested if isinstance(s, str)]

    parallel_groups_raw = data.get("parallel_groups") or []
    parallel_groups: list[list[str]] = []
    if isinstance(parallel_groups_raw, list):
        for g in parallel_groups_raw:
            if isinstance(g, list):
                parallel_groups.append([s for s in g if isinstance(s, str)])

    return Plan(
        intent=intent,
        focus=focus_ref,
        rationale=str(data.get("rationale") or "").strip(),
        suggested_tools=suggested,
        parallel_groups=parallel_groups,
        use_thinking=bool(data.get("use_thinking")),
    )


__all__ = ["Plan", "Planner", "Intent"]
