"""Anthropic tool-use agent loop with Claude Max OAuth support."""
import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING

import anthropic

from .constants import EntityType
from .graph import EntityRef
from .planner import Intent, Plan
from .tool_registry import DEFAULT_ESSENTIALS, ToolRegistry, tool_search_schema

if TYPE_CHECKING:
    from .planner import Planner

logger = logging.getLogger(__name__)

# Spread out tool-loop iterations so we don't burst against Claude Max OAuth's
# per-account concurrency budget. 200ms between rounds is invisible to the user
# but lets the server-side queue breathe.
ITER_DELAY_S = 0.25
# How many times we catch a RateLimitError ourselves (on TOP of SDK's retries).
RATE_LIMIT_RETRIES = 3

STATIC_SYSTEM_PROMPT = """You are {agent_name}, the user's AI chief of staff.

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

WHAT TO CAPTURE (always)
- Meetings → meeting-log (summary, attendees, decisions, transcript)
- Commitments the user makes ("I'll send X by Friday") → task-create with due_date
- Commitments the OTHER side makes → task-create, source='meeting', note who owes what
- Personal details about contacts (kids, hobbies, hometown, recent promotion) →
  contact-update personal_notes. These are gold for relationship building.
- Competitor mentions → deal-update competitors
- MEDDIC signals:
  * Metrics the buyer cares about ("reduce downtime 15%") → deal-set_meddic_field metrics
  * Economic buyer identified → deal-set_stakeholders economic_buyer_id
  * Champion identified → deal-set_stakeholders champion_id
  * Pain being solved → deal-set_meddic_field pain
  * Decision criteria → deal-set_meddic_field decision_criteria
  * Decision/paper process (security review, procurement steps) → deal-set_meddic_field
- Bid/RFP mentioned with a deadline → bid-create (auto-schedules T-7d/T-3d/T-1d pings)

PROACTIVE BEHAVIOR
- When the user schedules a meeting, offer to set a pre-meeting brief reminder
  (reminder-set_pre_meeting, default 30 min before). Do it without asking if
  the meeting is with a known contact and there's context to surface.
- When the user commits to something, set a reminder a day or two before the
  due date (reminder-set, kind='commitment').
- On any deal-get_context response, check meddic.gaps — if any critical fields
  are missing, flag them in a single line: "Note: economic buyer and decision
  process still unknown. Worth asking Markus next call?"
- Before a meeting, surface personal_notes on the attendee — "Markus's son
  started at MIT this year" — as conversation ammo.

WHEN THE USER ASKS "WHAT'S GOING ON WITH X"
1. company-find or deal-find to locate it
2. deal-get_context for full picture
3. Respond with: stage, value, next step, last meeting summary, open actions,
   MEDDIC gaps. Keep to 6-10 bullets.

REMINDERS
- The user can say "remind me in 2 hours to call Markus" or "ping me tomorrow
  9am about the spec" — use reminder-set with natural-language time.
- For anything meeting-related, prefer reminder-set_pre_meeting so the brief
  auto-generates.
- Calendar events with known contacts already get auto-scheduled pre-meeting
  briefs via the background sync — don't duplicate.

INBOX
- When the user asks about email, call emailtriage-rank_unread to get
  importance-scored list, not gmail-list_unread raw.
- When the agent drafts outbound email via gmail-send or gmail-create_draft,
  ALSO call emailtriage-track_sent so the no-reply nudger can engage.

DEAL INTELLIGENCE
- For "how's Bosch going?" → deal-get_context (MEDDIC + meetings + actions +
  bids) AND stakeholder-coverage (the 5-role map) AND dealhealth-score (temp).
- Stalled deals and overdue commitments are pinged proactively by the
  ProactiveMonitor background service — the user gets reminders without asking.

RESEARCH
- For "research Honeywell Forge" or "tell me about Anja Weber" → research-
  company_deepdive / research-exec_bio / research-competitive_analysis. Synthesize
  results into a tight 5-10 bullet brief. Don't paste raw search results.

PROPOSALS
- "Draft a proposal for Bosch" → proposal-draft_proposal with deal_id. Returns
  markdown — paste it back for user to review and edit.
- "Save this as precedent" → proposal-save_precedent so future drafts pull it.

COMPETITORS
- When user mentions a competitor by name → competitor-find_battle_card first
  (semantic search for the situation) before synthesizing a response.
- After a deal closes → competitor-log_win_loss for pattern mining.

NEVER
- Invent data. If you don't know, say so or look it up via research-search.
- Ask permission to log something the user already told you.
- Output lists with more than 10 items on mobile — summarize and offer drill-down."""


def _build_dynamic_context(facts: list, memories: list, plan_hint: str = "") -> str:
    """Facts + memories + plan hint change per-request — kept separate so the
    static prompt (above) + tools can be cache_controlled for 10x TPM savings."""
    parts = []
    if plan_hint:
        parts.append("PLANNER HINT (suggested but not binding):\n" + plan_hint)
    if facts:
        parts.append("Known facts:\n" + "\n".join(
            f"- [{f['category']}] {f['key']}: {f['value']}" for f in facts
        ))
    if memories:
        parts.append("Relevant memories:\n" + "\n".join(
            f"- {m['content']}" for m in memories
        ))
    return "\n\n".join(parts) if parts else ""


class Agent:
    def __init__(
        self,
        memory,
        skills: list,
        settings,
        token_manager=None,
        audit_logger=None,
        max_iterations: int = 15,
        planner: "Planner | None" = None,
        entity_extractor=None,
        lazy_tools: bool = False,
        essentials: tuple[str, ...] = DEFAULT_ESSENTIALS,
    ):
        self.memory = memory
        self.skills = {s.name: s for s in skills}
        self.settings = settings
        self.token_manager = token_manager
        self.audit_logger = audit_logger
        self.max_iterations = max_iterations
        self.planner = planner
        self.entity_extractor = entity_extractor
        # When True, only essentials + tool-search are loaded by default;
        # the agent must call tool-search to discover others. Saves prompt tokens.
        self.lazy_tools = lazy_tools
        self.essentials = tuple(essentials)

        self._init_client()

        self.tools: list[dict] = []      # full catalog (eager mode uses all of these)
        self.tool_map: dict = {}         # name → skill (always full)
        self.registry = ToolRegistry(skills)  # for lazy lookup

        for skill in skills:
            for schema in skill.get_tools():
                self.tools.append(schema)
                self.tool_map[schema["name"]] = skill

    def _cache_controlled_tools(self, tools: list | None = None) -> list:
        """Attach cache_control to the last tool so the entire tools block gets
        cached. Subsequent identical requests cost ~10% of the normal input tokens.

        Pass `tools` explicitly when in lazy mode (different per call); falls
        back to self.tools for legacy callers."""
        src = tools if tools is not None else self.tools
        if not src:
            return src
        tools_out = [dict(t) for t in src]
        tools_out[-1] = {**tools_out[-1], "cache_control": {"type": "ephemeral"}}
        return tools_out

    def _initial_active_tools(self, plan: Plan | None = None) -> list[dict]:
        """Build the per-call tool list.

        Eager mode: full catalog (back-compat).
        Lazy mode: essentials + tool-search + planner-suggested tools, in that
                   order. The essentials prefix stays cache-stable across calls."""
        if not self.lazy_tools:
            return list(self.tools)

        loaded_names: set[str] = set()
        out: list[dict] = []

        # 1. Essentials — always-loaded prefix (cacheable).
        for name in self.essentials:
            schema = self.registry.get_schema(name)
            if schema and name not in loaded_names:
                out.append(schema)
                loaded_names.add(name)

        # 2. Meta-tool — tool-search itself.
        out.append(tool_search_schema())
        loaded_names.add("tool-search")

        # 3. Planner-suggested tools (cache-busting but worth it for accuracy).
        if plan and plan.suggested_tools:
            for name in plan.suggested_tools:
                if name in loaded_names:
                    continue
                schema = self.registry.get_schema(name)
                if schema:
                    out.append(schema)
                    loaded_names.add(name)

        return out

    def _build_system(self, static: str, dynamic: str) -> list:
        """System prompt as content-block list: static part is cached (marked
        with cache_control), dynamic per-call facts/memories append fresh.
        Cache hit = ~10% of normal input cost for the cached prefix + tools."""
        blocks = [
            {"type": "text", "text": static, "cache_control": {"type": "ephemeral"}}
        ]
        if dynamic:
            blocks.append({"type": "text", "text": dynamic})
        return blocks

    def _init_client(self):
        # max_retries=5 so the SDK's built-in exponential backoff has more room
        # (default is 2, which gives up in ~1.2s — way too eager for OAuth bursts).
        # timeout bumped so slow-network retries aren't truncated.
        common = {"max_retries": 5, "timeout": 120.0}
        if self.token_manager and self.token_manager.access_token:
            self.client = anthropic.AsyncAnthropic(
                auth_token=self.token_manager.access_token,
                default_headers={"anthropic-beta": "oauth-2025-04-20"},
                **common,
            )
        elif self.token_manager and self.token_manager.api_key:
            self.client = anthropic.AsyncAnthropic(
                api_key=self.token_manager.api_key, **common
            )
        else:
            self.client = anthropic.AsyncAnthropic(
                api_key=self.settings.anthropic_api_key, **common
            )

    def register_skill(self, skill):
        self.skills[skill.name] = skill
        for schema in skill.get_tools():
            self.tools.append(schema)
            self.tool_map[schema["name"]] = skill

    async def run(
        self,
        message: str,
        session_id: str,
        interface: str = "telegram",
        model: str | None = None,
    ) -> str:
        model = model or self.settings.default_model

        if self.token_manager:
            await self.token_manager.ensure_token()
            self._init_client()

        await self.memory.add_message(session_id, "user", message, interface)

        conversation = await self.memory.get_conversation(session_id, limit=30)
        facts = await self.memory.get_facts()

        # Planner pass — identify focus entity + intent so recall_hybrid gets
        # proximity boost and the system prompt knows what mode to surface.
        plan = await self._run_planner(message, conversation)
        focus_ref = plan.focus

        memories = await self.memory.recall(message, limit=5, focus_ref=focus_ref)

        static_system = STATIC_SYSTEM_PROMPT.format(
            agent_name=self.settings.agent_name
        )
        dynamic_system = _build_dynamic_context(
            facts, memories, plan_hint=plan.to_prompt_hint() if plan.intent != Intent.AMBIGUOUS else "",
        )

        messages = []
        for msg in conversation:
            if msg["role"] in ("user", "assistant"):
                messages.append({"role": msg["role"], "content": msg["content"]})

        # Per-call active tools list. In lazy mode, grows during the loop as
        # the agent calls tool-search to discover capabilities.
        active_tools = self._initial_active_tools(plan)
        loaded_tool_names = {t["name"] for t in active_tools}

        recovered_this_run = False
        rl_retries_left = RATE_LIMIT_RETRIES

        for iteration in range(self.max_iterations):
            # Small cadence delay between iterations to avoid burst-triggering 429.
            # Skip on iteration 0 (don't add latency to user's first response).
            if iteration > 0:
                await asyncio.sleep(ITER_DELAY_S)

            response = None
            while response is None:
                try:
                    response = await self.client.messages.create(
                        model=model,
                        max_tokens=4096,
                        system=self._build_system(static_system, dynamic_system),
                        tools=self._cache_controlled_tools(active_tools) if active_tools else anthropic.NOT_GIVEN,
                        messages=messages,
                    )
                except anthropic.AuthenticationError:
                    if not recovered_this_run and self.token_manager:
                        recovered_this_run = True
                        logger.warning("Auth error — running token recovery")
                        recovered = await self.token_manager.handle_auth_error()
                        if recovered:
                            self._init_client()
                            continue  # retry same iteration with fresh creds
                    raise
                except anthropic.RateLimitError as e:
                    # SDK already exhausted its exponential retries. Read the
                    # server-provided Retry-After hint if present, else back off
                    # aggressively (server-side burst caps want ~10-30s breathing room).
                    retry_after = 15.0
                    try:
                        hdr = e.response.headers.get("retry-after") if getattr(e, "response", None) else None
                        if hdr:
                            retry_after = min(float(hdr), 60.0)
                    except (ValueError, AttributeError):
                        pass
                    if rl_retries_left <= 0:
                        logger.error("Rate limit exhausted after %d loop-level retries", RATE_LIMIT_RETRIES)
                        raise
                    rl_retries_left -= 1
                    logger.warning(
                        "Rate-limited. Waiting %.1fs before retry (%d loop retries left)",
                        retry_after, rl_retries_left,
                    )
                    await asyncio.sleep(retry_after)
                    continue  # retry same call

            if not any(b.type == "tool_use" for b in response.content):
                final_text = "".join(
                    b.text for b in response.content if b.type == "text"
                )
                await self.memory.add_message(
                    session_id, "assistant", final_text, interface
                )
                return final_text

            assistant_content = []
            tool_use_blocks = []

            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )
                    tool_use_blocks.append(block)

            # Parallel execution — Anthropic emits multiple tool_use blocks per
            # turn for genuinely independent reads (e.g. deal-get + stakeholder-list).
            # gather() also makes single-tool turns no slower.
            tool_results = []
            if tool_use_blocks:
                # Intercept tool-search calls: handled inline (mutates active_tools)
                # rather than dispatched through skill registry.
                exec_blocks = []
                for b in tool_use_blocks:
                    if b.name == "tool-search":
                        result_text = self._handle_tool_search(b.input, active_tools, loaded_tool_names)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": b.id,
                            "content": result_text,
                        })
                    else:
                        exec_blocks.append(b)

                if exec_blocks:
                    results = await asyncio.gather(
                        *(
                            self._execute_tool(b.name, b.input, session_id=session_id)
                            for b in exec_blocks
                        ),
                        return_exceptions=True,
                    )
                    for b, r in zip(exec_blocks, results):
                        content = (
                            f"Error executing {b.name}: {r}"
                            if isinstance(r, Exception) else r
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": b.id,
                            "content": content,
                        })

                # Re-order tool_results to match the assistant's tool_use order
                # (Anthropic SDK is strict — out-of-order tool_use_ids fail).
                order = {b.id: i for i, b in enumerate(tool_use_blocks)}
                tool_results.sort(key=lambda r: order.get(r["tool_use_id"], 0))

            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})

        fallback = "Hit the tool loop limit. Tell me to continue if you want me to keep going."
        await self.memory.add_message(session_id, "assistant", fallback, interface)
        return fallback

    async def _run_planner(self, message: str, conversation: list[dict]) -> Plan:
        """Optional pre-pass — Haiku call to identify focus + intent.

        Returns Plan() (empty) if no planner attached or if the call fails.
        Never blocks the main loop on its outcome."""
        if not self.planner:
            return Plan()
        try:
            recent = "\n".join(
                f"{m['role']}: {m['content'][:140]}" for m in conversation[-4:]
            )
            entities_summary = ""
            if self.entity_extractor is not None:
                entities_summary = self.entity_extractor.index.known_names_summary(limit=80)
            tool_names = [t["name"] for t in self.tools[:60]]  # cap to avoid prompt bloat
            return await self.planner.plan(
                user_message=message,
                recent_summary=recent,
                known_entities_summary=entities_summary,
                available_tools=tool_names,
                entity_resolver=self._resolve_entity_by_name,
            )
        except Exception as e:
            logger.warning("Planner failed, continuing without plan: %s", e)
            return Plan()

    def _handle_tool_search(
        self, tool_input: dict, active_tools: list[dict], loaded: set[str]
    ) -> str:
        """Run the registry search and append discovered schemas to active_tools.

        Returns a JSON string the agent reads as the tool result. The new tools
        are immediately callable on the next iteration of the same .run() call.
        """
        query = (tool_input.get("query") or "").strip()
        limit = int(tool_input.get("limit") or 5)
        results = self.registry.search(query, limit=limit)

        added = []
        for schema in results:
            name = schema["name"]
            if name in loaded:
                continue
            active_tools.append(schema)
            loaded.add(name)
            added.append(name)

        return json.dumps({
            "matched": [{"name": s["name"], "description": s["description"]} for s in results],
            "newly_loaded": added,
            "active_count": len(active_tools),
        })

    async def _resolve_entity_by_name(self, name: str, type_str: str) -> EntityRef | None:
        """Map a planner-produced (name, type) to a concrete EntityRef.

        Uses the entity_extractor's substring index for instant lookup.
        Returns None if no confident match exists."""
        if self.entity_extractor is None:
            return None
        # The index is keyed by lowercase name → list[EntityRef]
        candidates = self.entity_extractor.index._name_to_refs.get(name.lower(), [])
        for ref in candidates:
            if ref.type == type_str:
                return ref
        # Fallback — first-name match (lookup_substring already adds first names)
        if " " in name:
            first = name.split(" ", 1)[0].lower()
            for ref in self.entity_extractor.index._name_to_refs.get(first, []):
                if ref.type == type_str:
                    return ref
        return None

    async def _execute_tool(
        self, tool_name: str, tool_input: dict, session_id: str = ""
    ) -> str:
        skill = self.tool_map.get(tool_name)
        if not skill:
            return f"Error: unknown tool '{tool_name}'"

        method = skill.get_tool_method(tool_name)
        if not method:
            return f"Error: method not found for '{tool_name}'"

        t0 = time.perf_counter()
        try:
            result = await method(**tool_input)
            if isinstance(result, (dict, list)):
                result_str = json.dumps(result, default=str)
            else:
                result_str = str(result)

            if self.audit_logger:
                duration_ms = int((time.perf_counter() - t0) * 1000)
                await self.audit_logger.log(
                    tool_name=tool_name,
                    args=tool_input,
                    result_status="ok",
                    result_summary=result_str[:200],
                    session_id=session_id,
                    duration_ms=duration_ms,
                )

            return result_str
        except Exception as e:
            logger.error(f"Tool error [{tool_name}]: {e}", exc_info=True)
            error_str = f"Error executing {tool_name}: {e}"

            if self.audit_logger:
                duration_ms = int((time.perf_counter() - t0) * 1000)
                await self.audit_logger.log(
                    tool_name=tool_name,
                    args=tool_input,
                    result_status="error",
                    result_summary=str(e)[:200],
                    session_id=session_id,
                    duration_ms=duration_ms,
                )

            return error_str
