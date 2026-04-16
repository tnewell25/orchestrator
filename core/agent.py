"""Anthropic tool-use agent loop with Claude Max OAuth support."""
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncIterator

import anthropic

from .action_gate import ActionGate
from .compactor import Compactor
from .constants import EntityType
from .graph import EntityRef
from .planner import Intent, Plan
from .prompt_assembler import PromptAssembler, build_daily_context_lines
from .skill_base import Safety
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


@dataclass
class StreamEvent:
    """Event yielded by Agent.run_stream — caller picks which to render."""
    type: str           # 'text_delta' | 'tool_call' | 'tool_result' | 'complete' | 'error' | 'cancelled'
    text: str = ""      # populated for text_delta and complete
    tool_name: str = ""
    tool_input: dict | None = None
    tool_result: str = ""
    error: str = ""


class Cancelled(Exception):
    """Raised inside the stream loop when a caller-supplied event is set."""


class Agent:
    def __init__(
        self,
        memory,
        skills: list,
        settings,
        token_manager=None,
        audit_logger=None,
        max_iterations: int | None = None,
        planner: "Planner | None" = None,
        entity_extractor=None,
        lazy_tools: bool = False,
        essentials: tuple[str, ...] = DEFAULT_ESSENTIALS,
        compactor: "Compactor | None" = None,
        action_gate: "ActionGate | None" = None,
    ):
        self.memory = memory
        self.skills = {s.name: s for s in skills}
        self.settings = settings
        self.token_manager = token_manager
        self.audit_logger = audit_logger
        # Iteration cap — pulled from settings (8) unless caller overrides.
        # Most turns complete in 3-5 iters; 8 catches long chains without
        # letting runaways burn budget.
        self.max_iterations = (
            max_iterations if max_iterations is not None
            else int(getattr(settings, "max_agent_iterations", 8) or 8)
        )
        # How many recent messages to replay per turn. Smaller window = lower
        # cost + better cache stability; compaction covers older context.
        self.conversation_window = int(
            getattr(settings, "conversation_window_limit", 15) or 15
        )
        self.planner = planner
        self.entity_extractor = entity_extractor
        # When True, only essentials + tool-search are loaded by default;
        # the agent must call tool-search to discover others. Saves prompt tokens.
        self.lazy_tools = lazy_tools
        self.essentials = tuple(essentials)
        self.prompt_assembler = PromptAssembler(agent_name=settings.agent_name)
        # Daily context cache — recomputed at most once per (date, session_maker).
        self._daily_cache: dict[str, tuple] = {}
        # Optional — when present, run() fires it asynchronously after each turn
        # so the next turn sees a smaller context window.
        self.compactor = compactor
        # When present, approve_external tools are queued for user approval
        # instead of executing immediately. The agent gets a "queued" result.
        self.action_gate = action_gate

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

    def _select_model_and_thinking(self, plan: Plan) -> tuple[str, dict | None]:
        """Pick the cheapest model that still does the job + decide if extended
        thinking is worth the token tax for this intent.

        - CRUD/QUERY → fast_model (Haiku). ~60% cheaper, quality indistinguishable.
        - PREP/RESEARCH → default_model (Sonnet). Needs synthesis quality.
        - STRATEGY → default_model + extended thinking. Worth the tax for hard calls.
        - AMBIGUOUS → default_model, no thinking. Let the agent disambiguate.

        Override per-call via settings.fast_model_intents (comma-separated).
        """
        fast_intents = set(
            s.strip() for s in (getattr(self.settings, "fast_model_intents", "CRUD,QUERY") or "").split(",")
            if s.strip()
        )
        model = self.settings.default_model
        thinking = None

        if plan.intent in fast_intents:
            model = self.settings.fast_model
        elif plan.intent == Intent.STRATEGY or plan.use_thinking:
            budget = int(getattr(self.settings, "thinking_budget_tokens", 5000) or 0)
            if budget > 0:
                thinking = {"type": "enabled", "budget_tokens": budget}

        return model, thinking

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
        # Model override wins; otherwise we pick based on planner intent below
        # so cheap reads run on Haiku and strategy calls get thinking.
        override_model = model

        if self.token_manager:
            await self.token_manager.ensure_token()
            self._init_client()

        await self.memory.add_message(session_id, "user", message, interface)

        conversation = await self.memory.get_conversation(session_id, limit=self.conversation_window)
        facts = await self.memory.get_facts()

        # Planner pass — identify focus entity + intent so recall_hybrid gets
        # proximity boost and the system prompt knows what mode to surface.
        plan = await self._run_planner(message, conversation)
        focus_ref = plan.focus

        # Intent-aware model selection — huge cost lever on CRUD/QUERY traffic.
        intent_model, thinking = self._select_model_and_thinking(plan)
        model = override_model or intent_model

        memories = await self.memory.recall(message, limit=5, focus_ref=focus_ref)

        # Compaction summary — block D includes this so the agent retains context
        # that's been rolled out of the active conversation window.
        session_brief = ""
        try:
            session_brief = await self.memory.get_latest_session_brief(session_id)
        except Exception as e:
            logger.warning("Session brief fetch failed: %s", e)

        # Pull a focus subgraph so block D shows the relationship picture.
        focus_subgraph = None
        if focus_ref is not None and getattr(self.memory, "_graph", None) is not None:
            try:
                focus_subgraph = await self.memory._graph.subgraph(
                    focus_ref, max_depth=2, max_nodes=20,
                )
            except Exception as e:
                logger.warning("Focus subgraph fetch failed: %s", e)

        daily_lines = await self._get_daily_context()

        assembled = self.prompt_assembler.assemble(
            facts=facts,
            memories=memories,
            plan=plan,
            focus_subgraph=focus_subgraph,
            daily_context_lines=daily_lines,
            session_brief=session_brief,
        )
        system_blocks = assembled.to_anthropic_blocks()

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

        # Rewind tracking — detect when the agent re-calls a failing tool with
        # the same args. Instead of appending yet another error to history, we
        # DROP the prior failed exchange and inject a corrective note, so the
        # next iteration doesn't see the failure in context. Saves 200-800
        # tokens per loop and breaks infinite-retry patterns faster.
        last_error_sigs: set[str] = set()

        for iteration in range(self.max_iterations):
            # Small cadence delay between iterations to avoid burst-triggering 429.
            # Skip on iteration 0 (don't add latency to user's first response).
            if iteration > 0:
                await asyncio.sleep(ITER_DELAY_S)

            response = None
            call_start = time.perf_counter()
            while response is None:
                try:
                    kwargs = dict(
                        model=model,
                        max_tokens=4096,
                        system=system_blocks,
                        tools=self._cache_controlled_tools(active_tools) if active_tools else anthropic.NOT_GIVEN,
                        messages=messages,
                    )
                    if thinking is not None:
                        kwargs["thinking"] = thinking
                    response = await self.client.messages.create(**kwargs)
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

            # Token accounting — one audit row per messages.create round-trip.
            # Fire-and-forget; audit failures can't block the agent loop.
            if self.audit_logger and getattr(response, "usage", None) is not None:
                try:
                    await self.audit_logger.log_usage(
                        session_id=session_id, model=model,
                        usage=response.usage,
                        duration_ms=int((time.perf_counter() - call_start) * 1000),
                        iteration=iteration,
                    )
                except Exception:
                    pass

            if not any(b.type == "tool_use" for b in response.content):
                final_text = "".join(
                    b.text for b in response.content if b.type == "text"
                )
                await self.memory.add_message(
                    session_id, "assistant", final_text, interface
                )
                self._schedule_compaction(session_id, plan=plan)
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

            # Rewind check — did the agent re-call a failing tool with the
            # same args that failed last iteration?
            error_sigs_this_turn = self._error_signatures(tool_use_blocks, tool_results)
            repeated = error_sigs_this_turn & last_error_sigs
            last_error_sigs = error_sigs_this_turn

            if repeated and len(messages) >= 2:
                # Drop the PREVIOUS failed (assistant, tool_result) pair so the
                # API doesn't re-see the error. Replace this turn's pair with a
                # corrective user note.
                messages.pop()  # previous tool_result (user role)
                messages.pop()  # previous assistant turn
                lesson = (
                    "Previous attempts failed with the same arguments: "
                    + ", ".join(sorted(repeated))
                    + ". Don't retry them verbatim — try a different tool, "
                      "different arguments, or ask the user to clarify."
                )
                messages.append({"role": "user", "content": lesson})
                logger.info("Rewound %d repeated tool failures: %s", len(repeated), repeated)
                continue

            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})

        fallback = "Hit the tool loop limit. Tell me to continue if you want me to keep going."
        await self.memory.add_message(session_id, "assistant", fallback, interface)
        return fallback

    async def run_stream(
        self,
        message: str,
        session_id: str,
        interface: str = "telegram",
        model: str | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming variant of run() — yields incremental events.

        Event types:
          text_delta: a chunk of the user-visible response (most useful for UI)
          tool_call:  agent decided to invoke a tool
          tool_result: tool returned (compact summary)
          complete:   final response text (also accumulated from deltas)
          cancelled:  cancel_event was set
          error:      unrecoverable error mid-stream

        Pass cancel_event to allow interruption — if set between iterations or
        between stream events, the agent stops cleanly without a partial commit.
        """
        override_model = model
        if self.token_manager:
            await self.token_manager.ensure_token()
            self._init_client()

        await self.memory.add_message(session_id, "user", message, interface)

        conversation = await self.memory.get_conversation(session_id, limit=self.conversation_window)
        facts = await self.memory.get_facts()
        plan = await self._run_planner(message, conversation)
        focus_ref = plan.focus

        # Intent-aware model selection (same lever as run()).
        intent_model, thinking = self._select_model_and_thinking(plan)
        model = override_model or intent_model

        memories = await self.memory.recall(message, limit=5, focus_ref=focus_ref)

        session_brief = ""
        try:
            session_brief = await self.memory.get_latest_session_brief(session_id)
        except Exception as e:
            logger.warning("Session brief fetch failed: %s", e)

        focus_subgraph = None
        if focus_ref is not None and getattr(self.memory, "_graph", None) is not None:
            try:
                focus_subgraph = await self.memory._graph.subgraph(
                    focus_ref, max_depth=2, max_nodes=20,
                )
            except Exception as e:
                logger.warning("Focus subgraph fetch failed: %s", e)

        daily_lines = await self._get_daily_context()

        assembled = self.prompt_assembler.assemble(
            facts=facts, memories=memories, plan=plan,
            focus_subgraph=focus_subgraph,
            daily_context_lines=daily_lines,
            session_brief=session_brief,
        )
        system_blocks = assembled.to_anthropic_blocks()

        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in conversation if m["role"] in ("user", "assistant")
        ]
        active_tools = self._initial_active_tools(plan)
        loaded_tool_names = {t["name"] for t in active_tools}

        def _check_cancel():
            if cancel_event is not None and cancel_event.is_set():
                raise Cancelled()

        try:
            for iteration in range(self.max_iterations):
                _check_cancel()
                if iteration > 0:
                    await asyncio.sleep(ITER_DELAY_S)

                accumulated_text = ""
                final_response = None
                call_start = time.perf_counter()
                try:
                    stream_kwargs = dict(
                        model=model,
                        max_tokens=4096,
                        system=system_blocks,
                        tools=self._cache_controlled_tools(active_tools) if active_tools else anthropic.NOT_GIVEN,
                        messages=messages,
                    )
                    if thinking is not None:
                        stream_kwargs["thinking"] = thinking
                    async with self.client.messages.stream(**stream_kwargs) as stream:
                        async for ev in stream:
                            _check_cancel()
                            # SDK >= 0.25 emits TextEvent / ContentBlockDeltaEvent
                            if hasattr(ev, "type") and ev.type == "text":
                                # Convenience event from the streaming helper
                                accumulated_text += ev.text
                                yield StreamEvent(type="text_delta", text=ev.text)
                            elif hasattr(ev, "type") and ev.type == "content_block_delta":
                                delta = getattr(ev, "delta", None)
                                if delta is not None and getattr(delta, "type", "") == "text_delta":
                                    accumulated_text += delta.text
                                    yield StreamEvent(type="text_delta", text=delta.text)
                        final_response = await stream.get_final_message()
                except Cancelled:
                    yield StreamEvent(type="cancelled")
                    return
                except Exception as e:
                    logger.exception("Stream error on iteration %d", iteration)
                    yield StreamEvent(type="error", error=str(e))
                    return

                if final_response is None:
                    yield StreamEvent(type="error", error="empty stream response")
                    return

                if self.audit_logger and getattr(final_response, "usage", None) is not None:
                    try:
                        await self.audit_logger.log_usage(
                            session_id=session_id, model=model,
                            usage=final_response.usage,
                            duration_ms=int((time.perf_counter() - call_start) * 1000),
                            iteration=iteration,
                        )
                    except Exception:
                        pass

                if not any(b.type == "tool_use" for b in final_response.content):
                    final_text = accumulated_text or "".join(
                        b.text for b in final_response.content if b.type == "text"
                    )
                    await self.memory.add_message(session_id, "assistant", final_text, interface)
                    self._schedule_compaction(session_id, plan=plan)
                    yield StreamEvent(type="complete", text=final_text)
                    return

                # Handle tool_use blocks (same parallel logic as run())
                assistant_content = []
                tool_use_blocks = []
                for block in final_response.content:
                    if block.type == "text":
                        assistant_content.append({"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        assistant_content.append({
                            "type": "tool_use", "id": block.id,
                            "name": block.name, "input": block.input,
                        })
                        tool_use_blocks.append(block)
                        yield StreamEvent(
                            type="tool_call", tool_name=block.name, tool_input=block.input,
                        )

                tool_results = []
                exec_blocks = []
                for b in tool_use_blocks:
                    if b.name == "tool-search":
                        result_text = self._handle_tool_search(b.input, active_tools, loaded_tool_names)
                        tool_results.append({
                            "type": "tool_result", "tool_use_id": b.id, "content": result_text,
                        })
                        yield StreamEvent(
                            type="tool_result", tool_name=b.name, tool_result=result_text[:200],
                        )
                    else:
                        exec_blocks.append(b)

                if exec_blocks:
                    results = await asyncio.gather(
                        *(self._execute_tool(b.name, b.input, session_id=session_id)
                          for b in exec_blocks),
                        return_exceptions=True,
                    )
                    for b, r in zip(exec_blocks, results):
                        content = (
                            f"Error executing {b.name}: {r}"
                            if isinstance(r, Exception) else r
                        )
                        tool_results.append({
                            "type": "tool_result", "tool_use_id": b.id, "content": content,
                        })
                        yield StreamEvent(
                            type="tool_result", tool_name=b.name, tool_result=content[:200],
                        )

                order = {b.id: i for i, b in enumerate(tool_use_blocks)}
                tool_results.sort(key=lambda r: order.get(r["tool_use_id"], 0))

                messages.append({"role": "assistant", "content": assistant_content})
                messages.append({"role": "user", "content": tool_results})

            # Hit iteration limit
            fallback = "Hit the tool loop limit. Tell me to continue if you want me to keep going."
            await self.memory.add_message(session_id, "assistant", fallback, interface)
            yield StreamEvent(type="complete", text=fallback)

        except Cancelled:
            yield StreamEvent(type="cancelled")

    def _schedule_compaction(self, session_id: str, plan: Plan | None = None) -> None:
        """Fire-and-forget compaction after a turn completes.

        Passes focus + intent so the Haiku summarizer preserves the parts of
        the conversation the user is currently working on. The user already
        has their response — compaction runs in the background."""
        if self.compactor is None:
            return
        focus_hint = ""
        intent = ""
        if plan is not None:
            if plan.focus is not None:
                focus_hint = str(plan.focus)
            intent = plan.intent or ""
        try:
            asyncio.create_task(
                self._compact_safely(session_id, focus_hint=focus_hint, intent=intent)
            )
        except RuntimeError:
            # No running loop (test contexts) — caller can run compactor manually
            pass

    async def _compact_safely(
        self, session_id: str, focus_hint: str = "", intent: str = "",
    ) -> None:
        try:
            brief = await self.compactor.maybe_compact(
                session_id, focus_hint=focus_hint, intent=intent,
            )
            if brief:
                logger.info("Compacted session %s — %d rows summarized",
                            session_id, brief.rows_compacted)
        except Exception as e:
            logger.warning("Compaction failed for %s: %s", session_id, e)

    async def _get_daily_context(self) -> list[str]:
        """Return today's commitments/MEDDIC-gap lines, cached per-day.

        Cached for ~1 hour rather than per-day so newly-created action items
        and reminders surface without a long staleness gap. Stale cache means
        Block C breakpoint stays warm across consecutive turns within the hour.
        """
        from datetime import datetime, timezone
        if getattr(self.memory, "session_maker", None) is None:
            return []
        bucket = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")
        cached = self._daily_cache.get(bucket)
        if cached is not None:
            return cached
        try:
            lines = await build_daily_context_lines(self.memory.session_maker)
        except Exception as e:
            logger.warning("Daily context build failed: %s", e)
            lines = []
        # Trim cache to last 4 buckets to bound memory
        self._daily_cache = {bucket: lines, **{
            k: v for k, v in list(self._daily_cache.items())[-3:]
        }}
        return lines

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

    @staticmethod
    def _error_signatures(tool_use_blocks, tool_results) -> set[str]:
        """Build a set of stable signatures for tool calls whose result is an
        error string. Signature = tool_name + hashed args, so a retry with
        the same args is detectable in the next iteration."""
        import hashlib
        results_by_id = {r["tool_use_id"]: r for r in tool_results}
        sigs: set[str] = set()
        for b in tool_use_blocks:
            r = results_by_id.get(b.id)
            if not r:
                continue
            content = r.get("content", "")
            is_error = isinstance(content, str) and (
                content.startswith("Error executing") or content.startswith("Error:")
            )
            if not is_error:
                continue
            try:
                args_str = json.dumps(b.input or {}, default=str, sort_keys=True)
            except (TypeError, ValueError):
                args_str = str(b.input)
            sig_hash = hashlib.sha256(args_str.encode("utf-8")).hexdigest()[:10]
            sigs.add(f"{b.name}({sig_hash})")
        return sigs

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

    async def _execute_tool_bypass_gate(
        self, tool_name: str, tool_input: dict, session_id: str = ""
    ) -> str:
        """Run a tool without ActionGate interception. Used by the pending-action
        approver to execute a previously-queued external action."""
        return await self._execute_tool(
            tool_name, tool_input, session_id=session_id, _bypass_gate=True,
        )

    async def _execute_tool(
        self, tool_name: str, tool_input: dict, session_id: str = "", _bypass_gate: bool = False,
    ) -> str:
        skill = self.tool_map.get(tool_name)
        if not skill:
            return f"Error: unknown tool '{tool_name}'"

        method = skill.get_tool_method(tool_name)
        if not method:
            return f"Error: method not found for '{tool_name}'"

        # Approve-external safety gate — queue instead of executing.
        # Audit logger sees the call regardless (safety field captures intent).
        safety = skill.get_tool_safety(tool_name) if hasattr(skill, "get_tool_safety") else Safety.AUTO
        if self.action_gate is not None and safety == Safety.APPROVE_EXTERNAL and not _bypass_gate:
            queued = await self.action_gate.intercept(
                session_id=session_id, tool_name=tool_name, tool_input=tool_input,
            )
            if self.audit_logger:
                await self.audit_logger.log(
                    tool_name=tool_name, args=tool_input,
                    result_status="queued", result_summary=queued.get("summary", "")[:200],
                    session_id=session_id, duration_ms=0,
                )
            return json.dumps(queued)

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
