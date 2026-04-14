"""Anthropic tool-use agent loop with Claude Max OAuth support."""
import json
import logging
import time

import anthropic

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are {agent_name}, the user's AI chief of staff.

Your mission: capture everything the user tells you, keep their pipeline and
follow-ups organized, and act proactively so they never drop a ball.

Rules:
- Be concise — responses are read on a phone between meetings.
- Lead with the result ("Logged meeting with Markus — proposal due Fri") not process.
- For multi-step requests, do all the steps — don't ask permission at each one.
- Use tools freely. When the user describes a meeting, person, deal, or
  commitment, file it via the appropriate skill tool immediately.
- When capturing a meeting, extract and store: attendees, decisions, commitments
  (as action items with due dates), competitor mentions, and personal details
  about contacts (family, interests). The personal details are gold for
  relationship building — always save them.
- Any commitment the user makes ("I'll send him X by Friday") → create an
  action item with a due date.
- If the user asks about a deal, company, or person, pull full context: recent
  meetings, open action items, last touch.
- Never invent data. If you don't know something, say so or look it up.

{facts_context}
{memories_context}"""


class Agent:
    def __init__(
        self,
        memory,
        skills: list,
        settings,
        token_manager=None,
        audit_logger=None,
        max_iterations: int = 15,
    ):
        self.memory = memory
        self.skills = {s.name: s for s in skills}
        self.settings = settings
        self.token_manager = token_manager
        self.audit_logger = audit_logger
        self.max_iterations = max_iterations

        self._init_client()

        self.tools: list[dict] = []
        self.tool_map: dict = {}

        for skill in skills:
            for schema in skill.get_tools():
                self.tools.append(schema)
                self.tool_map[schema["name"]] = skill

    def _init_client(self):
        if self.token_manager and self.token_manager.access_token:
            self.client = anthropic.AsyncAnthropic(
                auth_token=self.token_manager.access_token,
                default_headers={"anthropic-beta": "oauth-2025-04-20"},
            )
        elif self.token_manager and self.token_manager.api_key:
            self.client = anthropic.AsyncAnthropic(api_key=self.token_manager.api_key)
        else:
            self.client = anthropic.AsyncAnthropic(
                api_key=self.settings.anthropic_api_key
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
        memories = await self.memory.recall(message, limit=5)

        facts_ctx = ""
        if facts:
            facts_ctx = "Known facts:\n" + "\n".join(
                f"- [{f['category']}] {f['key']}: {f['value']}" for f in facts
            )

        memories_ctx = ""
        if memories:
            memories_ctx = "Relevant memories:\n" + "\n".join(
                f"- {m['content']}" for m in memories
            )

        system = SYSTEM_PROMPT.format(
            agent_name=self.settings.agent_name,
            facts_context=facts_ctx,
            memories_context=memories_ctx,
        )

        messages = []
        for msg in conversation:
            if msg["role"] in ("user", "assistant"):
                messages.append({"role": msg["role"], "content": msg["content"]})

        recovered_this_run = False
        for _ in range(self.max_iterations):
            try:
                response = await self.client.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=system,
                    tools=self.tools if self.tools else anthropic.NOT_GIVEN,
                    messages=messages,
                )
            except anthropic.AuthenticationError:
                if not recovered_this_run and self.token_manager:
                    recovered_this_run = True
                    logger.warning("Auth error — running token recovery")
                    recovered = await self.token_manager.handle_auth_error()
                    if recovered:
                        self._init_client()
                        response = await self.client.messages.create(
                            model=model,
                            max_tokens=4096,
                            system=system,
                            tools=self.tools if self.tools else anthropic.NOT_GIVEN,
                            messages=messages,
                        )
                    else:
                        raise
                else:
                    raise

            if not any(b.type == "tool_use" for b in response.content):
                final_text = "".join(
                    b.text for b in response.content if b.type == "text"
                )
                await self.memory.add_message(
                    session_id, "assistant", final_text, interface
                )
                return final_text

            assistant_content = []
            tool_results = []

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
                    result = await self._execute_tool(
                        block.name, block.input, session_id=session_id
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )

            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})

        fallback = "Hit the tool loop limit. Tell me to continue if you want me to keep going."
        await self.memory.add_message(session_id, "assistant", fallback, interface)
        return fallback

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
