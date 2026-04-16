"""Agent.run_stream — text deltas, tool calls, cancellation."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.core.agent import Agent, Cancelled, StreamEvent


# ---- Fake Anthropic streaming infrastructure ----


class _FakeTextEvent:
    """Mimics the SDK's TextEvent — has .type='text' and .text=str."""
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeBlock:
    def __init__(self, kind, text="", id="", name="", input=None):
        self.type = kind
        self.text = text
        self.id = id
        self.name = name
        self.input = input or {}


class _FakeFinalMessage:
    def __init__(self, blocks):
        self.content = blocks


class _FakeStream:
    """Async context manager + async iterable + get_final_message()."""
    def __init__(self, events, final_blocks):
        self._events = events
        self._final = _FakeFinalMessage(final_blocks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        async def gen():
            for e in self._events:
                yield e
        return gen()

    async def get_final_message(self):
        return self._final


def _stream_returning(events, final_blocks):
    """Helper: returns a callable that ignores args and returns a fresh _FakeStream."""
    def make(*args, **kwargs):
        return _FakeStream(events, final_blocks)
    return make


def _build_agent(stream_factory):
    """Build a stripped-down Agent wired to a fake stream factory."""
    agent = Agent.__new__(Agent)
    agent.memory = MagicMock()
    agent.memory.add_message = AsyncMock()
    agent.memory.get_conversation = AsyncMock(return_value=[])
    agent.memory.get_facts = AsyncMock(return_value=[])
    agent.memory.recall = AsyncMock(return_value=[])
    agent.memory.get_latest_session_brief = AsyncMock(return_value="")
    agent.memory._graph = None
    agent.memory.session_maker = None
    agent.skills = {}
    agent.settings = SimpleNamespace(default_model="x", agent_name="Test")
    agent.token_manager = None
    agent.audit_logger = None
    agent.max_iterations = 5
    agent.planner = None
    agent.entity_extractor = None
    agent.lazy_tools = False
    agent.essentials = ()
    from orchestrator.core.prompt_assembler import PromptAssembler
    from orchestrator.core.tool_registry import ToolRegistry
    agent.prompt_assembler = PromptAssembler(agent_name="Test")
    agent._daily_cache = {}
    agent.compactor = None
    agent.registry = ToolRegistry([])
    agent.tools = []
    agent.tool_map = {}

    client = MagicMock()
    client.messages = MagicMock()
    client.messages.stream = stream_factory
    agent.client = client
    return agent


# ---- Tests ----


@pytest.mark.asyncio
async def test_streams_text_deltas_and_completes():
    events = [_FakeTextEvent("Hello "), _FakeTextEvent("world.")]
    final_blocks = [_FakeBlock("text", text="Hello world.")]
    agent = _build_agent(_stream_returning(events, final_blocks))

    received = []
    async for ev in agent.run_stream("hi", "s1"):
        received.append(ev)

    deltas = [e for e in received if e.type == "text_delta"]
    completes = [e for e in received if e.type == "complete"]

    assert [d.text for d in deltas] == ["Hello ", "world."]
    assert len(completes) == 1
    assert completes[0].text == "Hello world."


@pytest.mark.asyncio
async def test_stream_emits_tool_call_events_then_continues():
    """First iteration emits a tool_use; second iteration is the final text."""
    # iter1: model wants to call a tool
    iter1_events = []  # no text
    iter1_final = [_FakeBlock("tool_use", id="tu1", name="rdv-noop", input={})]
    # iter2: final response
    iter2_events = [_FakeTextEvent("Done.")]
    iter2_final = [_FakeBlock("text", text="Done.")]

    streams = iter([
        _FakeStream(iter1_events, iter1_final),
        _FakeStream(iter2_events, iter2_final),
    ])

    def factory(*a, **kw):
        return next(streams)

    agent = _build_agent(factory)

    # Register a fake tool
    from orchestrator.core.skill_base import Skill, tool as tool_decorator
    class RdvSkill(Skill):
        name = "rdv"
        @tool_decorator("does nothing")
        async def noop(self) -> str:
            return "ok"
    skill = RdvSkill()
    agent.skills["rdv"] = skill
    agent.tool_map["rdv-noop"] = skill
    sch = skill.get_tools()[0]
    agent.tools.append(sch)
    from orchestrator.core.tool_registry import ToolRegistry
    agent.registry = ToolRegistry([skill])

    received = []
    async for ev in agent.run_stream("call the tool", "s1"):
        received.append(ev)

    types = [e.type for e in received]
    assert "tool_call" in types
    assert "tool_result" in types
    assert types[-1] == "complete"
    assert received[-1].text == "Done."


@pytest.mark.asyncio
async def test_cancel_event_stops_stream_cleanly():
    events = [_FakeTextEvent("partial...")]
    final_blocks = [_FakeBlock("text", text="partial...")]
    agent = _build_agent(_stream_returning(events, final_blocks))

    cancel = asyncio.Event()
    cancel.set()  # already cancelled before we start

    received = []
    async for ev in agent.run_stream("hi", "s1", cancel_event=cancel):
        received.append(ev)

    types = [e.type for e in received]
    assert "cancelled" in types
    assert "complete" not in types


@pytest.mark.asyncio
async def test_stream_handles_api_error_gracefully():
    def factory(*a, **kw):
        raise RuntimeError("network down")
    agent = _build_agent(factory)

    received = []
    async for ev in agent.run_stream("hi", "s1"):
        received.append(ev)

    assert any(e.type == "error" for e in received)
    assert "network down" in next(e.error for e in received if e.type == "error")
