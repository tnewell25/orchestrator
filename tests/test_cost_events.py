"""Cost-aware event emission — TURN_COST_HIGH, SESSION_COST_EXCEEDED."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from orchestrator.core.agent import Agent
from orchestrator.core.events import EventBus, EventType
from orchestrator.core.rule_engine import (
    ActionDispatcher,
    PublishEvent,
    RuleEngine,
    rule_compact_on_high_cost,
)


class _FakeUsage:
    def __init__(self, input=0, output=0, cache_read=0, cache_creation=0):
        self.input_tokens = input
        self.output_tokens = output
        self.cache_read_input_tokens = cache_read
        self.cache_creation_input_tokens = cache_creation


def _build_agent(bus, turn_thresh=40_000, sess_thresh=500_000):
    agent = Agent.__new__(Agent)
    agent.event_bus = bus
    agent.settings = SimpleNamespace(
        turn_input_token_threshold=turn_thresh,
        session_input_token_threshold=sess_thresh,
        agent_name="Test",
    )
    agent._session_tokens = {}
    return agent


@pytest.mark.asyncio
async def test_turn_cost_high_fires_above_threshold():
    bus = EventBus()
    received = []
    async def h(e): received.append(e)
    bus.subscribe(EventType.TURN_COST_HIGH, h)

    agent = _build_agent(bus, turn_thresh=10_000)
    response = SimpleNamespace(usage=_FakeUsage(input=15_000, output=500))

    await agent._maybe_emit_cost_alerts("s1", response)
    assert len(received) == 1
    assert received[0].payload["input_tokens"] == 15_000


@pytest.mark.asyncio
async def test_turn_cost_high_silent_below_threshold():
    bus = EventBus()
    received = []
    async def h(e): received.append(e)
    bus.subscribe(EventType.TURN_COST_HIGH, h)

    agent = _build_agent(bus, turn_thresh=100_000)
    response = SimpleNamespace(usage=_FakeUsage(input=5000))
    await agent._maybe_emit_cost_alerts("s1", response)
    assert received == []


@pytest.mark.asyncio
async def test_session_exceeded_accumulates_across_turns():
    bus = EventBus()
    received = []
    async def h(e): received.append(e)
    bus.subscribe(EventType.SESSION_COST_EXCEEDED, h)

    agent = _build_agent(bus, turn_thresh=0, sess_thresh=15_000)

    # 3 turns at 6k each → exceeds after third
    for _ in range(3):
        await agent._maybe_emit_cost_alerts("s1", SimpleNamespace(usage=_FakeUsage(input=6000)))
    assert len(received) == 1
    assert received[0].payload["cumulative_input_tokens"] >= 15_000


@pytest.mark.asyncio
async def test_session_exceeded_resets_after_fire():
    """Once fired, the cumulative counter resets so we don't spam the event."""
    bus = EventBus()
    received = []
    async def h(e): received.append(e)
    bus.subscribe(EventType.SESSION_COST_EXCEEDED, h)

    agent = _build_agent(bus, turn_thresh=0, sess_thresh=10_000)
    await agent._maybe_emit_cost_alerts("s1", SimpleNamespace(usage=_FakeUsage(input=20_000)))
    assert len(received) == 1

    # Small follow-up turn — shouldn't trigger a second event
    await agent._maybe_emit_cost_alerts("s1", SimpleNamespace(usage=_FakeUsage(input=500)))
    assert len(received) == 1


@pytest.mark.asyncio
async def test_zero_threshold_disables_emission():
    bus = EventBus()
    received = []
    async def h(e): received.append(e)
    bus.subscribe_all(h)

    agent = _build_agent(bus, turn_thresh=0, sess_thresh=0)
    await agent._maybe_emit_cost_alerts("s1", SimpleNamespace(usage=_FakeUsage(input=1_000_000)))
    assert received == []


@pytest.mark.asyncio
async def test_no_event_bus_is_safe():
    agent = _build_agent(bus=None)
    await agent._maybe_emit_cost_alerts("s1", SimpleNamespace(usage=_FakeUsage(input=1000)))


@pytest.mark.asyncio
async def test_compact_rule_chains_on_session_cost_exceeded(session_maker):
    from orchestrator.core.events import Event as _Event
    ev = _Event(EventType.SESSION_COST_EXCEEDED, payload={"session_id": "s1"})
    actions = await rule_compact_on_high_cost(ev, session_maker)
    assert len(actions) == 1
    assert isinstance(actions[0], PublishEvent)
    assert actions[0].type == "compaction.requested"
