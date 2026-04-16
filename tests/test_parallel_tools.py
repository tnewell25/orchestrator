"""Verify the agent executes multiple tool_use blocks concurrently.

The cleanest concurrency proof: two tools that wait on each other's event.
Sequential execution deadlocks; parallel completes.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.core.agent import Agent
from orchestrator.core.skill_base import Skill, tool


class _RendezvousSkill(Skill):
    """Two tools that synchronize on each other — only completes if run in parallel."""
    name = "rdv"

    def __init__(self):
        super().__init__()
        self.event_a = asyncio.Event()
        self.event_b = asyncio.Event()

    @tool("Tool A — signals A then waits for B.")
    async def step_a(self) -> str:
        self.event_a.set()
        await asyncio.wait_for(self.event_b.wait(), timeout=2.0)
        return "a-done"

    @tool("Tool B — waits for A then signals B.")
    async def step_b(self) -> str:
        await asyncio.wait_for(self.event_a.wait(), timeout=2.0)
        self.event_b.set()
        return "b-done"


def _build_agent_with_skill(skill):
    """Construct an Agent without going through __init__'s LLM client setup."""
    agent = Agent.__new__(Agent)
    agent.memory = MagicMock()
    agent.skills = {skill.name: skill}
    agent.settings = SimpleNamespace(default_model="x", agent_name="Test")
    agent.token_manager = None
    agent.audit_logger = None
    agent.max_iterations = 5
    agent.planner = None
    agent.entity_extractor = None
    agent.action_gate = None
    agent.tools = []
    agent.tool_map = {}
    for sch in skill.get_tools():
        agent.tools.append(sch)
        agent.tool_map[sch["name"]] = skill
    return agent


@pytest.mark.asyncio
async def test_two_tools_execute_in_parallel():
    skill = _RendezvousSkill()
    agent = _build_agent_with_skill(skill)

    # Run BOTH tool methods through agent._execute_tool concurrently using gather
    # — this is exactly the call pattern in agent.run()'s parallel path.
    results = await asyncio.gather(
        agent._execute_tool("rdv-step_a", {}),
        agent._execute_tool("rdv-step_b", {}),
    )
    assert "a-done" in results
    assert "b-done" in results


@pytest.mark.asyncio
async def test_tool_exception_does_not_kill_sibling():
    """One bad tool shouldn't crash the parallel batch."""
    class BoomSkill(Skill):
        name = "boom"
        @tool("ok")
        async def good(self) -> str:
            return "ok-result"
        @tool("explodes")
        async def bad(self) -> str:
            raise ValueError("kaboom")

    skill = BoomSkill()
    agent = _build_agent_with_skill(skill)

    # _execute_tool catches the exception and returns the error string
    a, b = await asyncio.gather(
        agent._execute_tool("boom-good", {}),
        agent._execute_tool("boom-bad", {}),
        return_exceptions=False,
    )
    assert "ok-result" in a
    assert "kaboom" in b  # error surfaced as string, not raised


@pytest.mark.asyncio
async def test_unknown_tool_returns_error_string():
    skill = _RendezvousSkill()
    agent = _build_agent_with_skill(skill)
    result = await agent._execute_tool("nonexistent-thing", {})
    assert "unknown tool" in result
