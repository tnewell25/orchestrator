"""ActionGate — interception, approval flow, expiry, dedup of external tools."""
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.core.action_gate import ActionGate
from orchestrator.core.agent import Agent
from orchestrator.core.skill_base import Safety, Skill, tool
from orchestrator.core.tool_registry import ToolRegistry
from orchestrator.db.models import PendingAction


@pytest.mark.asyncio
async def test_intercept_creates_pending_row(session_maker):
    gate = ActionGate(session_maker)
    out = await gate.intercept(
        session_id="s1", tool_name="gmail-send",
        tool_input={"to": "anja@bosch.com", "subject": "Hi", "body": "..."},
    )
    assert out["queued_for_approval"] is True
    assert out["pending_action_id"]
    assert "anja@bosch.com" in out["summary"]


@pytest.mark.asyncio
async def test_approve_flips_status(session_maker):
    gate = ActionGate(session_maker)
    out = await gate.intercept(
        session_id="s1", tool_name="gmail-send",
        tool_input={"to": "x@y.com", "subject": "S", "body": "B"},
    )
    aid = out["pending_action_id"]

    approved = await gate.approve(aid)
    assert approved is not None
    assert approved.status == "approved"


@pytest.mark.asyncio
async def test_approve_expired_returns_none(session_maker):
    gate = ActionGate(session_maker, expiry=timedelta(seconds=1))
    out = await gate.intercept(
        session_id="s1", tool_name="gmail-send",
        tool_input={"to": "x@y.com", "subject": "S", "body": "B"},
    )
    aid = out["pending_action_id"]

    # Force expiry
    async with session_maker() as s:
        row = await s.get(PendingAction, aid)
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=10)
        await s.commit()

    result = await gate.approve(aid)
    assert result is None  # expired


@pytest.mark.asyncio
async def test_reject_flips_status(session_maker):
    gate = ActionGate(session_maker)
    out = await gate.intercept(
        session_id="s1", tool_name="calendar-create_event",
        tool_input={"title": "Test", "start": "2026-04-20T10:00:00", "end": "2026-04-20T11:00:00"},
    )
    aid = out["pending_action_id"]

    rejected = await gate.reject(aid)
    assert rejected.status == "rejected"


@pytest.mark.asyncio
async def test_mark_executed_records_result(session_maker):
    gate = ActionGate(session_maker)
    out = await gate.intercept(
        session_id="s1", tool_name="gmail-send",
        tool_input={"to": "x@y.com", "subject": "S", "body": "B"},
    )
    aid = out["pending_action_id"]
    await gate.approve(aid)
    await gate.mark_executed(aid, "sent ok, message_id=abc")

    async with session_maker() as s:
        row = await s.get(PendingAction, aid)
    assert row.status == "executed"
    assert "sent ok" in row.result_summary


@pytest.mark.asyncio
async def test_list_pending_filters_by_status_and_session(session_maker):
    gate = ActionGate(session_maker)
    await gate.intercept("s1", "gmail-send", {"to": "a", "subject": "x", "body": "y"})
    await gate.intercept("s2", "gmail-send", {"to": "b", "subject": "x", "body": "y"})
    out3 = await gate.intercept("s1", "calendar-create_event",
                                {"title": "t", "start": "x", "end": "y"})
    await gate.reject(out3["pending_action_id"])

    s1_pending = await gate.list_pending("s1")
    assert len(s1_pending) == 1   # one was rejected, one still pending


# ---- Agent integration -----------------------------------------------


class _ExternalSkill(Skill):
    name = "ext"
    @tool("Internal read-only tool", safety=Safety.AUTO)
    async def get_status(self) -> str:
        return "ok"
    @tool("Externally visible action", safety=Safety.APPROVE_EXTERNAL)
    async def fire_missile(self, target: str) -> str:
        return f"BOOM {target}"


def _build_agent_with_gate(session_maker):
    skills = [_ExternalSkill()]
    agent = Agent.__new__(Agent)
    agent.memory = MagicMock()
    agent.skills = {s.name: s for s in skills}
    agent.settings = SimpleNamespace(default_model="x", agent_name="Test")
    agent.token_manager = None
    agent.audit_logger = None
    agent.max_iterations = 5
    agent.planner = None
    agent.entity_extractor = None
    agent.lazy_tools = False
    agent.essentials = ()
    agent.compactor = None
    agent.action_gate = ActionGate(session_maker)
    agent.registry = ToolRegistry(skills)
    agent.tools = []
    agent.tool_map = {}
    for s in skills:
        for sch in s.get_tools():
            agent.tools.append(sch)
            agent.tool_map[sch["name"]] = s
    return agent


@pytest.mark.asyncio
async def test_agent_intercepts_external_tool(session_maker):
    agent = _build_agent_with_gate(session_maker)
    result = await agent._execute_tool("ext-fire_missile", {"target": "X"}, session_id="s1")
    parsed = json.loads(result)
    assert parsed["queued_for_approval"] is True
    # Real tool method should NOT have run — verify no "BOOM" in any side effect
    pending = await agent.action_gate.list_pending("s1")
    assert len(pending) == 1


@pytest.mark.asyncio
async def test_agent_runs_auto_tool_directly(session_maker):
    agent = _build_agent_with_gate(session_maker)
    result = await agent._execute_tool("ext-get_status", {}, session_id="s1")
    assert result == "ok"
    pending = await agent.action_gate.list_pending("s1")
    assert pending == []


@pytest.mark.asyncio
async def test_agent_bypass_gate_executes_external(session_maker):
    """The approval API uses _execute_tool_bypass_gate to actually run a
    previously-queued external action."""
    agent = _build_agent_with_gate(session_maker)
    result = await agent._execute_tool_bypass_gate(
        "ext-fire_missile", {"target": "X"}, session_id="s1",
    )
    assert result == "BOOM X"
