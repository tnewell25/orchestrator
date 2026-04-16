"""StrategyFanout — parallel Haiku sub-agents for STRATEGY intent."""
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.core.constants import EntityType
from orchestrator.core.graph import EntityRef
from orchestrator.core.planner import Intent, Plan
from orchestrator.core.strategy_fanout import StrategyContext, StrategyFanout
from orchestrator.db.models import (
    Company, Contact, Deal, DealStakeholder, WinLossRecord,
)


def _fake_client_returning(text: str):
    block = MagicMock(type="text", text=text)
    resp = MagicMock(content=[block])
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=resp)
    return client


# ---- Guards: should_run ------------------------------------------


def test_should_run_only_for_strategy_with_focus():
    fan = StrategyFanout(session_maker=MagicMock(), anthropic_client=MagicMock())
    assert fan.should_run(Plan(intent=Intent.STRATEGY, focus=EntityRef(EntityType.DEAL, "d1"))) is True
    assert fan.should_run(Plan(intent=Intent.CRUD, focus=EntityRef(EntityType.DEAL, "d1"))) is False
    assert fan.should_run(Plan(intent=Intent.STRATEGY, focus=None)) is False


def test_should_run_false_without_client():
    fan = StrategyFanout(session_maker=MagicMock(), anthropic_client=None)
    assert fan.should_run(Plan(intent=Intent.STRATEGY, focus=EntityRef(EntityType.DEAL, "d1"))) is False


# ---- StrategyContext rendering ----------------------------------


def test_strategy_context_render_both_parts():
    c = StrategyContext(lessons="• lesson A", relationships="• champion: Anja")
    block = c.to_block()
    assert "HISTORICAL LESSONS" in block
    assert "RELATIONSHIP MAP" in block
    assert c.is_empty() is False


def test_strategy_context_empty():
    assert StrategyContext().is_empty() is True
    assert StrategyContext().to_block() == ""


# ---- Fan-out integration ---------------------------------------


@pytest.mark.asyncio
async def test_gather_pulls_winloss_and_stakeholders(session_maker):
    """Full pipeline: seed DB → gather runs both sub-agents → returns synthesis."""
    async with session_maker() as s:
        s.add(Company(id="co1", name="Bosch"))
        s.add(Deal(id="d1", company_id="co1", name="Bosch Forge", stage="negotiation"))
        s.add(Deal(id="d0", company_id="co1", name="Prior Bosch", stage="closed_lost"))
        s.add(Contact(id="anja", name="Anja Weber", personal_notes="loves rugby"))
        s.add(WinLossRecord(
            id="w1", deal_id="d0", outcome="lost",
            primary_reason="price", what_worked="champion",
            what_didnt="pricing pushback", lessons="lead with TCO",
            value_usd=200_000,
        ))
        s.add(DealStakeholder(
            id="ds1", deal_id="d1", contact_id="anja",
            role="champion", influence="high", sentiment="supportive",
        ))
        await s.commit()

    client = _fake_client_returning("synthesized output")
    fan = StrategyFanout(session_maker, anthropic_client=client)

    plan = Plan(intent=Intent.STRATEGY, focus=EntityRef(EntityType.DEAL, "d1"))
    result = await fan.gather(plan)

    assert result.lessons == "synthesized output"
    assert result.relationships == "synthesized output"
    # Two LLM calls (one per sub-agent) — verify parallel invocation
    assert client.messages.create.call_count == 2


@pytest.mark.asyncio
async def test_gather_runs_sub_agents_concurrently(session_maker):
    """Verify parallelism: if we sleep in the mocked call, total time ~= one
    sleep, not two (proves asyncio.gather was used)."""
    async with session_maker() as s:
        s.add(Deal(id="d1", name="Active", stage="proposal"))
        s.add(WinLossRecord(
            id="w1", deal_id="d1", outcome="won",
            primary_reason="relationship", lessons="keep doing X",
        ))
        s.add(Contact(id="c1", name="Someone", personal_notes="x"))
        s.add(DealStakeholder(
            id="ds1", deal_id="d1", contact_id="c1",
            role="champion", influence="high", sentiment="supportive",
        ))
        await s.commit()

    async def slow_create(*args, **kwargs):
        await asyncio.sleep(0.1)
        block = MagicMock(type="text", text="out")
        return MagicMock(content=[block])

    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = slow_create

    fan = StrategyFanout(session_maker, anthropic_client=client)
    plan = Plan(intent=Intent.STRATEGY, focus=EntityRef(EntityType.DEAL, "d1"))

    t0 = asyncio.get_event_loop().time()
    await fan.gather(plan)
    elapsed = asyncio.get_event_loop().time() - t0
    # Serial would be ~0.2s; parallel ~0.1s. Allow slack.
    assert elapsed < 0.18


@pytest.mark.asyncio
async def test_gather_empty_when_no_data(session_maker):
    """No deals/stakeholders → empty context (no wasteful LLM calls)."""
    async with session_maker() as s:
        s.add(Deal(id="lonely", name="No Stakes", stage="proposal"))
        await s.commit()

    client = _fake_client_returning("should not be called")
    fan = StrategyFanout(session_maker, anthropic_client=client)
    plan = Plan(intent=Intent.STRATEGY, focus=EntityRef(EntityType.DEAL, "lonely"))

    result = await fan.gather(plan)
    assert result.lessons == ""
    assert result.relationships == ""
    assert client.messages.create.call_count == 0  # no LLM calls wasted


@pytest.mark.asyncio
async def test_gather_returns_empty_on_non_strategy(session_maker):
    fan = StrategyFanout(session_maker, anthropic_client=_fake_client_returning("x"))
    result = await fan.gather(Plan(intent=Intent.CRUD, focus=EntityRef(EntityType.DEAL, "x")))
    assert result.is_empty() is True


@pytest.mark.asyncio
async def test_gather_tolerates_one_sub_agent_exception(session_maker):
    """If the lessons sub crashes, relationships still returns."""
    async with session_maker() as s:
        s.add(Deal(id="d1", name="x", stage="proposal"))
        s.add(Contact(id="c1", name="Anja", personal_notes="rugby"))
        s.add(DealStakeholder(
            id="ds1", deal_id="d1", contact_id="c1",
            role="champion", influence="high", sentiment="supportive",
        ))
        s.add(WinLossRecord(id="w1", deal_id="d1", outcome="won"))
        await s.commit()

    call_count = {"n": 0}
    async def flaky(*a, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("lessons API down")
        block = MagicMock(type="text", text="relationships out")
        return MagicMock(content=[block])

    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = flaky

    fan = StrategyFanout(session_maker, anthropic_client=client)
    result = await fan.gather(Plan(intent=Intent.STRATEGY, focus=EntityRef(EntityType.DEAL, "d1")))

    # Lessons failed softly (empty); relationships still populated.
    assert result.lessons == ""
    assert "relationships out" in result.relationships
