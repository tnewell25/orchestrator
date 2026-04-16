"""PromptAssembler — block layout, cache_control markers, mode hints, daily context."""
from datetime import date, datetime, timedelta, timezone

import pytest

from orchestrator.core.constants import EdgeKind, EntityType
from orchestrator.core.graph import EdgeRow, EntityRef, Subgraph
from orchestrator.core.planner import Intent, Plan
from orchestrator.core.prompt_assembler import (
    AssembledPrompt,
    BLOCK_A_TEMPLATE,
    MODE_HINTS,
    PromptAssembler,
    build_daily_context_lines,
)
from orchestrator.db.models import ActionItem, Deal, Reminder


# ---- Block A is stable -----------------------------------------------


def test_block_a_includes_agent_name_and_core_rules():
    asm = PromptAssembler(agent_name="Tethyr")
    p = asm.assemble(facts=[], memories=[], plan=Plan())
    assert "Tethyr" in p.block_a
    assert "CORE RULES" in p.block_a
    assert "NEVER" in p.block_a


def test_block_a_constant_across_calls():
    asm = PromptAssembler(agent_name="X")
    p1 = asm.assemble(facts=[], memories=[], plan=Plan())
    p2 = asm.assemble(facts=[{"category": "x", "key": "y", "value": "z"}],
                      memories=[], plan=Plan())
    assert p1.block_a == p2.block_a


# ---- Block B (user profile from facts) -------------------------------


def test_block_b_groups_facts_by_category():
    facts = [
        {"category": "user_company", "key": "name", "value": "Acme"},
        {"category": "user_company", "key": "size", "value": "5000"},
        {"category": "user_role", "key": "title", "value": "Sr SE"},
    ]
    asm = PromptAssembler()
    p = asm.assemble(facts=facts, memories=[], plan=Plan())
    assert "USER PROFILE" in p.block_b
    assert "[user_company]" in p.block_b
    assert "[user_role]" in p.block_b
    assert "Acme" in p.block_b


def test_block_b_empty_when_no_facts():
    asm = PromptAssembler()
    p = asm.assemble(facts=[], memories=[], plan=Plan())
    assert p.block_b == ""


# ---- Block C (daily context) -----------------------------------------


def test_block_c_renders_daily_lines():
    asm = PromptAssembler()
    p = asm.assemble(
        facts=[], memories=[], plan=Plan(),
        daily_context_lines=["OPEN: Send proposal", "MEDDIC gap [Bosch]: missing economic buyer"],
    )
    assert "TODAY:" in p.block_c
    assert "Send proposal" in p.block_c
    assert "Bosch" in p.block_c


def test_block_c_empty_when_no_lines():
    asm = PromptAssembler()
    p = asm.assemble(facts=[], memories=[], plan=Plan(), daily_context_lines=[])
    assert p.block_c == ""


# ---- Block D (per-turn focus, memories, mode hint) -------------------


def test_block_d_includes_mode_hint_for_known_intent():
    asm = PromptAssembler()
    plan = Plan(intent=Intent.PREP, focus=EntityRef(EntityType.DEAL, "d1"))
    p = asm.assemble(facts=[], memories=[], plan=plan)
    assert MODE_HINTS[Intent.PREP] in p.block_d
    assert "FOCUS: Deal:d1" in p.block_d


def test_block_d_renders_focus_subgraph():
    asm = PromptAssembler()
    sg = Subgraph(root=EntityRef(EntityType.DEAL, "d1"))
    sg.edges.append(EdgeRow(
        from_ref=EntityRef(EntityType.DEAL, "d1"),
        to_ref=EntityRef(EntityType.CONTACT, "c-anja"),
        kind=EdgeKind.STAKEHOLDER_IN,
        reinforcement_count=3,
    ))
    plan = Plan(intent=Intent.QUERY, focus=EntityRef(EntityType.DEAL, "d1"))
    p = asm.assemble(facts=[], memories=[], plan=plan, focus_subgraph=sg)
    assert "FOCUS SUBGRAPH" in p.block_d
    assert "stakeholder_in" in p.block_d
    assert "reinforced" in p.block_d


def test_block_d_renders_memories_with_source():
    asm = PromptAssembler()
    p = asm.assemble(
        facts=[], memories=[
            {"content": "Anja loves rugby", "source": "voice"},
            {"content": "Markus runs procurement", "source": "meeting"},
        ], plan=Plan(intent=Intent.PREP),
    )
    assert "RELEVANT MEMORIES" in p.block_d
    assert "[voice]" in p.block_d
    assert "Anja loves rugby" in p.block_d


def test_block_d_always_contains_datetime_anchor():
    """Regression guard — even AMBIGUOUS intent with no data must include the
    datetime header so the model knows what day it is."""
    asm = PromptAssembler()
    p = asm.assemble(facts=[], memories=[], plan=Plan(intent=Intent.AMBIGUOUS))
    assert "CURRENT DATE/TIME" in p.block_d
    # But still shouldn't contain mode-hint / focus sections
    assert "MODE:" not in p.block_d
    assert "FOCUS:" not in p.block_d


# ---- Anthropic block conversion + cache breakpoints ------------------


def test_to_anthropic_blocks_marks_cacheable_blocks_only():
    asm = PromptAssembler()
    p = asm.assemble(
        facts=[{"category": "x", "key": "k", "value": "v"}],
        memories=[{"content": "m", "source": "s"}],
        plan=Plan(intent=Intent.QUERY),
        daily_context_lines=["lunch at noon"],
    )
    blocks = p.to_anthropic_blocks()

    # Block A always present + cached
    assert blocks[0]["text"] == p.block_a
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}

    # Block B + Block C also cached (separate breakpoints)
    cached_count = sum(1 for b in blocks if "cache_control" in b)
    assert cached_count == 3

    # Block D last and NOT cached
    assert "cache_control" not in blocks[-1]


def test_to_anthropic_blocks_omits_optional_blocks():
    """With no facts or daily lines, B and C are skipped — but A and D
    (datetime anchor) are always present."""
    asm = PromptAssembler()
    p = asm.assemble(facts=[], memories=[], plan=Plan(intent=Intent.AMBIGUOUS))
    blocks = p.to_anthropic_blocks()
    # A (cached) + D (volatile datetime header) — B and C omitted
    assert len(blocks) == 2
    assert "cache_control" in blocks[0]    # A cached
    assert "cache_control" not in blocks[1] # D volatile
    assert "CURRENT DATE/TIME" in blocks[1]["text"]


# ---- Daily context builder -------------------------------------------


@pytest.mark.asyncio
async def test_build_daily_context_surfaces_overdue_actions(session_maker):
    yesterday = date.today() - timedelta(days=2)
    async with session_maker() as s:
        s.add(ActionItem(
            id="a1", description="Send pricing to Anja",
            due_date=yesterday, status="open",
        ))
        await s.commit()

    lines = await build_daily_context_lines(session_maker)
    assert any("Send pricing" in ln for ln in lines)


@pytest.mark.asyncio
async def test_build_daily_context_surfaces_meddic_gaps(session_maker):
    async with session_maker() as s:
        s.add(Deal(
            id="d1", name="Bosch Forge",
            stage="proposal",
            # no economic_buyer_id, champion_id, metrics
        ))
        await s.commit()

    lines = await build_daily_context_lines(session_maker)
    assert any("MEDDIC gap [Bosch Forge]" in ln for ln in lines)
    assert any("economic buyer" in ln for ln in lines)


@pytest.mark.asyncio
async def test_build_daily_context_skips_complete_deals(session_maker):
    async with session_maker() as s:
        s.add(Deal(
            id="d2", name="Complete Deal",
            stage="proposal",
            economic_buyer_id="x", champion_id="y", metrics="m",
        ))
        await s.commit()

    lines = await build_daily_context_lines(session_maker)
    assert not any("Complete Deal" in ln for ln in lines)
