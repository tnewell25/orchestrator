"""Planner JSON parsing + focus resolution + intent detection."""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.core.constants import EntityType
from orchestrator.core.graph import EntityRef
from orchestrator.core.planner import Intent, Plan, Planner, _parse_plan


# ---- Plan.to_prompt_hint ------------------------------------


def test_plan_hint_omits_empty_fields():
    p = Plan(intent=Intent.QUERY, focus=EntityRef(EntityType.DEAL, "d1"))
    hint = p.to_prompt_hint()
    assert "INTENT: QUERY" in hint
    assert "FOCUS: Deal:d1" in hint
    assert "WHY:" not in hint  # rationale empty
    assert "SUGGESTED" not in hint


def test_plan_hint_includes_parallel_groups():
    p = Plan(
        intent=Intent.PREP,
        suggested_tools=["a", "b", "c"],
        parallel_groups=[["a", "b"], ["c"]],
    )
    hint = p.to_prompt_hint()
    assert "PARALLELIZABLE" in hint
    assert "[a, b]" in hint  # only parallel-able groups (>1 item) shown


# ---- _parse_plan ---------------------------------------------


@pytest.mark.asyncio
async def test_parse_plan_full_object():
    payload = json.dumps({
        "intent": "STRATEGY",
        "focus": {"type": "Deal", "name": "Bosch Forge"},
        "rationale": "user wants game plan",
        "suggested_tools": ["graph-context_for", "dealhealth-score"],
        "parallel_groups": [["graph-context_for", "dealhealth-score"]],
        "use_thinking": True,
    })

    async def resolver(name, type_):
        return EntityRef(EntityType.DEAL, "d-bosch-resolved")

    plan = await _parse_plan(payload, resolver)
    assert plan.intent == Intent.STRATEGY
    assert plan.focus == EntityRef(EntityType.DEAL, "d-bosch-resolved")
    assert plan.use_thinking is True
    assert "graph-context_for" in plan.suggested_tools
    assert plan.parallel_groups == [["graph-context_for", "dealhealth-score"]]


@pytest.mark.asyncio
async def test_parse_plan_handles_markdown_fenced_json():
    raw = "```json\n" + json.dumps({"intent": "CRUD"}) + "\n```"
    plan = await _parse_plan(raw, None)
    assert plan.intent == Intent.CRUD


@pytest.mark.asyncio
async def test_parse_plan_extracts_object_from_prose():
    raw = "Here you go: {\"intent\": \"QUERY\"} -- enjoy"
    plan = await _parse_plan(raw, None)
    assert plan.intent == Intent.QUERY


@pytest.mark.asyncio
async def test_parse_plan_unknown_intent_falls_back_to_ambiguous():
    raw = json.dumps({"intent": "MADE_UP_INTENT"})
    plan = await _parse_plan(raw, None)
    assert plan.intent == Intent.AMBIGUOUS


@pytest.mark.asyncio
async def test_parse_plan_returns_empty_on_garbage():
    plan = await _parse_plan("not json", None)
    assert plan.intent == Intent.AMBIGUOUS
    assert plan.focus is None


@pytest.mark.asyncio
async def test_parse_plan_focus_unresolvable_keeps_intent():
    payload = json.dumps({
        "intent": "PREP",
        "focus": {"type": "Contact", "name": "Unknown Person"},
    })

    async def resolver(name, type_):
        return None  # not in DB

    plan = await _parse_plan(payload, resolver)
    assert plan.intent == Intent.PREP
    assert plan.focus is None  # unresolved


# ---- Planner.plan (mocked LLM) -------------------------------


@pytest.mark.asyncio
async def test_planner_plan_full_pipeline():
    payload = json.dumps({
        "intent": "QUERY",
        "focus": {"type": "Company", "name": "Bosch"},
        "rationale": "status check",
        "suggested_tools": ["graph-context_for"],
    })
    fake_block = MagicMock(type="text", text=payload)
    fake_resp = MagicMock(content=[fake_block])

    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=fake_resp)

    async def resolver(name, type_):
        return EntityRef(type_, "co-bosch")

    planner = Planner(client)
    plan = await planner.plan(
        user_message="what's going on with Bosch?",
        recent_summary="user: hi",
        known_entities_summary="Companies: Bosch",
        available_tools=["graph-context_for", "deal-find"],
        entity_resolver=resolver,
    )

    assert plan.intent == Intent.QUERY
    assert plan.focus == EntityRef(EntityType.COMPANY, "co-bosch")
    assert plan.rationale == "status check"


@pytest.mark.asyncio
async def test_planner_returns_empty_plan_without_client():
    planner = Planner(anthropic_client=None)
    plan = await planner.plan("hello", "", "", [], None)
    assert plan.intent == Intent.AMBIGUOUS
    assert plan.focus is None


@pytest.mark.asyncio
async def test_planner_handles_llm_exception_gracefully():
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("api down"))
    planner = Planner(client)
    plan = await planner.plan("hi", "", "", [], None)
    assert plan.intent == Intent.AMBIGUOUS  # fail-soft
