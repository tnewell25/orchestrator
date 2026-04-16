"""Intent-aware model selection — Haiku for CRUD/QUERY, thinking for STRATEGY."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from orchestrator.core.agent import Agent
from orchestrator.core.planner import Intent, Plan


def _build_agent(fast_intents: str = "CRUD,QUERY", thinking_budget: int = 5000):
    agent = Agent.__new__(Agent)
    agent.settings = SimpleNamespace(
        default_model="claude-sonnet-4-5",
        fast_model="claude-haiku-4-5",
        fast_model_intents=fast_intents,
        thinking_budget_tokens=thinking_budget,
        agent_name="Test",
    )
    return agent


def test_crud_routes_to_fast_model():
    agent = _build_agent()
    model, thinking = agent._select_model_and_thinking(Plan(intent=Intent.CRUD))
    assert model == "claude-haiku-4-5"
    assert thinking is None


def test_query_routes_to_fast_model():
    agent = _build_agent()
    model, thinking = agent._select_model_and_thinking(Plan(intent=Intent.QUERY))
    assert model == "claude-haiku-4-5"
    assert thinking is None


def test_prep_stays_on_default_model():
    agent = _build_agent()
    model, thinking = agent._select_model_and_thinking(Plan(intent=Intent.PREP))
    assert model == "claude-sonnet-4-5"
    assert thinking is None


def test_strategy_enables_thinking():
    agent = _build_agent()
    model, thinking = agent._select_model_and_thinking(Plan(intent=Intent.STRATEGY))
    assert model == "claude-sonnet-4-5"
    assert thinking == {"type": "enabled", "budget_tokens": 5000}


def test_plan_use_thinking_flag_forces_thinking_on_sonnet():
    """Planner can explicitly request thinking on non-STRATEGY intents."""
    agent = _build_agent()
    model, thinking = agent._select_model_and_thinking(
        Plan(intent=Intent.AMBIGUOUS, use_thinking=True)
    )
    assert model == "claude-sonnet-4-5"
    assert thinking is not None


def test_thinking_disabled_when_budget_zero():
    agent = _build_agent(thinking_budget=0)
    model, thinking = agent._select_model_and_thinking(Plan(intent=Intent.STRATEGY))
    assert thinking is None   # budget=0 means feature off


def test_fast_intents_override_via_settings():
    """User can customize which intents route to fast model."""
    agent = _build_agent(fast_intents="CRUD")  # only CRUD, not QUERY
    _, _ = agent._select_model_and_thinking(Plan(intent=Intent.QUERY))
    model, _ = agent._select_model_and_thinking(Plan(intent=Intent.QUERY))
    assert model == "claude-sonnet-4-5"   # not in fast list
    model, _ = agent._select_model_and_thinking(Plan(intent=Intent.CRUD))
    assert model == "claude-haiku-4-5"


def test_empty_fast_intents_disables_haiku_routing():
    agent = _build_agent(fast_intents="")
    model, _ = agent._select_model_and_thinking(Plan(intent=Intent.CRUD))
    assert model == "claude-sonnet-4-5"
