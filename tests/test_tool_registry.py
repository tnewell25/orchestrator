"""ToolRegistry search/lookup + lazy Agent integration."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from orchestrator.core.agent import Agent
from orchestrator.core.planner import Plan
from orchestrator.core.skill_base import Skill, tool
from orchestrator.core.tool_registry import (
    DEFAULT_ESSENTIALS,
    ToolRegistry,
    tool_search_schema,
)


# Fixture skills resembling the real catalog
class _DealSkill(Skill):
    name = "deal"
    @tool("Find deals by name or stage filter.")
    async def find(self, query: str = "") -> list:
        return []
    @tool("Create a new deal in the pipeline.")
    async def create(self, name: str) -> dict:
        return {"id": "x"}


class _ContactSkill(Skill):
    name = "contact"
    @tool("Find contacts by name, email, or title.")
    async def find(self, query: str) -> list:
        return []
    @tool("Update personal_notes for a contact.")
    async def update(self, contact_id: str, personal_notes: str = "") -> dict:
        return {"ok": True}


class _GraphSkill(Skill):
    name = "graph"
    @tool("Get a subgraph slice rooted at one entity.")
    async def context_for(self, entity_type: str, entity_id: str) -> dict:
        return {}


def _build_skills():
    return [_DealSkill(), _ContactSkill(), _GraphSkill()]


# ---- ToolRegistry --------------------------------------------


def test_registry_indexes_all_tools():
    reg = ToolRegistry(_build_skills())
    names = set(reg.all_names())
    assert {"deal-find", "deal-create", "contact-find", "contact-update", "graph-context_for"} <= names


def test_registry_get_schema_returns_copy():
    reg = ToolRegistry(_build_skills())
    schema = reg.get_schema("deal-find")
    assert schema is not None
    assert schema["name"] == "deal-find"
    # Mutating the copy shouldn't affect future calls
    schema["name"] = "mutated"
    again = reg.get_schema("deal-find")
    assert again["name"] == "deal-find"


def test_registry_search_finds_by_keyword():
    reg = ToolRegistry(_build_skills())
    results = reg.search("personal notes", limit=3)
    names = [r["name"] for r in results]
    assert "contact-update" in names


def test_registry_search_ranks_substring_matches_higher():
    reg = ToolRegistry(_build_skills())
    results = reg.search("deal", limit=3)
    # Both deal-find and deal-create should appear; substring match boosts them
    deal_names = [r["name"] for r in results if r["name"].startswith("deal")]
    assert len(deal_names) == 2


def test_registry_search_empty_query_returns_empty():
    reg = ToolRegistry(_build_skills())
    assert reg.search("", limit=5) == []


# ---- Agent lazy mode -----------------------------------------


def _build_agent(lazy: bool, essentials=None):
    skills = _build_skills()
    agent = Agent.__new__(Agent)
    agent.memory = MagicMock()
    agent.skills = {s.name: s for s in skills}
    agent.settings = SimpleNamespace(default_model="x", agent_name="Test")
    agent.token_manager = None
    agent.audit_logger = None
    agent.max_iterations = 5
    agent.planner = None
    agent.entity_extractor = None
    agent.lazy_tools = lazy
    agent.essentials = essentials or ("contact-find", "deal-find")
    agent.registry = ToolRegistry(skills)
    agent.tools = []
    agent.tool_map = {}
    for s in skills:
        for sch in s.get_tools():
            agent.tools.append(sch)
            agent.tool_map[sch["name"]] = s
    return agent


def test_eager_mode_loads_all_tools():
    agent = _build_agent(lazy=False)
    initial = agent._initial_active_tools(plan=None)
    # All 5 tools, no tool-search meta
    assert len(initial) == 5
    assert not any(t["name"] == "tool-search" for t in initial)


def test_lazy_mode_loads_essentials_plus_search():
    agent = _build_agent(lazy=True, essentials=("contact-find", "deal-find"))
    initial = agent._initial_active_tools(plan=None)
    names = {t["name"] for t in initial}
    assert "contact-find" in names
    assert "deal-find" in names
    assert "tool-search" in names
    # contact-update / deal-create should NOT be loaded yet
    assert "contact-update" not in names
    assert "deal-create" not in names


def test_lazy_mode_with_planner_preloads_suggested():
    agent = _build_agent(lazy=True, essentials=("contact-find",))
    plan = Plan(suggested_tools=["deal-create", "graph-context_for"])
    initial = agent._initial_active_tools(plan=plan)
    names = {t["name"] for t in initial}
    assert {"contact-find", "tool-search", "deal-create", "graph-context_for"} <= names


def test_handle_tool_search_appends_new_tools():
    agent = _build_agent(lazy=True, essentials=("contact-find",))
    active = agent._initial_active_tools(plan=None)
    loaded = {t["name"] for t in active}
    initial_count = len(active)

    result_str = agent._handle_tool_search(
        {"query": "create deal"}, active, loaded,
    )
    result = json.loads(result_str)

    assert result["active_count"] > initial_count
    assert "deal-create" in {t["name"] for t in active}
    assert "deal-create" in result["newly_loaded"]


def test_handle_tool_search_idempotent_for_already_loaded():
    agent = _build_agent(lazy=True, essentials=("deal-find",))
    active = agent._initial_active_tools(plan=None)
    loaded = {t["name"] for t in active}

    result_str = agent._handle_tool_search({"query": "deal"}, active, loaded)
    result = json.loads(result_str)
    # deal-find is already loaded — should NOT appear in newly_loaded
    assert "deal-find" not in result["newly_loaded"]
    # deal-create might be loaded though
    assert "deal-create" in {t["name"] for t in active}
