"""Lazy tool registry — fetch tool schemas on demand instead of registering all.

Why: the orchestrator currently has ~45 tool schemas across 18 skills, totaling
roughly 6k tokens injected into every agent call. Most turns use 1-3 tools.

Solution (mirrors Claude Code's ToolSearch): always load a small "essentials"
set + a `tool-search` meta-tool. The agent calls tool-search when it needs
something not in the essentials, gets the schemas back inline, and uses them
in the next iteration.

Cache impact: the essentials block is stable across calls and stays cached.
Newly loaded schemas don't cache, but they're rare.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# Tools always loaded — picked because they appear in nearly every conversation
# and give the agent something useful before any tool-search call.
DEFAULT_ESSENTIALS = (
    "graph-context_for",
    "graph-neighbors",
    "contact-find",
    "deal-find",
    "company-find",
)


@dataclass(frozen=True)
class ToolEntry:
    name: str
    description: str
    schema: dict
    skill_name: str
    keywords: frozenset[str]  # tokens used for search ranking


class ToolRegistry:
    """Holds the full catalog of available tool schemas and indexes them
    for fast substring + keyword search."""

    def __init__(self, skills):
        self._entries: dict[str, ToolEntry] = {}
        for skill in skills:
            for schema in skill.get_tools():
                name = schema["name"]
                self._entries[name] = ToolEntry(
                    name=name,
                    description=schema.get("description", ""),
                    schema=schema,
                    skill_name=skill.name,
                    keywords=_tokenize(name + " " + schema.get("description", "")),
                )

    def all_names(self) -> list[str]:
        return list(self._entries.keys())

    def get_schema(self, name: str) -> dict | None:
        e = self._entries.get(name)
        return dict(e.schema) if e else None

    def get_schemas(self, names) -> list[dict]:
        out = []
        for n in names:
            e = self._entries.get(n)
            if e:
                out.append(dict(e.schema))
        return out

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Score each tool by token overlap + name substring match."""
        if not query.strip():
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        ranked = []
        for entry in self._entries.values():
            overlap = len(q_tokens & entry.keywords)
            substring_hit = 1 if any(t in entry.name.lower() for t in q_tokens) else 0
            score = overlap + 1.5 * substring_hit
            if score > 0:
                ranked.append((score, entry))

        ranked.sort(key=lambda r: (-r[0], r[1].name))
        return [
            {"name": e.name, "description": e.description, "input_schema": e.schema["input_schema"]}
            for _, e in ranked[:limit]
        ]


_TOKEN_RE = re.compile(r"[a-zA-Z]+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "with",
    "by", "is", "be", "at", "this", "that",
}


def _tokenize(s: str) -> frozenset[str]:
    if not s:
        return frozenset()
    return frozenset(
        t.lower() for t in _TOKEN_RE.findall(s)
        if len(t) > 1 and t.lower() not in _STOP
    )


# ----------------------------------------------------------------------
# Tool-search meta-skill — the only tool ALWAYS exposed in lazy mode.
# Implemented inline (not as a Skill subclass) because it talks directly to
# the Agent's mutable active_tools list — see Agent._handle_tool_search.
# ----------------------------------------------------------------------


def tool_search_schema() -> dict:
    """The schema for the meta-tool the agent uses to discover other tools."""
    return {
        "name": "tool-search",
        "description": (
            "Find and load schemas for additional tools. Use when you need a "
            "capability not in the currently-loaded tool list. Pass keywords "
            "describing what you want to do (e.g. 'log meeting', 'send email', "
            "'set reminder'). Returns matching tool schemas which you can then "
            "call directly in the next turn."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords describing the capability needed.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of tools to return (default 5).",
                },
            },
            "required": ["query"],
        },
    }


__all__ = [
    "ToolRegistry",
    "ToolEntry",
    "DEFAULT_ESSENTIALS",
    "tool_search_schema",
]
