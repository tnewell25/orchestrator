"""Rewind on repeated tool failures — drop the failed exchange instead of
piling error messages in context."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from orchestrator.core.agent import Agent


class _FakeBlock:
    def __init__(self, id_, name, input_):
        self.id = id_
        self.name = name
        self.input = input_


def _build_minimal_agent():
    agent = Agent.__new__(Agent)
    return agent


# ---- _error_signatures --------------------------------------------


def test_error_signatures_detects_error_results():
    agent = _build_minimal_agent()
    blocks = [
        _FakeBlock("t1", "deal-find", {"q": "Bosch"}),
        _FakeBlock("t2", "deal-get_context", {"deal_id": "x"}),
    ]
    results = [
        {"tool_use_id": "t1", "content": "Error executing deal-find: DB down"},
        {"tool_use_id": "t2", "content": "{stage: 'proposal'}"},
    ]
    sigs = agent._error_signatures(blocks, results)
    assert len(sigs) == 1
    assert any("deal-find" in s for s in sigs)


def test_error_signatures_same_args_produces_same_sig():
    agent = _build_minimal_agent()
    blocks_a = [_FakeBlock("t1", "deal-find", {"q": "Bosch"})]
    blocks_b = [_FakeBlock("t9", "deal-find", {"q": "Bosch"})]
    err = [{"tool_use_id": "t1", "content": "Error executing x"}]
    err_b = [{"tool_use_id": "t9", "content": "Error executing x"}]

    sig_a = agent._error_signatures(blocks_a, err)
    sig_b = agent._error_signatures(blocks_b, err_b)
    assert sig_a == sig_b  # same tool + args → same signature (id differs, ignored)


def test_error_signatures_different_args_produce_different_sigs():
    agent = _build_minimal_agent()
    a = [_FakeBlock("t1", "deal-find", {"q": "Bosch"})]
    b = [_FakeBlock("t2", "deal-find", {"q": "Honeywell"})]
    err_a = [{"tool_use_id": "t1", "content": "Error executing deal-find: x"}]
    err_b = [{"tool_use_id": "t2", "content": "Error executing deal-find: x"}]
    assert agent._error_signatures(a, err_a) != agent._error_signatures(b, err_b)


def test_error_signatures_empty_on_successful_results():
    agent = _build_minimal_agent()
    blocks = [_FakeBlock("t1", "ok", {})]
    results = [{"tool_use_id": "t1", "content": "success"}]
    assert agent._error_signatures(blocks, results) == set()


def test_error_signatures_recognizes_various_error_prefixes():
    agent = _build_minimal_agent()
    blocks = [
        _FakeBlock("t1", "a", {}),
        _FakeBlock("t2", "b", {}),
    ]
    results = [
        {"tool_use_id": "t1", "content": "Error executing a: boom"},
        {"tool_use_id": "t2", "content": "Error: something else"},
    ]
    sigs = agent._error_signatures(blocks, results)
    assert len(sigs) == 2
