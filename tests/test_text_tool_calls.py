"""Tests for the text-based tool-call recovery path.

Local GGUF models (Qwen2.5 without a tool-calling chat template) often emit
tool calls as literal ``<tool_call>{...}</tool_call>`` text instead of the
structured ``tool_calls`` field. Aurora must recover those, or the agent loop
silently does nothing (the original smoke-test failure).
"""

from __future__ import annotations

from aurora.llm import _extract_text_tool_calls
from aurora.types import ToolSpec


def _specs(*names):
    return [
        ToolSpec(name=n, description="d", parameters={"type": "object", "properties": {}})
        for n in names
    ]


def test_recovers_single_text_tool_call():
    content = (
        "Sure, writing the file now.\n"
        '<tool_call>\n{"name": "write_file", '
        '"arguments": {"path": "demo.txt", "content": "Aurora is alive"}}\n</tool_call>'
    )
    calls = _extract_text_tool_calls(content, _specs("write_file", "read_file"))
    assert len(calls) == 1
    assert calls[0].name == "write_file"
    assert calls[0].arguments == {"path": "demo.txt", "content": "Aurora is alive"}


def test_recovers_multiple_text_tool_calls():
    content = (
        '<tool_call>{"name": "write_file", "arguments": {"path": "a.txt", "content": "x"}}</tool_call>\n'
        '<tool_call>{"name": "read_file", "arguments": {"path": "a.txt"}}</tool_call>'
    )
    calls = _extract_text_tool_calls(content, _specs("write_file", "read_file"))
    assert [c.name for c in calls] == ["write_file", "read_file"]


def test_tolerates_doubled_braces():
    """Some templates wrap the JSON object in an extra brace pair."""
    content = '<tool_call>{{"name": "read_file", "arguments": {"path": "a.txt"}}}</tool_call>'
    calls = _extract_text_tool_calls(content, _specs("read_file"))
    assert len(calls) == 1
    assert calls[0].arguments == {"path": "a.txt"}


def test_ignores_unknown_tool_names():
    content = '<tool_call>{"name": "delete_everything", "arguments": {}}</tool_call>'
    assert _extract_text_tool_calls(content, _specs("read_file")) == []


def test_no_tool_call_returns_empty():
    assert _extract_text_tool_calls("just a normal answer", _specs("read_file")) == []
    assert _extract_text_tool_calls("", _specs("read_file")) == []


def test_malformed_json_is_skipped():
    content = '<tool_call>{"name": "read_file", "arguments": {not json}}</tool_call>'
    assert _extract_text_tool_calls(content, _specs("read_file")) == []
