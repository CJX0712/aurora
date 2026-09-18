"""Core data types shared across Aurora modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    role: Role
    content: str = ""
    # For tool messages: id of the tool call this is responding to.
    tool_call_id: Optional[str] = None
    # For assistant messages that request tool calls.
    tool_calls: List["ToolCall"] = field(default_factory=list)
    name: Optional[str] = None

    def to_openai(self) -> Dict[str, Any]:
        msg: Dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
            if self.name:
                msg["name"] = self.name
        if self.tool_calls:
            msg["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
        return msg


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]

    def to_openai(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": _dumps(self.arguments)},
        }


@dataclass
class ToolSpec:
    """Declarative description of a tool, fed to the model as a JSON schema."""

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON-schema object

    def to_openai(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolResult:
    ok: bool
    output: str
    # Optional structured payload (e.g. retrieved docs) for the UI.
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Step:
    """One observable step of the agent loop, for traces / UI."""

    kind: str  # "think" | "tool" | "observe" | "final"
    text: str
    tool: Optional[str] = None
    ok: Optional[bool] = None


def _dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)
