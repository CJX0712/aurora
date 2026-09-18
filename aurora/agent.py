"""The autonomous agent core: a planning→act→observe→reflect ReAct loop.

The loop is intentionally backend-agnostic - it only needs a ``ChatModel`` and
a ``Toolbox``. A ``FakeChatModel`` can drive it offline for tests.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .config import AgentConfig
from .exceptions import MaxStepsExceeded
from .llm import ChatModel
from .memory import Memory
from .tools import Toolbox, ToolResult
from .types import Message, Role, Step


SYSTEM_PROMPT = (
    "You are Aurora, a fully local autonomous agent running on the user's own hardware. "
    "You solve tasks by reasoning and using tools. Follow this discipline:\n"
    "1. Think step by step about what the user wants.\n"
    "2. When you need information or must act, call exactly one tool with precise arguments.\n"
    "3. After a tool returns, reflect on the result and decide the next step.\n"
    "4. When the task is fully done, reply with a concise final answer in the user's language.\n"
    "Never invent tool results. If a tool fails, adapt or report the blocker honestly. "
    "Prefer search_knowledge before guessing facts."
)


@dataclass
class AgentResult:
    answer: str
    steps: List[Step] = field(default_factory=list)
    tool_calls: int = 0
    elapsed: float = 0.0


class Agent:
    def __init__(
        self,
        cfg: AgentConfig,
        chat: ChatModel,
        toolbox: Toolbox,
        memory: Optional[Memory] = None,
        on_step: Optional[Callable[[Step], None]] = None,
    ):
        self.cfg = cfg
        self.chat = chat
        self.toolbox = toolbox
        self.memory = memory or Memory(cfg, SYSTEM_PROMPT)
        self.on_step = on_step

    def _emit(self, step: Step) -> None:
        if self.on_step:
            self.on_step(step)

    def run(self, user_input: str, max_steps: Optional[int] = None) -> AgentResult:
        max_steps = max_steps or self.cfg.max_steps
        self.memory.add(Message(role=Role.USER, content=user_input))
        steps: List[Step] = []
        tool_calls = 0
        start = time.time()

        for _ in range(max_steps):
            window = self.memory.window()
            tools = self.toolbox.specs()
            resp = self.chat.chat(window, tools, temperature=self.cfg.temperature, max_tokens=1024)

            # The model may answer directly without tools.
            if not resp.tool_calls:
                final = (resp.content or "").strip()
                if final:
                    self.memory.add(Message(role=Role.ASSISTANT, content=final))
                    self._emit(Step(kind="final", text=final))
                    steps.append(Step(kind="final", text=final))
                    return AgentResult(answer=final, steps=steps, tool_calls=tool_calls, elapsed=time.time() - start)
                # No content and no tool call: nudge the model.
                self.memory.add(
                    Message(role=Role.ASSISTANT, content="(I should call a tool or answer.)")
                )
                continue

            # Execute each requested tool call.
            assistant_msg = Message(role=Role.ASSISTANT, content=resp.content or "", tool_calls=resp.tool_calls)
            self.memory.add(assistant_msg)

            for call in resp.tool_calls:
                tool_calls += 1
                self._emit(Step(kind="tool", text=f"{call.name}({json.dumps(call.arguments, ensure_ascii=False)})", tool=call.name))
                steps.append(Step(kind="tool", text=f"{call.name}", tool=call.name))
                result: ToolResult = self.toolbox.dispatch(call.name, call.arguments)
                obs = result.output[:6000]
                self._emit(Step(kind="observe", text=obs[:500], tool=call.name, ok=result.ok))
                steps.append(Step(kind="observe", text=obs[:500], tool=call.name, ok=result.ok))
                self.memory.add(
                    Message(role=Role.TOOL, content=obs, tool_call_id=call.id, name=call.name)
                )

        # Step budget exhausted: ask the model for a best-effort summary.
        summary = self._summarize()
        if not summary:
            raise MaxStepsExceeded("agent hit step budget without a final answer")
        return AgentResult(answer=summary, steps=steps, tool_calls=tool_calls, elapsed=time.time() - start)

    def _summarize(self) -> str:
        self.memory.add(
            Message(role=Role.USER, content="You have used all tool steps. Provide a final answer now based on what you know.")
        )
        resp = self.chat.chat(self.memory.window(), [], temperature=0.3, max_tokens=800)
        return (resp.content or "").strip()
