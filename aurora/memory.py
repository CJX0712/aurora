"""Short-term conversation memory with a rolling token budget.

Long-term memory is provided separately by the RAG knowledge base. This module
only manages the working context window fed to the model each turn.
"""

from __future__ import annotations

from typing import List

from .config import AgentConfig
from .types import Message, Role


def _estimate_tokens(text: str) -> int:
    # Cheap, model-agnostic estimate (~4 chars / token for English/code;
    # CJK is denser but this is only used for a soft budget).
    return max(1, len(text) // 4)


class Memory:
    def __init__(self, cfg: AgentConfig, system_prompt: str):
        self.cfg = cfg
        self.system = Message(role=Role.SYSTEM, content=system_prompt)
        self.history: List[Message] = []

    def add(self, msg: Message) -> None:
        self.history.append(msg)

    def reset(self) -> None:
        self.history.clear()

    def window(self) -> List[Message]:
        """Return system + a trimmed history that fits the token budget.

        Never drops the most recent user/assistant turns; trims oldest first.
        """
        budget = self.cfg.context_token_budget
        sys_tokens = _estimate_tokens(self.system.content)
        available = budget - sys_tokens
        out: List[Message] = []
        used = 0
        for m in reversed(self.history):
            cost = _estimate_tokens(m.content) + sum(
                _estimate_tokens(tc.name) + _estimate_tokens(str(tc.arguments)) for tc in m.tool_calls
            )
            if used + cost > available and out:
                break
            out.insert(0, m)
            used += cost
        return [self.system, *out]

    def __len__(self) -> int:
        return len(self.history)
