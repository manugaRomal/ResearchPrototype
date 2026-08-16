"""Per-conversation state, owned by the Conversation Manager (MemoryManager).
One instance is created per conversation_id and lives only in process memory —
nothing here is persisted to disk.

`metadata` is the escape hatch for anything a module needs to persist across
turns that isn't part of the shared schema (e.g. RAG's FAISS index, the
summarization module's running summary) — see the interface contract, §2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationState:
    conversation_id: str
    turn_count: int = 0
    token_budget_total: int = 8192
    token_budget_used: int = 0
    task_type: str = "casual"
    task_type_confidence: float = 0.0
    raw_history: list[dict[str, Any]] = field(default_factory=list)
    current_user_message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_user_turn(self, message: str) -> None:
        self.turn_count += 1
        self.current_user_message = message
        self.raw_history.append({"role": "user", "content": message, "turn": self.turn_count})
