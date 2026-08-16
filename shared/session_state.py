"""Per-session state. One SessionState instance is created per session_id and
lives only in process memory — nothing here is persisted to disk.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class SessionState:
    session_id: str

    # Full turn history for the session (shared by all modules).
    turns: List[str] = field(default_factory=list)

    # Component 2 (summarization) writes/reads this.
    running_summary: str = ""

    # Component 3 (RAG retrieval) owns these two: a FAISS index plus a
    # parallel list mapping each vector row back to its source turn text
    # (FAISS itself only stores vectors, not text).
    faiss_index: Optional[Any] = None
    indexed_turns: List[str] = field(default_factory=list)

    def add_turn(self, turn: str) -> None:
        self.turns.append(turn)

    @property
    def turn_count(self) -> int:
        return len(self.turns)
