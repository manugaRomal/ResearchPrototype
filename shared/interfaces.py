"""Common interface for the 3 memory components (sliding window, summarization,
RAG retrieval). The controller is intentionally NOT a MemoryModule — it
consumes their Signal outputs instead of session_state directly, so it never
needs to know how any signal was computed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from shared.session_state import SessionState
from shared.signals import Signal


class MemoryModule(ABC):
    @abstractmethod
    def process(self, turn: str, session_state: SessionState) -> Signal:
        """Consume the latest turn (already appended to session_state.turns)
        and return this module's signal for the controller to read.
        """
        raise NotImplementedError
