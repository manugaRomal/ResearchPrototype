"""Common interface for the 3 memory components (sliding window, summarization,
RAG retrieval). The controller is intentionally NOT a MemoryModule — it owns
instances of the three and only ever reads their `MemorySignal` output, so it
never needs to know how any signal was computed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from shared.conversation_state import ConversationState
from shared.signals import MemorySignal


class MemoryModule(ABC):
    @abstractmethod
    def get_context(self, state: ConversationState) -> MemorySignal:
        """Called by the controller every turn. Must not mutate `state`
        beyond this module's own reserved `state.metadata` keys, and must
        never raise — on failure, return a MemorySignal with confidence=0.0
        and metadata["error"] set instead.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, state: ConversationState, llm_response: str) -> None:
        """Called after the LLM responds, regardless of whether this module
        was the chosen strategy — every module keeps its internal state
        current even when it wasn't picked.
        """
        raise NotImplementedError
