"""Component 1 — PLACEHOLDER.

Naive fixed-size window over the most recent turns. No token-awareness,
no importance weighting. Replace with real logic later.
"""
from __future__ import annotations

import time

from shared.conversation_state import ConversationState
from shared.interfaces import MemoryModule
from shared.signals import MemorySignal

DEFAULT_WINDOW_SIZE = 5


class SlidingWindowModule(MemoryModule):
    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE):
        self.window_size = window_size

    def get_context(self, state: ConversationState) -> MemorySignal:
        start = time.perf_counter()

        window = state.raw_history[-self.window_size :]
        messages_included = len(window)
        context_payload = "\n".join(turn["content"] for turn in window)
        freshness_score = messages_included / self.window_size if self.window_size else 0.0

        return MemorySignal(
            strategy_name="sliding_window",
            context_payload=context_payload,
            confidence=freshness_score,
            token_cost=len(context_payload.split()),
            latency_ms=(time.perf_counter() - start) * 1000,
            metadata={
                "freshness_score": freshness_score,
                "window_size_tokens": self.window_size,
                "messages_included": messages_included,
            },
        )

    def update(self, state: ConversationState, llm_response: str) -> None:
        pass
