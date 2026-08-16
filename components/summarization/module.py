"""Component 2 — PLACEHOLDER.

`_fake_llm_summarize` stands in for a single LLM call. It does no real
summarization (just a naive join/truncate) so the whole system runs offline
with no API keys. Swap it for a real LLM call later — the rest of the module
(trigger condition, signal shape) shouldn't need to change.

The running summary itself is persisted in `state.metadata["running_summary"]`
(this module's reserved key — see the interface contract §2's note on the
`metadata` escape hatch).
"""
from __future__ import annotations

import time
from typing import List

from shared.conversation_state import ConversationState
from shared.interfaces import MemoryModule
from shared.signals import MemorySignal

DEFAULT_TRIGGER_LENGTH = 20
RECENT_TURNS_EXCLUDED_FROM_SUMMARY = 5


class SummarizationModule(MemoryModule):
    def __init__(self, trigger_length: int = DEFAULT_TRIGGER_LENGTH):
        self.trigger_length = trigger_length

    def get_context(self, state: ConversationState) -> MemorySignal:
        start = time.perf_counter()

        triggered = state.turn_count > self.trigger_length
        old_turns = [turn["content"] for turn in state.raw_history[:-RECENT_TURNS_EXCLUDED_FROM_SUMMARY]]

        if triggered:
            state.metadata["running_summary"] = self._fake_llm_summarize(old_turns)

        summary = state.metadata.get("running_summary", "")
        source_length = sum(len(turn) for turn in old_turns)
        compression_ratio = len(summary) / source_length if source_length else 0.0

        return MemorySignal(
            strategy_name="summarization",
            context_payload=summary,
            confidence=0.6 if triggered else 0.0,
            token_cost=len(summary.split()),
            latency_ms=(time.perf_counter() - start) * 1000,
            metadata={
                "info_loss_estimate": 1 - compression_ratio if source_length else 0.0,
                "compression_ratio": compression_ratio,
                "summary_level": 1,
                "triggered": triggered,
            },
        )

    def update(self, state: ConversationState, llm_response: str) -> None:
        pass

    @staticmethod
    def _fake_llm_summarize(turns: List[str]) -> str:
        """STUB — not a real LLM call. Replace with an actual single-call
        summarization request later."""
        joined = " | ".join(turns)
        return joined[:200]
