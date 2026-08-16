"""Component 2 — PLACEHOLDER.

`_fake_llm_summarize` stands in for a single LLM call. It does no real
summarization (just a naive join/truncate) so the whole system runs offline
with no API keys. Swap it for a real LLM call later — the rest of the module
(trigger condition, signal shape) shouldn't need to change.
"""
from __future__ import annotations

from typing import List

from shared.interfaces import MemoryModule
from shared.session_state import SessionState
from shared.signals import SummarizationSignal

DEFAULT_TRIGGER_LENGTH = 20
RECENT_TURNS_EXCLUDED_FROM_SUMMARY = 5


class SummarizationModule(MemoryModule):
    def __init__(self, trigger_length: int = DEFAULT_TRIGGER_LENGTH):
        self.trigger_length = trigger_length

    def process(self, turn: str, session_state: SessionState) -> SummarizationSignal:
        turn_count = session_state.turn_count
        triggered = turn_count > self.trigger_length

        if triggered:
            old_turns = session_state.turns[:-RECENT_TURNS_EXCLUDED_FROM_SUMMARY]
            session_state.running_summary = self._fake_llm_summarize(old_turns)

        return SummarizationSignal(
            summary=session_state.running_summary,
            triggered=triggered,
            turn_count=turn_count,
        )

    @staticmethod
    def _fake_llm_summarize(turns: List[str]) -> str:
        """STUB — not a real LLM call. Replace with an actual single-call
        summarization request later."""
        joined = " | ".join(turns)
        return joined[:200]
