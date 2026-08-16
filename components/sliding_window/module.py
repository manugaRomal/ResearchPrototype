"""Component 1 — PLACEHOLDER.

Naive fixed-size window over the most recent turns. No token-awareness,
no importance weighting. Replace with real logic later.
"""
from __future__ import annotations

from shared.interfaces import MemoryModule
from shared.session_state import SessionState
from shared.signals import SlidingWindowSignal

DEFAULT_WINDOW_SIZE = 5


class SlidingWindowModule(MemoryModule):
    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE):
        self.window_size = window_size

    def process(self, turn: str, session_state: SessionState) -> SlidingWindowSignal:
        window = session_state.turns[-self.window_size :]
        return SlidingWindowSignal(turns=window, window_size=self.window_size)
