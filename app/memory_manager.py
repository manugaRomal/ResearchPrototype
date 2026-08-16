"""Orchestrates a single turn through all 4 components. Holds one SessionState
per session_id, in-memory only — restarting the process drops all sessions.
"""
from __future__ import annotations

from typing import Dict

from components.controller.module import Controller
from components.rag_retrieval.module import RagRetrievalModule
from components.summarization.module import SummarizationModule
from components.sliding_window.module import SlidingWindowModule
from shared.session_state import SessionState
from shared.signals import ControllerDecision


class MemoryManager:
    def __init__(self) -> None:
        self._sessions: Dict[str, SessionState] = {}
        self.sliding_window = SlidingWindowModule()
        self.summarization = SummarizationModule()
        self.rag_retrieval = RagRetrievalModule()
        self.controller = Controller()

    def handle_turn(self, session_id: str, message: str) -> ControllerDecision:
        session_state = self._get_session(session_id)
        session_state.add_turn(message)

        sliding_window_signal = self.sliding_window.process(message, session_state)
        summarization_signal = self.summarization.process(message, session_state)
        retrieval_signal = self.rag_retrieval.process(message, session_state)

        return self.controller.decide(sliding_window_signal, summarization_signal, retrieval_signal)

    def _get_session(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
        return self._sessions[session_id]
