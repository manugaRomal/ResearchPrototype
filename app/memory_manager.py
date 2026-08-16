"""Conversation Manager & orchestrator. Holds one ConversationState per
conversation_id, in-memory only — restarting the process drops all sessions.

Note: this skeleton doesn't call an LLM (see app/main.py), so
`Controller.notify_response` — which would call `update()` on every module
after the LLM replies — isn't wired up here yet. It's still exercised
directly in tests/test_controller.py.
"""
from __future__ import annotations

from typing import Dict

from components.controller.module import Controller
from components.rag_retrieval.module import RagRetrievalModule
from components.summarization.module import SummarizationModule
from components.sliding_window.module import SlidingWindowModule
from shared.conversation_state import ConversationState
from shared.signals import ControllerDecision


class MemoryManager:
    def __init__(self) -> None:
        self._conversations: Dict[str, ConversationState] = {}
        self.controller = Controller(
            sliding_window=SlidingWindowModule(),
            summarization=SummarizationModule(),
            rag_retrieval=RagRetrievalModule(),
        )

    def handle_turn(self, conversation_id: str, message: str) -> ControllerDecision:
        state = self._get_conversation(conversation_id)
        state.add_user_turn(message)

        return self.controller.decide(state)

    def _get_conversation(self, conversation_id: str) -> ConversationState:
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = ConversationState(conversation_id=conversation_id)
        return self._conversations[conversation_id]
