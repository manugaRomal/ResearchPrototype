"""Component 4 — PLACEHOLDER.

Pure rule-based strategy selection: no trained/ML models here, just threshold
checks over the 3 signals. The controller only reads MemorySignal fields — it
has no idea how sliding_window/summarization/rag_retrieval computed them.

Per the interface contract §8, the controller owns calling `get_context` on
all three modules concurrently (a thread pool here, since the modules are
synchronous), and owns calling `update` on all three after the LLM responds.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from shared.conversation_state import ConversationState
from shared.interfaces import MemoryModule
from shared.signals import ControllerDecision, MemorySignal

RAG_CONFIDENCE_THRESHOLD = 0.7
SUMMARIZATION_TURN_THRESHOLD = 20


class Controller:
    def __init__(
        self,
        sliding_window: MemoryModule,
        summarization: MemoryModule,
        rag_retrieval: MemoryModule,
    ):
        self.modules: dict[str, MemoryModule] = {
            "sliding_window": sliding_window,
            "summarization": summarization,
            "rag_retrieval": rag_retrieval,
        }
        self._previous_strategy: str | None = None

    def decide(self, state: ConversationState) -> ControllerDecision:
        all_candidates = self._gather_candidates(state)
        rag = all_candidates["rag_retrieval"]
        summarization = all_candidates["summarization"]
        sliding_window = all_candidates["sliding_window"]

        if rag.confidence > RAG_CONFIDENCE_THRESHOLD:
            chosen = rag
            rationale = (
                f"retrieval confidence {rag.confidence:.2f} > {RAG_CONFIDENCE_THRESHOLD}"
            )
        elif state.turn_count > SUMMARIZATION_TURN_THRESHOLD:
            chosen = summarization
            rationale = (
                f"turn_count {state.turn_count} > {SUMMARIZATION_TURN_THRESHOLD}"
            )
        else:
            chosen = sliding_window
            rationale = "below both the retrieval-confidence and turn-count thresholds"

        switched = self._previous_strategy is not None and self._previous_strategy != chosen.strategy_name
        decision = ControllerDecision(
            chosen_signal=chosen,
            all_candidates=all_candidates,
            rationale=rationale,
            switched=switched,
            previous_strategy=self._previous_strategy,
            was_override=False,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._previous_strategy = chosen.strategy_name
        return decision

    def notify_response(self, state: ConversationState, llm_response: str) -> None:
        for module in self.modules.values():
            module.update(state, llm_response)

    def _gather_candidates(self, state: ConversationState) -> dict[str, MemorySignal]:
        with ThreadPoolExecutor(max_workers=len(self.modules)) as pool:
            futures = {name: pool.submit(module.get_context, state) for name, module in self.modules.items()}
            return {name: future.result() for name, future in futures.items()}
