"""Component 4 — PLACEHOLDER.

Pure rule-based strategy selection: no trained/ML models here, just threshold
checks over the 3 signals. The controller only reads Signal fields — it has
no idea how sliding_window/summarization/rag_retrieval computed them.
"""
from __future__ import annotations

from shared.signals import (
    ControllerDecision,
    RetrievalSignal,
    SlidingWindowSignal,
    SummarizationSignal,
)

RETRIEVAL_CONFIDENCE_THRESHOLD = 0.7
SUMMARIZATION_TURN_THRESHOLD = 20


class Controller:
    def decide(
        self,
        sliding_window_signal: SlidingWindowSignal,
        summarization_signal: SummarizationSignal,
        retrieval_signal: RetrievalSignal,
    ) -> ControllerDecision:
        if retrieval_signal.confidence > RETRIEVAL_CONFIDENCE_THRESHOLD:
            return ControllerDecision(
                strategy="rag_retrieval",
                assembled_context="\n".join(retrieval_signal.retrieved_turns),
                reasoning=(
                    f"retrieval confidence {retrieval_signal.confidence:.2f} "
                    f"> {RETRIEVAL_CONFIDENCE_THRESHOLD}"
                ),
            )

        if summarization_signal.turn_count > SUMMARIZATION_TURN_THRESHOLD:
            return ControllerDecision(
                strategy="summarization",
                assembled_context=summarization_signal.summary,
                reasoning=(
                    f"turn_count {summarization_signal.turn_count} "
                    f"> {SUMMARIZATION_TURN_THRESHOLD}"
                ),
            )

        return ControllerDecision(
            strategy="sliding_window",
            assembled_context="\n".join(sliding_window_signal.turns),
            reasoning="below both the retrieval-confidence and turn-count thresholds",
        )
