from components.controller.module import Controller
from shared.conversation_state import ConversationState
from shared.interfaces import MemoryModule
from shared.signals import MemorySignal


class _StubModule(MemoryModule):
    """Stands in for a real memory module — see interface contract §7."""

    def __init__(self, strategy_name: str, context_payload: str, confidence: float):
        self.strategy_name = strategy_name
        self.context_payload = context_payload
        self.confidence = confidence

    def get_context(self, state: ConversationState) -> MemorySignal:
        return MemorySignal(
            strategy_name=self.strategy_name,
            context_payload=self.context_payload,
            confidence=self.confidence,
            token_cost=len(self.context_payload.split()),
            latency_ms=0.0,
        )

    def update(self, state: ConversationState, llm_response: str) -> None:
        pass


def _controller(retrieval_confidence: float) -> Controller:
    return Controller(
        sliding_window=_StubModule("sliding_window", "hi", 0.2),
        summarization=_StubModule("summarization", "a running summary", 0.5),
        rag_retrieval=_StubModule("rag_retrieval", "relevant past turn", retrieval_confidence),
    )


def test_picks_rag_retrieval_when_confidence_is_high():
    controller = _controller(retrieval_confidence=0.9)
    state = ConversationState(conversation_id="s1", turn_count=1)

    decision = controller.decide(state)

    assert decision.chosen_signal.strategy_name == "rag_retrieval"
    assert decision.chosen_signal.context_payload == "relevant past turn"


def test_picks_summarization_when_turn_count_high_and_confidence_low():
    controller = _controller(retrieval_confidence=0.1)
    state = ConversationState(conversation_id="s1", turn_count=25)

    decision = controller.decide(state)

    assert decision.chosen_signal.strategy_name == "summarization"
    assert decision.chosen_signal.context_payload == "a running summary"


def test_falls_back_to_sliding_window():
    controller = _controller(retrieval_confidence=0.1)
    state = ConversationState(conversation_id="s1", turn_count=1)

    decision = controller.decide(state)

    assert decision.chosen_signal.strategy_name == "sliding_window"
    assert decision.chosen_signal.context_payload == "hi"


def test_all_candidates_include_every_module():
    controller = _controller(retrieval_confidence=0.9)
    state = ConversationState(conversation_id="s1", turn_count=1)

    decision = controller.decide(state)

    assert set(decision.all_candidates) == {"sliding_window", "summarization", "rag_retrieval"}


def test_switched_and_previous_strategy_track_across_turns():
    controller = _controller(retrieval_confidence=0.9)
    state = ConversationState(conversation_id="s1", turn_count=1)

    first = controller.decide(state)
    second = controller.decide(state)

    assert first.previous_strategy is None
    assert first.switched is False
    assert second.previous_strategy == "rag_retrieval"
    assert second.switched is False
