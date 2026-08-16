from components.controller.module import Controller
from shared.signals import RetrievalSignal, SlidingWindowSignal, SummarizationSignal


def _fake_signals(retrieval_confidence: float, turn_count: int):
    sliding_window_signal = SlidingWindowSignal(turns=["hi"], window_size=5)
    summarization_signal = SummarizationSignal(
        summary="a running summary", triggered=True, turn_count=turn_count
    )
    retrieval_signal = RetrievalSignal(
        retrieved_turns=["relevant past turn"],
        scores=[retrieval_confidence],
        confidence=retrieval_confidence,
    )
    return sliding_window_signal, summarization_signal, retrieval_signal


def test_picks_rag_retrieval_when_confidence_is_high():
    controller = Controller()
    sw, summ, rag = _fake_signals(retrieval_confidence=0.9, turn_count=1)

    decision = controller.decide(sw, summ, rag)

    assert decision.strategy == "rag_retrieval"
    assert decision.assembled_context == "relevant past turn"


def test_picks_summarization_when_turn_count_high_and_confidence_low():
    controller = Controller()
    sw, summ, rag = _fake_signals(retrieval_confidence=0.1, turn_count=25)

    decision = controller.decide(sw, summ, rag)

    assert decision.strategy == "summarization"
    assert decision.assembled_context == "a running summary"


def test_falls_back_to_sliding_window():
    controller = Controller()
    sw, summ, rag = _fake_signals(retrieval_confidence=0.1, turn_count=1)

    decision = controller.decide(sw, summ, rag)

    assert decision.strategy == "sliding_window"
    assert decision.assembled_context == "hi"
