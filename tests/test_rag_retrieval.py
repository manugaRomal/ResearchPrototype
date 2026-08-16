from components.rag_retrieval.module import RagRetrievalModule
from shared.session_state import SessionState


def test_first_turn_has_nothing_to_retrieve():
    module = RagRetrievalModule(top_k=3)
    state = SessionState(session_id="s1")

    signal = module.process("hello there", state)

    assert signal.retrieved_turns == []
    assert signal.confidence == 0.0


def test_semantically_similar_turn_is_retrieved_with_high_confidence():
    module = RagRetrievalModule(top_k=3)
    state = SessionState(session_id="s1")

    module.process("I love hiking in the mountains", state)
    module.process("What's the weather like today?", state)
    signal = module.process("I really enjoy hiking in the mountains", state)

    assert "I love hiking in the mountains" in signal.retrieved_turns
    assert signal.confidence > 0.5
