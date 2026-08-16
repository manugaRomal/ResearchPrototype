from components.rag_retrieval.module import RagRetrievalModule
from shared.conversation_state import ConversationState


def test_first_turn_has_nothing_to_retrieve():
    module = RagRetrievalModule(top_k=3)
    state = ConversationState(conversation_id="s1")
    state.current_user_message = "hello there"

    signal = module.get_context(state)

    assert signal.context_payload == ""
    assert signal.confidence == 0.0


def test_semantically_similar_turn_is_retrieved_with_high_confidence():
    module = RagRetrievalModule(top_k=3)
    state = ConversationState(conversation_id="s1")

    state.current_user_message = "I love hiking in the mountains"
    module.get_context(state)

    state.current_user_message = "What's the weather like today?"
    module.get_context(state)

    state.current_user_message = "I really enjoy hiking in the mountains"
    signal = module.get_context(state)

    assert "I love hiking in the mountains" in signal.context_payload
    assert signal.confidence > 0.5
