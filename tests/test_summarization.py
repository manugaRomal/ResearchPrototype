from components.summarization.module import SummarizationModule
from shared.conversation_state import ConversationState


def _state_with_turns(count: int) -> ConversationState:
    state = ConversationState(conversation_id="s1")
    for i in range(count):
        state.add_user_turn(f"turn {i}")
    return state


def test_not_triggered_below_threshold():
    module = SummarizationModule(trigger_length=20)
    state = _state_with_turns(5)

    signal = module.get_context(state)

    assert signal.metadata["triggered"] is False
    assert signal.context_payload == ""


def test_triggered_above_threshold_produces_a_summary():
    module = SummarizationModule(trigger_length=3)
    state = _state_with_turns(6)

    signal = module.get_context(state)

    assert signal.metadata["triggered"] is True
    assert signal.context_payload != ""
