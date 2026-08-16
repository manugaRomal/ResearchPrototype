from components.sliding_window.module import SlidingWindowModule
from shared.conversation_state import ConversationState


def _state_with_turns(*turns: str) -> ConversationState:
    state = ConversationState(conversation_id="s1")
    for turn in turns:
        state.add_user_turn(turn)
    return state


def test_returns_only_last_window_size_turns():
    module = SlidingWindowModule(window_size=3)
    state = _state_with_turns("t1", "t2", "t3", "t4", "t5")

    signal = module.get_context(state)

    assert signal.metadata["messages_included"] == 3
    assert signal.context_payload == "t3\nt4\nt5"


def test_history_shorter_than_window_returns_all_turns():
    module = SlidingWindowModule(window_size=5)
    state = _state_with_turns("only turn")

    signal = module.get_context(state)

    assert signal.metadata["messages_included"] == 1
    assert signal.context_payload == "only turn"
