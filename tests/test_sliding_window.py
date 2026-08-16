from components.sliding_window.module import SlidingWindowModule
from shared.session_state import SessionState


def test_returns_only_last_window_size_turns():
    module = SlidingWindowModule(window_size=3)
    state = SessionState(session_id="s1")
    for turn in ["t1", "t2", "t3", "t4", "t5"]:
        state.add_turn(turn)

    signal = module.process("t5", state)

    assert signal.window_size == 3
    assert signal.turns == ["t3", "t4", "t5"]


def test_history_shorter_than_window_returns_all_turns():
    module = SlidingWindowModule(window_size=5)
    state = SessionState(session_id="s1")
    state.add_turn("only turn")

    signal = module.process("only turn", state)

    assert signal.turns == ["only turn"]
