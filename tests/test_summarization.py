from components.summarization.module import SummarizationModule
from shared.session_state import SessionState


def test_not_triggered_below_threshold():
    module = SummarizationModule(trigger_length=20)
    state = SessionState(session_id="s1")
    for i in range(5):
        state.add_turn(f"turn {i}")

    signal = module.process("turn 4", state)

    assert signal.triggered is False
    assert signal.summary == ""
    assert signal.turn_count == 5


def test_triggered_above_threshold_produces_a_summary():
    module = SummarizationModule(trigger_length=3)
    state = SessionState(session_id="s1")
    for i in range(6):
        state.add_turn(f"turn {i}")

    signal = module.process("turn 5", state)

    assert signal.triggered is True
    assert signal.summary != ""
    assert signal.turn_count == 6
