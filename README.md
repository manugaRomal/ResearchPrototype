# Memory Middleware — Prototype Skeleton

An adaptive memory middleware for multi-turn LLM conversations. This is a
**structural skeleton only**: every component below is a naive placeholder
meant to prove the architecture connects end-to-end. None of them contain
the real research logic yet.

This structure follows the shared interface contract (J26-DS-354): a single
`MemorySignal` shape returned by every memory module, so the controller never
needs to know how any signal was computed.

## Architecture

```
POST /chat  →  MemoryManager.handle_turn(conversation_id, message)
                 │
                 ├─ state.add_user_turn(message)   → ConversationState
                 │
                 └─ Controller.decide(state)        → ControllerDecision  (returned as JSON)
                      │
                      ├─ SlidingWindowModule.get_context   → MemorySignal
                      ├─ SummarizationModule.get_context    → MemorySignal
                      ├─ RagRetrievalModule.get_context      → MemorySignal
                      │  (called concurrently via a thread pool)
                      │
                      └─ picks ONE MemorySignal as chosen_signal
```

- **shared/signals.py** — the shared contract: `MemorySignal` (one shape for
  all 3 components' output — `strategy_name`, `context_payload`, `confidence`,
  `token_cost`, `latency_ms`, plus a module-specific `metadata` dict), and
  `ControllerDecision` (the controller's output).
- **shared/interfaces.py** — `MemoryModule`, the abstract base the 3 memory
  components implement: `get_context(state) -> MemorySignal` and
  `update(state, llm_response) -> None`. The controller is *not* a
  `MemoryModule` — it owns instances of the three and only ever reads their
  `MemorySignal` output, so it has zero knowledge of how any signal was
  computed.
- **shared/conversation_state.py** — `ConversationState`: turn history
  (`raw_history`), the current user message, task-type/token-budget fields,
  and a `metadata` escape hatch each module uses to persist its own internal
  state across turns (e.g. RAG's FAISS index, the running summary). One
  instance per `conversation_id`, created lazily and held in an in-memory
  dict on `MemoryManager`. Nothing is persisted to disk; restarting the
  process drops all sessions.
- **components/** — the 4 pluggable components, each independently importable
  and testable without the others running. `Controller` takes the other three
  as constructor dependencies, calls `get_context` on all of them concurrently
  each turn, and applies threshold rules over their `MemorySignal.confidence`
  to pick one.
- **app/memory_manager.py** — orchestrates: appends the turn to
  `ConversationState`, hands the state to the controller.
- **app/main.py** — FastAPI app exposing `POST /chat`.

## Component status (all placeholders — replace internals later)

| # | Component | Current (naive) logic |
|---|---|---|
| 1 | Sliding window | Fixed-size slice of the last N turns; `confidence` = fraction of the window that's filled |
| 2 | Summarization | **Stubbed** — no real LLM call. `_fake_llm_summarize` just joins/truncates old turns once turn count exceeds a threshold. Swap this one function for a real LLM call later. |
| 3 | RAG retrieval | Embeds each turn with `sentence-transformers` (`all-MiniLM-L6-v2`), stores vectors in a per-conversation in-memory FAISS `IndexFlatIP`, does plain top-K cosine retrieval. Top score is used directly as `confidence` — no decay, no granularity logic. |
| 4 | Controller | Pure rule-based, no ML: `if rag.confidence > 0.7: use rag; elif turn_count > 20: use summarization; else: use sliding window` |

## Setup

```bash
pip install -r requirements.txt
```

First run of the RAG component downloads the `all-MiniLM-L6-v2` model from
Hugging Face (~90 MB) — needs internet access once, then it's cached locally.

## Run the API

```bash
cd memory-middleware
uvicorn app.main:app --reload
```

Example call:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "demo-session", "message": "Hi, I love hiking in the mountains."}'
```

Example response:

```json
{
  "chosen_signal": {
    "strategy_name": "sliding_window",
    "context_payload": "Hi, I love hiking in the mountains.",
    "confidence": 0.2,
    "token_cost": 7,
    "latency_ms": 0.03,
    "metadata": {"freshness_score": 0.2, "window_size_tokens": 5, "messages_included": 1}
  },
  "all_candidates": { "sliding_window": {"...": "..."}, "summarization": {"...": "..."}, "rag_retrieval": {"...": "..."} },
  "rationale": "below both the retrieval-confidence and turn-count thresholds",
  "switched": false,
  "previous_strategy": null,
  "was_override": false,
  "timestamp": "2026-08-16T12:00:00+00:00"
}
```

Send more turns with the same `conversation_id` to see the strategy change
once turn count passes 20, or once a later message is semantically close
enough (cosine similarity > 0.7) to an earlier one.

## Run the tests

```bash
cd memory-middleware
pytest
```

Each `tests/test_*.py` file imports only the one component it tests (plus
`shared/`) — none of them spin up `MemoryManager` or the FastAPI app, so each
component is verified in isolation with hand-built fake inputs.
`test_controller.py` uses stub `MemoryModule` implementations (per the
contract's §7 mock-module pattern) instead of the real sliding
window/summarization/RAG modules.
