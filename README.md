# Memory Middleware — Prototype Skeleton

An adaptive memory middleware for multi-turn LLM conversations. This is a
**structural skeleton only**: every component below is a naive placeholder
meant to prove the architecture connects end-to-end. None of them contain
the real research logic yet.

## Architecture

```
POST /chat  →  MemoryManager.handle_turn(session_id, message)
                 │
                 ├─ SlidingWindowModule.process   → SlidingWindowSignal
                 ├─ SummarizationModule.process    → SummarizationSignal
                 ├─ RagRetrievalModule.process      → RetrievalSignal
                 │
                 └─ Controller.decide(signals)     → ControllerDecision  (returned as JSON)
```

- **shared/signals.py** — the shared contract. One Pydantic model per
  component's output, plus `ControllerDecision`.
- **shared/interfaces.py** — `MemoryModule`, the one-method abstract base
  class the 3 memory components implement (`process(turn, session_state) -> Signal`).
  The controller is *not* a `MemoryModule` — it only reads the 3 signals, never
  `session_state` directly, so it has zero knowledge of how any signal was computed.
- **shared/session_state.py** — `SessionState`: turn history, running summary,
  and the per-session FAISS index. One instance per `session_id`, created
  lazily and held in an in-memory dict on `MemoryManager`. Nothing is
  persisted to disk; restarting the process drops all sessions.
- **components/** — the 4 pluggable components, each independently importable
  and testable without the others running.
- **app/memory_manager.py** — orchestrates: appends the turn, calls the 3
  memory modules, hands their signals to the controller.
- **app/main.py** — FastAPI app exposing `POST /chat`.

## Component status (all placeholders — replace internals later)

| # | Component | Current (naive) logic |
|---|---|---|
| 1 | Sliding window | Fixed-size slice of the last N turns |
| 2 | Summarization | **Stubbed** — no real LLM call. `_fake_llm_summarize` just joins/truncates old turns once turn count exceeds a threshold. Swap this one function for a real LLM call later. |
| 3 | RAG retrieval | Embeds each turn with `sentence-transformers` (`all-MiniLM-L6-v2`), stores vectors in a per-session in-memory FAISS `IndexFlatIP`, does plain top-K cosine retrieval. Top score is used directly as "confidence" — no decay, no granularity logic. |
| 4 | Controller | Pure rule-based, no ML: `if retrieval.confidence > 0.7: use RAG; elif turn_count > 20: use summarization; else: use sliding window` |

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
  -d '{"session_id": "demo-session", "message": "Hi, I love hiking in the mountains."}'
```

Example response:

```json
{
  "strategy": "sliding_window",
  "assembled_context": "Hi, I love hiking in the mountains.",
  "reasoning": "below both the retrieval-confidence and turn-count thresholds"
}
```

Send more turns with the same `session_id` to see the strategy change once
turn count passes 20, or once a later message is semantically close enough
(cosine similarity > 0.7) to an earlier one.

## Run the tests

```bash
cd memory-middleware
pytest
```

Each `tests/test_*.py` file imports only the one component it tests (plus
`shared/`) — none of them spin up `MemoryManager` or the FastAPI app, so each
component is verified in isolation with hand-built fake inputs.
