"""Component 3 — PLACEHOLDER.

Embeds each turn with sentence-transformers (all-MiniLM-L6-v2) and stores
vectors in an in-memory FAISS IndexFlatIP, per conversation. Retrieval is
plain top-K cosine similarity (via normalized inner product); the top score
is used directly as "confidence". No decay, no granularity logic yet —
replace later.

The FAISS index and its parallel turn-text list are persisted in
`state.metadata["faiss_index"]` / `state.metadata["indexed_turns"]` (this
module's reserved keys — FAISS itself only stores vectors, not text).
"""
from __future__ import annotations

import time
from typing import List

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

from shared.conversation_state import ConversationState
from shared.interfaces import MemoryModule
from shared.signals import MemorySignal

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DEFAULT_TOP_K = 3


class RagRetrievalModule(MemoryModule):
    def __init__(self, top_k: int = DEFAULT_TOP_K):
        self.top_k = top_k
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def get_context(self, state: ConversationState) -> MemorySignal:
        start = time.perf_counter()

        faiss_index = state.metadata.get("faiss_index")
        if faiss_index is None:
            faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
            state.metadata["faiss_index"] = faiss_index
        indexed_turns: List[str] = state.metadata.setdefault("indexed_turns", [])

        query = state.current_user_message
        query_vector = self._embed(query)

        retrieved_turns: List[str] = []
        scores: List[float] = []
        if faiss_index.ntotal > 0:
            k = min(self.top_k, faiss_index.ntotal)
            similarities, row_indices = faiss_index.search(query_vector, k)
            for score, row in zip(similarities[0], row_indices[0]):
                if row == -1:
                    continue
                retrieved_turns.append(indexed_turns[row])
                scores.append(float(score))

        # Index the current turn AFTER searching, so it can't match itself.
        faiss_index.add(query_vector)
        indexed_turns.append(query)

        confidence = scores[0] if scores else 0.0
        context_payload = "\n".join(retrieved_turns)

        return MemorySignal(
            strategy_name="rag_retrieval",
            context_payload=context_payload,
            confidence=confidence,
            token_cost=len(context_payload.split()),
            latency_ms=(time.perf_counter() - start) * 1000,
            metadata={
                "relevance_score": confidence,
                "retrieval_k": self.top_k,
                "temporal_decay_applied": False,
            },
        )

    def update(self, state: ConversationState, llm_response: str) -> None:
        pass

    def _embed(self, text: str) -> np.ndarray:
        vector = self.model.encode([text], normalize_embeddings=True)
        return np.asarray(vector, dtype="float32")
