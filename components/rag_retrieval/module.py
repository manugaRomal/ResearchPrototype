"""Component 3 — PLACEHOLDER.

Embeds each turn with sentence-transformers (all-MiniLM-L6-v2) and stores
vectors in an in-memory FAISS IndexFlatIP, per session. Retrieval is plain
top-K cosine similarity (via normalized inner product); the top score is used
directly as "confidence". No decay, no granularity logic yet — replace later.
"""
from __future__ import annotations

from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from shared.interfaces import MemoryModule
from shared.session_state import SessionState
from shared.signals import RetrievalSignal

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DEFAULT_TOP_K = 3


class RagRetrievalModule(MemoryModule):
    def __init__(self, top_k: int = DEFAULT_TOP_K):
        self.top_k = top_k
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def process(self, turn: str, session_state: SessionState) -> RetrievalSignal:
        if session_state.faiss_index is None:
            session_state.faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)

        query_vector = self._embed(turn)

        retrieved_turns: List[str] = []
        scores: List[float] = []
        if session_state.faiss_index.ntotal > 0:
            k = min(self.top_k, session_state.faiss_index.ntotal)
            similarities, row_indices = session_state.faiss_index.search(query_vector, k)
            for score, row in zip(similarities[0], row_indices[0]):
                if row == -1:
                    continue
                retrieved_turns.append(session_state.indexed_turns[row])
                scores.append(float(score))

        # Index the current turn AFTER searching, so it can't match itself.
        session_state.faiss_index.add(query_vector)
        session_state.indexed_turns.append(turn)

        confidence = scores[0] if scores else 0.0
        return RetrievalSignal(retrieved_turns=retrieved_turns, scores=scores, confidence=confidence)

    def _embed(self, text: str) -> np.ndarray:
        vector = self.model.encode([text], normalize_embeddings=True)
        return np.asarray(vector, dtype="float32")
