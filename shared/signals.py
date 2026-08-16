"""Shared contract: one Pydantic signal model per memory component, plus the
controller's output. Modules and the controller only ever talk to each other
through these types.
"""
from __future__ import annotations

from typing import List, Union

from pydantic import BaseModel, Field


class SlidingWindowSignal(BaseModel):
    """Output of Component 1 (sliding window)."""

    turns: List[str] = Field(description="The last `window_size` turns, oldest first")
    window_size: int


class SummarizationSignal(BaseModel):
    """Output of Component 2 (summarization)."""

    summary: str = Field(description="Running summary text; empty until first triggered")
    triggered: bool = Field(description="Whether the length threshold was exceeded on this turn")
    turn_count: int = Field(description="Total turns in the session so far")


class RetrievalSignal(BaseModel):
    """Output of Component 3 (RAG retrieval)."""

    retrieved_turns: List[str] = Field(description="Top-K most similar past turns, best first")
    scores: List[float] = Field(description="Cosine similarity score per retrieved turn, same order")
    confidence: float = Field(description="Top similarity score; naive confidence proxy for the controller")


class ControllerDecision(BaseModel):
    """Output of Component 4 (controller). Returned to the caller as-is."""

    strategy: str = Field(description="Selected strategy: 'rag_retrieval' | 'summarization' | 'sliding_window'")
    assembled_context: str = Field(description="Context text assembled from the winning signal")
    reasoning: str = Field(description="Human-readable note on which rule fired")


Signal = Union[SlidingWindowSignal, SummarizationSignal, RetrievalSignal]
