"""FastAPI entrypoint. Run with: uvicorn app.main:app --reload

POST /chat runs a turn through all 4 memory components and returns the
controller's decision as JSON. It does NOT call an LLM to generate a reply —
that's out of scope for this skeleton.
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from app.memory_manager import MemoryManager
from shared.signals import ControllerDecision

app = FastAPI(title="Memory Middleware Prototype")
memory_manager = MemoryManager()


class ChatRequest(BaseModel):
    conversation_id: str
    message: str


@app.post("/chat", response_model=ControllerDecision)
def chat(request: ChatRequest) -> ControllerDecision:
    return memory_manager.handle_turn(request.conversation_id, request.message)
