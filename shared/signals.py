"""Shared contract: one `MemorySignal` shape returned by every memory module,
plus the controller's `ControllerDecision` output. Modules and the controller
only ever talk to each other through these two types — the controller never
touches a module's internals, only `MemorySignal` fields.

See the interface contract §3, §4, §6 for the field-by-field rationale and the
per-module `metadata` key conventions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemorySignal:
    strategy_name: str
    context_payload: str
    confidence: float
    token_cost: int
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ControllerDecision:
    chosen_signal: MemorySignal
    all_candidates: dict[str, MemorySignal]
    rationale: str
    switched: bool
    previous_strategy: str | None
    was_override: bool
    timestamp: str
