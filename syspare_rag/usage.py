"""Per-request Vertex usage aggregation for AI credit calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List


GENERATION_PRICES_USD_PER_MILLION = {
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
}
TEXT_EMBEDDING_PRICE_USD_PER_MILLION = 0.10


def _usage_value(metadata: Any, *names: str) -> int:
    for name in names:
        value = getattr(metadata, name, None)
        if value is None and isinstance(metadata, dict):
            value = metadata.get(name)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return 0


@dataclass
class VertexUsageAccumulator:
    model_name: str = "gemini-2.5-flash"
    generation_events: List[Dict[str, Any]] = field(default_factory=list)
    embedding_input_tokens: int = 0
    embedding_calls: int = 0
    retrieval_used: bool = False
    retrieval_expanded: bool = False
    retrieved_context_chars: int = 0

    def record_generation(self, response: Any, operation: str) -> None:
        metadata = getattr(response, "usage_metadata", None)
        prompt = _usage_value(metadata, "prompt_token_count", "promptTokenCount")
        candidates = _usage_value(
            metadata, "candidates_token_count", "candidatesTokenCount"
        )
        total = _usage_value(metadata, "total_token_count", "totalTokenCount")
        reported_thoughts = _usage_value(
            metadata, "thoughts_token_count", "thoughtsTokenCount"
        )
        # Some Vertex SDK versions omit thoughts_token_count even though total_token_count
        # includes billed thinking tokens. Infer the otherwise-unaccounted generated tokens.
        thoughts = max(reported_thoughts, total - prompt - candidates, 0)
        self.generation_events.append(
            {
                "operation": operation,
                "prompt_tokens": prompt,
                "output_tokens": candidates,
                "thinking_tokens": thoughts,
                "total_tokens": total,
                "metadata_available": metadata is not None,
            }
        )

    def record_text_embeddings(self, text: str, *, calls: int = 1) -> None:
        # Vertex's legacy text embedding response does not expose billable usage metadata.
        # Four characters per token is a conservative English estimate; the cost is tiny
        # compared with generation and remains separately identified as estimated.
        estimated_tokens = max(1, math.ceil(len(text or "") / 4))
        self.embedding_input_tokens += estimated_tokens * calls
        self.embedding_calls += calls

    def record_retrieved_context(self, chars: int, *, expanded: bool = False) -> None:
        self.retrieval_used = True
        self.retrieval_expanded = self.retrieval_expanded or expanded
        self.retrieved_context_chars += max(0, chars)

    def to_dict(self) -> Dict[str, Any]:
        prices = GENERATION_PRICES_USD_PER_MILLION.get(
            self.model_name, GENERATION_PRICES_USD_PER_MILLION["gemini-2.5-flash"]
        )
        prompt_tokens = sum(e["prompt_tokens"] for e in self.generation_events)
        output_tokens = sum(e["output_tokens"] for e in self.generation_events)
        thinking_tokens = sum(e["thinking_tokens"] for e in self.generation_events)
        total_tokens = sum(e["total_tokens"] for e in self.generation_events)
        generation_cost = (
            prompt_tokens * prices["input"]
            + (output_tokens + thinking_tokens) * prices["output"]
        ) / 1_000_000
        embedding_cost = (
            self.embedding_input_tokens
            * TEXT_EMBEDDING_PRICE_USD_PER_MILLION
            / 1_000_000
        )
        return {
            "model_name": self.model_name,
            "generation_calls": len(self.generation_events),
            "generation_events": self.generation_events,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": thinking_tokens,
            "total_tokens": total_tokens,
            "embedding_calls": self.embedding_calls,
            "embedding_input_tokens_estimated": self.embedding_input_tokens,
            "retrieval_used": self.retrieval_used,
            "retrieval_expanded": self.retrieval_expanded,
            "retrieved_context_chars": self.retrieved_context_chars,
            "generation_cost_usd": round(generation_cost, 8),
            "embedding_cost_usd_estimated": round(embedding_cost, 8),
            "estimated_cost_usd": round(generation_cost + embedding_cost, 8),
            "all_generation_metadata_available": all(
                e["metadata_available"] for e in self.generation_events
            ),
        }
