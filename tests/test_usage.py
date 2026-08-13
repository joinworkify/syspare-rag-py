from types import SimpleNamespace

from syspare_rag.usage import VertexUsageAccumulator


def _response(prompt=1000, output=200, thinking=50, total=1250):
    return SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt,
            candidates_token_count=output,
            thoughts_token_count=thinking,
            total_token_count=total,
        )
    )


def test_usage_accumulates_generation_and_retrieval_costs():
    usage = VertexUsageAccumulator()
    usage.record_generation(_response(), "answer_generation")
    usage.record_generation(
        _response(prompt=500, output=10, thinking=0, total=510),
        "retrieval_decision",
    )
    usage.record_text_embeddings("engine oil pressure", calls=2)
    usage.record_retrieved_context(10_000)

    result = usage.to_dict()

    assert result["generation_calls"] == 2
    assert result["prompt_tokens"] == 1500
    assert result["output_tokens"] == 210
    assert result["thinking_tokens"] == 50
    assert result["embedding_calls"] == 2
    assert result["retrieval_used"] is True
    assert result["retrieval_expanded"] is False
    assert result["estimated_cost_usd"] == 0.001101


def test_usage_marks_expanded_retrieval_and_missing_metadata():
    usage = VertexUsageAccumulator()
    usage.record_generation(SimpleNamespace(usage_metadata=None), "answer_generation")
    usage.record_retrieved_context(5000, expanded=True)

    result = usage.to_dict()

    assert result["retrieval_expanded"] is True
    assert result["all_generation_metadata_available"] is False
    assert result["estimated_cost_usd"] == 0
