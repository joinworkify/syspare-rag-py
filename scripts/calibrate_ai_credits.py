"""Measure the arithmetic-mean Vertex cost of representative English chat answers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Local calibration uses the checked-in manual registry/cache and must not require production
# Postgres.
os.environ.setdefault("RAG_USE_LOCAL_MANUAL_REGISTRY", "1")

import rag_server  # noqa: E402
from starlette.requests import Request  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "eval" / "ai_credit_calibration.json"
MONTHLY_COST_BUDGETS = {"individual": 2.40, "team": 6.00, "professional": 15.00}


SCENARIOS: List[Dict[str, Any]] = [
    {"name": "fresh_tire_pressure", "question": "What tire pressure should I use?"},
    {"name": "fresh_engine_oil", "question": "How do I change the engine oil and filter?"},
    {"name": "fresh_fuel_filter", "question": "How do I replace the fuel filter?"},
    {"name": "fresh_starting", "question": "What should I check when the engine will not start?"},
    {"name": "fresh_pto", "question": "How should I safely engage and operate the PTO?"},
    {"name": "fresh_hydraulics", "question": "What should I inspect if the hydraulic lift is slow?"},
    {
        "name": "followup_history_sufficient",
        "question": "Can you summarize those steps in one sentence?",
        "history": [
            {
                "role": "user",
                "content": "How do I check engine oil?",
            },
            {
                "role": "model",
                "content": (
                    "Park on level ground, stop the engine, wait for the oil to settle, "
                    "remove and wipe the dipstick, reinstall it, then check that the level "
                    "is between the marks."
                ),
            },
        ],
    },
    {
        "name": "followup_requires_retrieval",
        "question": "Now tell me the specified rear tire pressure.",
        "history": [
            {"role": "user", "content": "Hello"},
            {"role": "model", "content": "Hello! How can I help with your tractor?"},
        ],
    },
    {
        "name": "followup_history_sufficient_second",
        "question": "Which of those checks should I do first?",
        "history": [
            {"role": "user", "content": "Why will the engine not start?"},
            {
                "role": "model",
                "content": (
                    "First confirm there is fuel, the fuel shutoff is open, the transmission "
                    "is in neutral, and the battery terminals are clean and tight."
                ),
            },
        ],
    },
    {
        "name": "thin_context_expansion_candidate",
        "question": (
            "What is the exact torque specification for the left rear canopy antenna "
            "mounting bolt?"
        ),
    },
]


def _request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/chat",
            "headers": [],
            "client": ("127.0.0.1", 1234),
            "server": ("calibration", 80),
            "scheme": "http",
            "query_string": b"",
        }
    )


def _friendly_quota(budget: float, mean_cost: float) -> int:
    raw = math.floor(budget / mean_cost)
    increment = 50 if raw >= 500 else 10
    return max(increment, (raw // increment) * increment)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rag_server.RATE_LIMIT_ENABLED = False
    scenarios = SCENARIOS[: args.limit] if args.limit else SCENARIOS
    results: List[Dict[str, Any]] = []

    for index, scenario in enumerate(scenarios, start=1):
        print(f"[{index}/{len(scenarios)}] {scenario['name']}", flush=True)
        history = [rag_server.ChatMessage(**item) for item in scenario.get("history", [])]
        payload = rag_server.ChatRequest(
            question=scenario["question"],
            history=history,
            manual_id="YM358_operation",
            answer_language="en",
            log_response=False,
            temp=0.2,
        )
        response = rag_server.api_chat(payload, _request())
        if not isinstance(response, rag_server.ChatResponse):
            raise RuntimeError(f"{scenario['name']} returned {type(response).__name__}")
        usage = response.usage or {}
        if not usage.get("all_generation_metadata_available"):
            raise RuntimeError(f"{scenario['name']} did not return complete usage metadata")
        results.append(
            {
                "name": scenario["name"],
                "question": scenario["question"],
                "history_messages": len(history),
                "answer_chars": len(response.answer),
                "usage": usage,
            }
        )

    costs = [float(result["usage"]["estimated_cost_usd"]) for result in results]
    mean_cost = statistics.fmean(costs)
    retrieval_costs = [
        float(result["usage"]["estimated_cost_usd"])
        for result in results
        if result["usage"]["retrieval_used"]
    ]
    no_retrieval_costs = [
        float(result["usage"]["estimated_cost_usd"])
        for result in results
        if not result["usage"]["retrieval_used"]
    ]
    quotas = {
        tier: _friendly_quota(budget, mean_cost)
        for tier, budget in MONTHLY_COST_BUDGETS.items()
    }
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": rag_server.GEMINI_MODEL,
        "sample_count": len(results),
        "mean_cost_usd": round(mean_cost, 8),
        "median_cost_usd": round(statistics.median(costs), 8),
        "minimum_cost_usd": round(min(costs), 8),
        "maximum_cost_usd": round(max(costs), 8),
        "retrieval_sample_count": len(retrieval_costs),
        "retrieval_mean_cost_usd": (
            round(statistics.fmean(retrieval_costs), 8) if retrieval_costs else None
        ),
        "no_retrieval_sample_count": len(no_retrieval_costs),
        "no_retrieval_mean_cost_usd": (
            round(statistics.fmean(no_retrieval_costs), 8)
            if no_retrieval_costs
            else None
        ),
        "monthly_cost_budgets_usd": MONTHLY_COST_BUDGETS,
        "recommended_monthly_credits": quotas,
        "credit_definition": "one successful English answer",
    }
    payload = {"summary": summary, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
