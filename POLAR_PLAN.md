# Polar.sh Billing Integration — Implementation Plan

This document describes how to add LLM token-based billing to Aegra using [Polar.sh](https://polar.sh) for usage-based billing. It builds on the run lifecycle hooks system defined in [PLAN.md](./PLAN.md).

## Overview

The billing model:

1. Users subscribe to a plan that includes a monthly token allowance (e.g., 100,000 tokens)
2. Every agent run reports token consumption to Polar after completion
3. Polar deducts from the credit balance first
4. Any usage beyond the credit balance is billed as overage at the end of the billing cycle
5. Optionally, `before_run` hooks can enforce a hard cap by checking the credit balance before execution

## Prerequisites

- Aegra run lifecycle hooks implemented (see [PLAN.md](./PLAN.md))
- A [Polar.sh](https://polar.sh) account with an organization
- The `polar-sdk` Python package installed: `pip install polar-sdk`
- A Polar Organization Access Token with `events:write` and `customer_meters:read` scopes

## Polar Setup

### Step 1: Create a Meter

In the Polar dashboard, create a meter with these settings:

| Setting       | Value                                    |
| ------------- | ---------------------------------------- |
| Name          | `AI Token Usage`                         |
| Filter        | Event name equals `ai_usage`             |
| Aggregation   | Sum of `total_tokens` metadata property  |

This meter will track total token consumption across all events named `ai_usage`.

### Step 2: Create a Subscription Product

Create a subscription product (e.g., "Pro Plan — $29/month") with:

1. **A Metered Price** attached to the `AI Token Usage` meter
   - Set the unit price (e.g., $0.01 per 1,000 tokens beyond the included credits)
   - This is what gets billed as overage at the end of each billing cycle

2. **A Meter Credits Benefit** attached to the same meter
   - Set the credit amount (e.g., 100,000 tokens per month)
   - Credits are granted at the start of each subscription cycle
   - Consumed tokens are deducted from credits first; only excess triggers the metered price

### Step 3: Map Customers

Polar identifies customers by either `customer_id` (Polar's internal ID) or `external_customer_id` (your user ID). The hooks integration uses `external_customer_id` set to `ctx.user.identity` — the authenticated user's identity from Aegra's auth system.

Ensure your users are registered as Polar customers with matching `external_id` values. This can be done:
- Via the Polar dashboard manually
- Via the Polar API (`POST /v1/customers`) during user registration
- Via Polar's checkout flow (customers are created automatically on subscription)

---

## Event Schema

Each event ingested into Polar must contain the fields Polar needs to filter, aggregate, and display usage. Polar has a first-class `_llm` structured metadata key designed for LLM usage events.

### Required Event Structure

```json
{
  "name": "ai_usage",
  "external_customer_id": "<user identity from ctx.user.identity>",
  "metadata": {
    "_llm": {
      "vendor": "<provider name>",
      "model": "<model identifier>",
      "input_tokens": <integer>,
      "output_tokens": <integer>,
      "total_tokens": <integer>
    }
  }
}
```

### `_llm` Metadata Fields

These are defined by Polar's `LLMMetadata` schema:

| Field                | Type    | Required | Description                                          |
| -------------------- | ------- | -------- | ---------------------------------------------------- |
| `vendor`             | string  | Yes      | The LLM provider (e.g., `"openai"`, `"anthropic"`, `"google"`) |
| `model`              | string  | Yes      | The model identifier (e.g., `"gpt-4o-mini"`, `"claude-3-5-haiku-20241022"`) |
| `input_tokens`       | integer | Yes      | Number of input (prompt) tokens consumed              |
| `output_tokens`      | integer | Yes      | Number of output (completion) tokens consumed         |
| `total_tokens`       | integer | Yes      | Total tokens (`input_tokens + output_tokens`)         |
| `cached_input_tokens`| integer | No       | Number of cached input tokens (if provider reports it)|
| `prompt`             | string  | No       | The LLM prompt (for debugging/auditing)               |
| `response`           | string  | No       | The LLM response (for debugging/auditing)             |

### Additional Metadata Fields

You can add up to 50 key-value pairs to `metadata` alongside `_llm`. Useful additions:

| Field          | Type   | Description                                    |
| -------------- | ------ | ---------------------------------------------- |
| `graph_id`     | string | Which agent graph produced this usage          |
| `run_id`       | string | The Aegra run ID for traceability              |
| `thread_id`    | string | The Aegra thread ID for conversation tracking  |
| `assistant_id` | string | The Aegra assistant ID                         |

### Example: Complete Event

```json
{
  "name": "ai_usage",
  "external_customer_id": "user_abc123",
  "metadata": {
    "_llm": {
      "vendor": "openai",
      "model": "gpt-4o-mini-2024-07-18",
      "input_tokens": 1250,
      "output_tokens": 340,
      "total_tokens": 1590
    },
    "graph_id": "research-agent",
    "run_id": "550e8400-e29b-41d4-a716-446655440000",
    "thread_id": "660e8400-e29b-41d4-a716-446655440001"
  }
}
```

### One Event Per Model Per Run

If a single run uses multiple models (e.g., `gpt-4o` for reasoning and `gpt-4o-mini` for summarization), ingest **one event per model**. The `UsageMetadataCallbackHandler` from `langchain-core` groups token counts by model name automatically, so you iterate over its dictionary and emit one event per entry.

---

## Token Tracking with `UsageMetadataCallbackHandler`

Hooks themselves don't track tokens — they are general-purpose (see [PLAN.md](./PLAN.md)). Token tracking is opt-in using LangGraph's `UsageMetadataCallbackHandler` from `langchain-core`.

### How It Works

`langchain-core` provides `get_usage_metadata_callback()`, a context manager that creates a `UsageMetadataCallbackHandler` and auto-injects it into all LLM calls within its scope via `ContextVar`. The handler listens for `on_llm_end` events and accumulates `usage_metadata` from `AIMessage` responses. After the graph finishes, the callback's `usage_metadata` attribute contains a dict keyed by model name:

```python
{
    "gpt-4o-mini-2024-07-18": {
        "input_tokens": 1250,
        "output_tokens": 340,
        "total_tokens": 1590,
        "input_token_details": {"audio": 0, "cache_read": 0},
        "output_token_details": {"audio": 0, "reasoning": 0},
    },
    "claude-3-5-haiku-20241022": {
        "input_tokens": 800,
        "output_tokens": 210,
        "total_tokens": 1010,
        "input_token_details": {"cache_read": 0, "cache_creation": 0},
    },
}
```

### Injecting the Callback

Since hooks can't modify the run config (they are observe-only), token tracking must be enabled at the server level. `langchain-core` (>= 0.3.49) provides `get_usage_metadata_callback()` — a context manager that uses `ContextVar` + `register_configure_hook` to **auto-inject** the callback into every LLM call within its scope. No manual `config["callbacks"]` manipulation is needed.

**Wrap graph execution in `execute_run_async`**

```python
# In execute_run_async, wrap the graph execution block:
from langchain_core.callbacks import get_usage_metadata_callback

with get_usage_metadata_callback() as usage_cb:
    async with langgraph_service.get_graph(graph_id) as graph:
        async for event_type, event_data in stream_graph_events(...):
            ...

# After execution, build extras for RunContext:
run_extras: dict[str, Any] = {}
if usage_cb.usage_metadata:
    run_extras["usage_metadata"] = dict(usage_cb.usage_metadata)

# Pass run_extras as the `extras` kwarg when constructing RunContext
# for after_run and on_run_error hooks
```

How it works under the hood:
1. `get_usage_metadata_callback()` calls `register_configure_hook()` with a `ContextVar` marked `inheritable=True`
2. LangChain's `_configure()` — which runs on every callback manager creation, including inside each LangGraph node — checks the `ContextVar` and auto-injects the handler
3. `ContextVar` values propagate through `await` and `asyncio.create_task()`, so it works correctly in async server code
4. The handler's `on_llm_end` extracts `AIMessage.usage_metadata` and `response_metadata["model_name"]`, accumulating per-model totals

The hooks themselves never import or interact with the callback handler. They simply read `ctx.extras.get("usage_metadata")`. This keeps the hooks system decoupled from `langchain-core`. **No graph code changes are required** — the context manager intercepts all LLM calls transparently.

---

## Implementation

### `hooks.py` — The Billing Hooks File

```python
"""Aegra billing hooks using Polar.sh for usage-based billing."""

import os
from typing import Any

import structlog
from polar_sdk import Polar

from aegra_api.hooks import RunHooks

logger = structlog.get_logger("billing")

hooks = RunHooks()
polar = Polar(access_token=os.environ["POLAR_ACCESS_TOKEN"])

# Agents that require billing. Free agents are excluded.
PAID_AGENTS: set[str] = {"research-agent", "coding-assistant"}

# Map LLM model prefixes to vendor names for Polar's _llm metadata.
VENDOR_MAP: dict[str, str] = {
    "gpt-": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
    "claude-": "anthropic",
    "gemini-": "google",
    "command-": "cohere",
    "mistral-": "mistral",
}


def resolve_vendor(model_name: str) -> str:
    """Resolve LLM vendor from model name prefix."""
    for prefix, vendor in VENDOR_MAP.items():
        if model_name.startswith(prefix):
            return vendor
    return "unknown"


@hooks.after_run
async def report_usage(ctx) -> None:
    """Ingest token usage events into Polar.sh after every successful run.

    Requires UsageMetadataCallbackHandler to be injected into the run config
    at the server level. When injected, the server populates
    ctx.extras["usage_metadata"] with per-model token counts.
    """
    if ctx.graph_id not in PAID_AGENTS:
        return

    # usage_metadata is populated by the server via extras when
    # UsageMetadataCallbackHandler is injected. If not present, skip silently.
    usage_metadata: dict[str, Any] | None = ctx.extras.get("usage_metadata")
    if not usage_metadata:
        return

    events = []
    for model_name, usage in usage_metadata.items():
        events.append({
            "name": "ai_usage",
            "external_customer_id": ctx.user.identity,
            "metadata": {
                "_llm": {
                    "vendor": resolve_vendor(model_name),
                    "model": model_name,
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                "graph_id": ctx.graph_id,
                "run_id": ctx.run_id,
                "thread_id": ctx.thread_id,
            },
        })

    if events:
        try:
            await polar.events.ingest({"events": events})
        except Exception:
            logger.exception(
                "Failed to ingest usage events to Polar",
                run_id=ctx.run_id,
                event_count=len(events),
            )


@hooks.on_run_error
async def report_partial_usage(ctx) -> None:
    """Ingest partial usage for failed runs.

    Even when a run fails, tokens were consumed up to the point of failure.
    Whether to bill for partial usage is a business decision — this hook
    reports it. Remove this hook if you don't want to bill for failed runs.
    """
    if ctx.graph_id not in PAID_AGENTS:
        return

    usage_metadata: dict[str, Any] | None = ctx.extras.get("usage_metadata")
    if not usage_metadata:
        return

    events = []
    for model_name, usage in usage_metadata.items():
        events.append({
            "name": "ai_usage",
            "external_customer_id": ctx.user.identity,
            "metadata": {
                "_llm": {
                    "vendor": resolve_vendor(model_name),
                    "model": model_name,
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                "graph_id": ctx.graph_id,
                "run_id": ctx.run_id,
                "error_type": ctx.error_type,
            },
        })

    if events:
        try:
            await polar.events.ingest({"events": events})
        except Exception:
            logger.exception(
                "Failed to ingest partial usage events to Polar",
                run_id=ctx.run_id,
            )
```

### `aegra.json` Configuration

```json
{
  "graphs": {
    "research-agent": "./agents/research.py:graph",
    "coding-assistant": "./agents/coding.py:graph"
  },
  "hooks": {
    "path": "./hooks.py:hooks",
    "timeout": 10
  }
}
```

### Environment Variables

```bash
# .env
POLAR_ACCESS_TOKEN=polar_at_xxxxxxxxxxxxx
```

---

## Optional: Hard Cap with `before_run`

With overage billing, `before_run` is **not required** — Polar automatically bills excess usage. However, if you want a hard cap (e.g., free tier users get 10,000 tokens/month and nothing more), you must check the balance before each run.

From the Polar docs:

> Polar doesn't block usage if the customer exceeds their balance. You're responsible for implementing the logic you need to prevent usage if they exceed it.

### Adding a Credit Check

```python
@hooks.before_run
async def check_credits(ctx) -> None:
    """Reject runs if the user has exhausted their token credits."""
    if ctx.graph_id not in PAID_AGENTS:
        return

    try:
        # Query Polar for the customer's meter balance
        # API: GET /v1/customer-meters/?external_customer_id=<id>
        result = await polar.customer_meters.list(
            external_customer_id=ctx.user.identity,
        )

        for item in result.items:
            if item.meter.name == "AI Token Usage" and item.balance <= 0:
                raise hooks.RejectRun(
                    "Token credits exhausted. Please upgrade your plan.",
                    status_code=402,
                )
    except hooks.RejectRun:
        raise  # Let RejectRun propagate
    except Exception:
        # If we can't reach Polar, allow the run (fail-open).
        # Change to fail-closed (raise RejectRun) if you prefer strictness.
        logger.warning(
            "Could not check credit balance, allowing run",
            run_id=ctx.run_id,
            user_id=ctx.user.identity,
        )
```

### Important: Token Consumption Is Unpredictable

LLM token consumption is fundamentally unpredictable before execution. A single run can consume anywhere from 100 to 100,000+ tokens depending on the graph, tools called, and input complexity.

The `before_run` check can only answer "does this user have **some** credits left?", not "does this user have **enough** credits for this specific run?" This means:

- A run may push the balance negative
- The **next** run will then be blocked
- This follows a common pattern in usage-based billing: gate access based on current balance, record actual consumption after the fact

---

## How Billing Works Across Interrupts and Cancellations

### Interrupts (Human-in-the-Loop)

Each resume creates a **new run** with its own `run_id`. Token tracking and billing events are per-run:

```
Run 1: user input -> LLM calls -> graph hits interrupt
  after_run fires -> ingest 500 tokens to Polar

Run 2: user sends resume command -> more LLM calls -> success
  after_run fires -> ingest 300 tokens to Polar

Total billed: 800 tokens across 2 events
```

Polar aggregates all `ai_usage` events for the customer regardless of run boundaries.

### Cancellations

When a client disconnects and the run is cancelled, `on_run_error` fires with `error_type="CancelledError"`. The `report_partial_usage` hook ingests whatever tokens were consumed before cancellation.

---

## Billing Architecture Summary

```
User sends request
    |
    v
before_run hook (optional: check credit balance)
    |
    v
with get_usage_metadata_callback() as usage_cb:
    |
    v
    Graph executes (callback auto-injected via ContextVar, tracks tokens)
    |
    v
Server reads usage_cb.usage_metadata -> populates ctx.extras["usage_metadata"]
    |
    v
after_run hook reads ctx.extras["usage_metadata"] -> polar.events.ingest({ai_usage event})
    |
    v
Polar aggregates: credits deducted first, overage billed end-of-month
```

### What Polar Manages

| Concern                       | Handled by    |
| ----------------------------- | ------------- |
| Credit granting (monthly)     | Polar (Meter Credits Benefit) |
| Credit deduction              | Polar (automatic from meter)  |
| Overage calculation           | Polar (metered price)         |
| Invoice generation            | Polar                         |
| Customer portal / usage view  | Polar                         |

### What Aegra Manages

| Concern                       | Handled by    |
| ----------------------------- | ------------- |
| Token counting per run        | `get_usage_metadata_callback()` context manager (langchain-core), surfaced via `ctx.extras["usage_metadata"]` |
| Event ingestion to Polar      | `after_run` / `on_run_error` hooks (read from `ctx.extras`) |
| Pre-run credit checks         | `before_run` hook (optional) |
| User identity mapping         | Aegra auth system (`ctx.user.identity`) |

---

## Dependencies

| Package       | Purpose                           | Required? |
| ------------- | --------------------------------- | --------- |
| `polar-sdk`   | Polar API client                  | Yes       |
| `langchain-core` | `get_usage_metadata_callback()` context manager (>= 0.3.49, installed: transitive via `langgraph>=1.0.3`) | Yes (for token tracking) |

Both should be added to the project's dependencies if not already present. `langchain-core` is available transitively via `langgraph>=1.0.3` but should be listed explicitly if billing is a core feature.
