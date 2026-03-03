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

If a single run uses multiple models (e.g., `gpt-4o` for reasoning and `gpt-4o-mini` for summarization), ingest **one event per model**. The `get_usage_metadata_callback()` context manager used inside graph nodes groups token counts by model name automatically, so the hook iterates over its dictionary and emits one event per entry.

---

## Token Tracking — Graph-Level Responsibility

Token tracking is **not** a server-level concern. It is opt-in at the **graph level**. Graphs that want to track token usage do so by using `get_usage_metadata_callback()` from `langchain-core` inside their nodes and writing the result to graph state.

### How It Works

The `react_agent` example demonstrates the pattern:

1. **Graph node uses the callback** (`examples/react_agent/graph.py:42-62`):

```python
from langchain_core.callbacks import get_usage_metadata_callback

async def call_model(state: State, runtime: Runtime[Context]) -> dict:
    model = load_chat_model(runtime.context.model).bind_tools(TOOLS)

    with get_usage_metadata_callback() as cb:
        response = await model.ainvoke([system_message, *state.messages])

    return {"messages": [response], "usage_metadata": dict(cb.usage_metadata)}
```

2. **State schema defines the field** with a reducer (`examples/react_agent/state.py:14-77`):

```python
def _merge_usage_metadata(current: dict | None, update: dict | None) -> dict:
    """Last-write-wins — the callback accumulates cumulatively."""
    if update is not None:
        return update
    return current or {}

@dataclass
class State(InputState):
    usage_metadata: Annotated[dict[str, Any], _merge_usage_metadata] = field(default_factory=dict)
```

3. **Final graph state flows to hooks via `ctx.output`**:
   - When the graph finishes, the final state (including `usage_metadata`) becomes `final_output` in `execute_run_async` (`runs.py:1055-1056`)
   - The `after_run` hook receives it as `ctx.output`
   - Token data is at `ctx.output["usage_metadata"]`

### What the Data Looks Like

After a run, `ctx.output["usage_metadata"]` contains a dict keyed by model name:

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

### Prerequisite: `model_name` in Response Metadata

`get_usage_metadata_callback()` requires both `AIMessage.usage_metadata` and `response_metadata["model_name"]` to be non-None. If either is missing, that LLM call's tokens are silently skipped. `ChatOpenAI`, `ChatAnthropic`, and most major providers populate both.

### Graphs That Don't Track Tokens

Graphs that don't use `get_usage_metadata_callback()` simply won't have `usage_metadata` in their output. The billing hooks handle this gracefully — they check for the key and skip silently if absent.

---

## Implementation

### `hooks.py` — The Billing Hooks File

```python
"""Aegra billing hooks using Polar.sh for usage-based billing.

Token usage data comes from the graph's final output state. Graphs that
track tokens via ``get_usage_metadata_callback()`` write ``usage_metadata``
to their state, which becomes available as ``ctx.output["usage_metadata"]``
in after_run and on_run_error hooks.
"""

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


def _extract_usage_metadata(ctx) -> dict[str, Any] | None:
    """Extract usage_metadata from graph output.

    Token usage is a graph-level concern. Graphs that use
    ``get_usage_metadata_callback()`` write ``usage_metadata`` into their
    state, which becomes the run's final output (``ctx.output``).

    Returns None if the graph didn't track tokens or the output is unavailable.
    """
    if not isinstance(ctx.output, dict):
        return None
    return ctx.output.get("usage_metadata")


@hooks.after_run
async def report_usage(ctx) -> None:
    """Ingest token usage events into Polar.sh after every successful run.

    Reads usage_metadata from the graph's final output state
    (ctx.output["usage_metadata"]).
    """
    if ctx.graph_id not in PAID_AGENTS:
        return

    usage_metadata = _extract_usage_metadata(ctx)
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
            await polar.events.ingest_async(request={"events": events})
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

    Note: partial usage is only available if the graph emitted at least one
    ``values`` event before the error occurred. If the graph failed before
    producing any output, ``ctx.output`` will be None and no usage is reported.
    """
    if ctx.graph_id not in PAID_AGENTS:
        return

    usage_metadata = _extract_usage_metadata(ctx)
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
            await polar.events.ingest_async(request={"events": events})
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
        result = await polar.customer_meters.list_async(
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

When a client disconnects and the run is cancelled, `on_run_error` fires with `error_type="CancelledError"`. The `report_partial_usage` hook ingests whatever tokens were consumed before cancellation — but only if the graph had emitted at least one `values` event before the cancellation occurred. If the graph was cancelled before producing any output, `ctx.output` is `None` and no usage is reported.

### Error Cases

When a run fails with an exception, `on_run_error` fires. Whether `ctx.output` contains `usage_metadata` depends on timing:

- If the graph completed at least one node that wrote to `usage_metadata` and emitted a `values` event, partial usage data will be available
- If the graph failed before any `values` event, `ctx.output` will be `None` or `{}`
- The `_extract_usage_metadata()` helper handles all these cases gracefully

---

## Billing Architecture Summary

```
User sends request
    |
    v
before_run hook (optional: check credit balance via Polar API)
    |
    v
Graph executes
    |--- graph nodes use get_usage_metadata_callback() to track tokens
    |--- graph writes usage_metadata to state
    |--- final state becomes final_output in execute_run_async
    |
    v
after_run hook reads ctx.output["usage_metadata"]
    |--- builds Polar events (one per model)
    |--- calls polar.events.ingest_async()
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
| Token counting per run        | Graph nodes using `get_usage_metadata_callback()`, written to graph state as `usage_metadata` |
| Event ingestion to Polar      | `after_run` / `on_run_error` hooks (read from `ctx.output["usage_metadata"]`) |
| Pre-run credit checks         | `before_run` hook (optional) |
| User identity mapping         | Aegra auth system (`ctx.user.identity`) |

---

## Dependencies

| Package       | Purpose                           | Required? |
| ------------- | --------------------------------- | --------- |
| `polar-sdk`   | Polar API client                  | Yes       |
| `langchain-core` | `get_usage_metadata_callback()` context manager (>= 0.3.49, installed: transitive via `langgraph>=1.0.3`) | Yes (for token tracking in graphs) |

Both should be added to the project's dependencies if not already present. `langchain-core` is available transitively via `langgraph>=1.0.3` but should be listed explicitly if billing is a core feature.
