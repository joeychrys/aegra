"""Polar.sh billing hooks for Aegra.

Usage-based billing via Polar.sh. Reports LLM token consumption after
every run by reading ``usage_metadata`` from the graph's final output state.

Graphs that track tokens (like ``react_agent``) use
``get_usage_metadata_callback()`` in their nodes and write the result to
graph state.  After the run completes, the final state is available as
``ctx.output`` in hooks — token data lives at ``ctx.output["usage_metadata"]``.

Uses the Polar SDK's ``Ingestion`` helper which handles batching and
background sending without blocking the main thread.

Configuration:
Add this to your aegra.json:

{
  "hooks": {
    "path": "./examples/polar_billing_hooks.py:hooks",
    "timeout": 10
  }
}

Environment variables:
  POLAR_ACCESS_TOKEN  - Organization access token with events:write scope
  POLAR_SERVER        - "production" (default) or "sandbox"

Polar dashboard setup:
  1. Create a meter: name="AI Token Usage", filter event name == "ai_usage",
     aggregation = sum of "total_tokens"
  2. Create a subscription product with a metered price on that meter
  3. Optionally add a Meter Credits Benefit for included monthly tokens
  4. Map customers via external_customer_id = ctx.user.identity

See POLAR_PLAN.md for the full integration plan.
"""

import os
from typing import Any

import structlog

from aegra_api.hooks import RunHooks

logger = structlog.get_logger("polar_billing")

hooks = RunHooks()

# --- Polar Ingestion client (lazy) ---

_ingestion = None


def _get_ingestion():
    """Get or create the Polar Ingestion singleton.

    The Ingestion helper handles batching and background sending via
    a daemon thread — it won't block hook execution.

    Returns None if POLAR_ACCESS_TOKEN is not set, which puts the
    hooks into dry-run mode (log only, no ingestion).
    """
    global _ingestion
    if _ingestion is None:
        token = os.environ.get("POLAR_ACCESS_TOKEN")
        if token:
            from polar_sdk.ingestion import Ingestion

            server = os.environ.get("POLAR_SERVER", "production")
            _ingestion = Ingestion(access_token=token, server=server)
    return _ingestion


# --- Configuration ---

# Agents that require billing. Add your paid graph IDs here.
# Graphs not in this set are free and skip billing entirely.
PAID_AGENTS: set[str] = {"agent"}

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


# --- Helpers ---


def _resolve_vendor(model_name: str) -> str:
    """Resolve LLM vendor from model name prefix."""
    for prefix, vendor in VENDOR_MAP.items():
        if model_name.startswith(prefix):
            return vendor
    return "unknown"


def _extract_usage_metadata(ctx: Any) -> dict[str, Any] | None:
    """Extract usage_metadata from graph output.

    Graphs that use ``get_usage_metadata_callback()`` write
    ``usage_metadata`` into their state.  The final state becomes
    ``ctx.output`` in hooks.

    Returns None if the graph didn't track tokens or the output
    is unavailable (e.g., error before any values event).
    """
    if not isinstance(ctx.output, dict):
        return None
    usage = ctx.output.get("usage_metadata")
    if not usage:
        return None
    return usage


def _is_billable_user(ctx: Any) -> bool:
    """Check if the user is a real (non-anonymous) authenticated user."""
    user_id = ctx.user.identity
    return bool(user_id) and user_id != "anonymous"


def _ingest_usage(
    ctx: Any,
    usage_metadata: dict[str, Any],
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """Build and ingest Polar events from per-model usage data.

    One event per model per run. Uses flat metadata keys matching the
    Polar SDK convention (``request_tokens``, ``response_tokens``,
    ``total_tokens``).  The ``_llm`` structured key is included for
    Polar's LLM-specific display features.
    """
    ingestion = _get_ingestion()

    for model_name, usage in usage_metadata.items():
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        # Skip if no tokens to report
        if total_tokens == 0 and input_tokens == 0 and output_tokens == 0:
            continue

        vendor = _resolve_vendor(model_name)

        # Flat metadata keys for meter aggregation (matches Polar SDK conventions)
        metadata: dict[str, Any] = {
            "request_tokens": input_tokens,
            "response_tokens": output_tokens,
            "total_tokens": total_tokens,
            "requests": 1,
            "model": model_name,
            "vendor": vendor,
            "graph_id": ctx.graph_id,
            "run_id": ctx.run_id,
            "thread_id": ctx.thread_id,
            # Structured _llm key for Polar's LLM display features
            "_llm": {
                "vendor": vendor,
                "model": model_name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        event = {
            "name": "ai_usage",
            "external_customer_id": ctx.user.identity,
            "metadata": metadata,
        }

        # Log the event being sent
        logger.info(
            "polar ingest event",
            run_id=ctx.run_id,
            user=ctx.user.identity,
            model=model_name,
            vendor=vendor,
            request_tokens=input_tokens,
            response_tokens=output_tokens,
            total_tokens=total_tokens,
            dry_run=ingestion is None,
        )

        if ingestion:
            ingestion.ingest(event)


# --- Hooks ---


@hooks.after_run
async def report_usage(ctx: Any) -> None:
    """Ingest token usage events into Polar after successful runs.

    Reads ``usage_metadata`` from the graph's final output state.
    Skips silently if the graph doesn't track tokens, isn't a paid agent,
    or the user is anonymous.
    """
    if ctx.graph_id not in PAID_AGENTS:
        return

    if not _is_billable_user(ctx):
        logger.debug("Skipping usage tracking for anonymous user", run_id=ctx.run_id)
        return

    usage_metadata = _extract_usage_metadata(ctx)
    if not usage_metadata:
        return

    _ingest_usage(ctx, usage_metadata)


@hooks.on_run_error
async def report_partial_usage(ctx: Any) -> None:
    """Ingest partial token usage for failed or cancelled runs.

    Tokens were consumed up to the point of failure. Whether to bill for
    partial usage is a business decision — this hook reports it.

    Note: partial usage is only available if the graph emitted at least
    one ``values`` event before the error.  If the graph failed before
    producing any output, ``ctx.output`` is None and nothing is reported.
    """
    if ctx.graph_id not in PAID_AGENTS:
        return

    if not _is_billable_user(ctx):
        return

    usage_metadata = _extract_usage_metadata(ctx)
    if not usage_metadata:
        return

    _ingest_usage(
        ctx,
        usage_metadata,
        extra_metadata={"error_type": ctx.error_type or "unknown"},
    )


# --- Optional: Hard cap with before_run ---
# Uncomment this hook to enforce a hard cap on token credits.
# When enabled, users with zero or negative credit balance are blocked
# from running paid agents.
#
# @hooks.before_run
# async def check_credits(ctx: Any) -> None:
#     """Reject runs if the user has exhausted their token credits."""
#     if ctx.graph_id not in PAID_AGENTS:
#         return
#
#     client = _get_ingestion()
#     if not client:
#         return
#
#     try:
#         result = await polar.customer_meters.list_async(
#             external_customer_id=ctx.user.identity,
#         )
#         for item in result.items:
#             if item.meter.name == "AI Token Usage" and item.balance <= 0:
#                 raise hooks.RejectRun(
#                     "Token credits exhausted. Please upgrade your plan.",
#                     status_code=402,
#                 )
#     except hooks.RejectRun:
#         raise
#     except Exception:
#         # Fail-open: if we can't reach Polar, allow the run.
#         logger.warning(
#             "Could not check credit balance, allowing run",
#             run_id=ctx.run_id,
#             user_id=ctx.user.identity,
#         )
