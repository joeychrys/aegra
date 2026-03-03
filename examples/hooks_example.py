"""Example run lifecycle hooks for Aegra.

Demonstrates how to use before_run, after_run, and on_run_error hooks
to observe, gate, and react to run lifecycle events — including logging
per-model token usage from graphs that track it via
``get_usage_metadata_callback()``.

Configuration:
Add this to your aegra.json:

{
  "hooks": {
    "path": "./examples/hooks_example.py:hooks",
    "timeout": 10
  }
}

Hook types:
- before_run:   Fires before graph execution. Can reject runs via RejectRun.
- after_run:    Fires after successful completion or interrupt. Errors are logged, not propagated.
- on_run_error: Fires when a run fails or is cancelled. Errors are logged, not propagated.

Token usage tracking:
Graphs that use ``get_usage_metadata_callback()`` and write the result into their
state (e.g., the react_agent example) will have ``usage_metadata`` in the final
output. The ``after_run`` hook reads it from ``ctx.output`` and logs per-model
token counts.
"""

from typing import Any

import structlog

from aegra_api.hooks import RunHooks

logger = structlog.get_logger("hooks_example")

hooks = RunHooks()


# ---------------------------------------------------------------------------
# before_run — gate and observe incoming runs
# ---------------------------------------------------------------------------


@hooks.before_run
async def log_run_start(ctx) -> None:
    """Log every incoming run with key context."""
    logger.info(
        "run started",
        run_id=ctx.run_id,
        thread_id=ctx.thread_id,
        assistant_id=ctx.assistant_id,
        graph_id=ctx.graph_id,
        user=ctx.user.identity,
    )


# Uncomment to test run rejection:
#
# @hooks.before_run
# async def require_subscription(ctx) -> None:
#     """Reject runs from users without a subscription tier."""
#     tier = getattr(ctx.user, "subscription_tier", None)
#     if not tier:
#         raise hooks.RejectRun("Subscription required", status_code=402)


# ---------------------------------------------------------------------------
# after_run — react to completed runs
# ---------------------------------------------------------------------------


@hooks.after_run
async def log_run_complete(ctx) -> None:
    """Log run completion with status and token usage.

    Graphs that track token usage via ``get_usage_metadata_callback()`` store
    the result in their state as ``usage_metadata``. Since the final graph
    state becomes ``ctx.output``, we read it from there.
    """
    # Extract usage_metadata from graph output (if the graph tracks it)
    usage: dict[str, Any] | None = None
    if isinstance(ctx.output, dict):
        usage = ctx.output.get("usage_metadata")

    logger.info(
        "run completed",
        run_id=ctx.run_id,
        status=ctx.status,
        graph_id=ctx.graph_id,
        user=ctx.user.identity,
        has_output=ctx.output is not None,
        usage=usage,
    )

    # Log per-model breakdown if available
    if usage:
        for model_name, tokens in usage.items():
            logger.info(
                "token usage",
                run_id=ctx.run_id,
                model=model_name,
                input_tokens=tokens.get("input_tokens", 0),
                output_tokens=tokens.get("output_tokens", 0),
                total_tokens=tokens.get("total_tokens", 0),
            )


# ---------------------------------------------------------------------------
# on_run_error — react to failed runs
# ---------------------------------------------------------------------------


@hooks.on_run_error
async def log_run_error(ctx) -> None:
    """Log run failures with error details."""
    logger.error(
        "run failed",
        run_id=ctx.run_id,
        graph_id=ctx.graph_id,
        user=ctx.user.identity,
        error=ctx.error,
        error_type=ctx.error_type,
    )
