# Run Lifecycle Hooks — Implementation Plan

## Problem

There is no way for users to hook into the run execution lifecycle at the server level. This blocks use cases like:

- **Per-agent gating** — some agents free, some require a subscription
- **Audit logging** — recording run start/end/error with full context
- **Usage reporting** — reporting run completions to external analytics or billing systems
- **Error alerting** — sending notifications when runs fail
- **Rate limiting** — throttling runs per user or per agent

### Why LangGraph callbacks alone are insufficient

LangGraph's `config={"callbacks": [...]}` mechanism handles **graph-level** instrumentation (e.g., per-node tracing), but it cannot:

1. **Gate runs before execution** — callbacks fire during execution, not before. There is no clean way to return a 429 to the HTTP client from inside a callback.
2. **Fire exactly once on completion** — `on_chain_end` fires for every node in the graph, not just the top-level run.
3. **Access server-level context** — callbacks don't receive user identity, thread_id, or run_id without smuggling them through config metadata.

---

## Proposal

Add a `hooks` config key to `aegra.json` that lets users register async callbacks for run lifecycle events. Hooks are general-purpose — they can observe, gate, and react to runs without modifying the graph or its config.

### Configuration

```json
{
  "graphs": { "agent": "./agent.py:graph" },
  "hooks": {
    "path": "./hooks.py:hooks",
    "timeout": 10
  }
}
```

- `path` — import path for the `RunHooks` instance (same format as `auth.path` and `http.app`)
- `timeout` — max seconds a hook can run before being killed (default: 10s)

### Hook Points

| Hook           | When it fires                              | Can block?              | Usage                                    |
| -------------- | ------------------------------------------ | ----------------------- | ---------------------------------------- |
| `before_run`   | After validation, before graph execution   | Yes (raise `RejectRun`) | Subscription checks, rate limiting       |
| `after_run`    | After run completes (success or interrupt)  | No (errors logged)      | Usage reporting, audit logging, analytics |
| `on_run_error` | After run fails with an exception           | No (errors logged)      | Error alerting, incident logging         |

---

## User-Facing API

### Example: Audit Logging

```python
# hooks.py
import structlog
from aegra_api.hooks import RunHooks

hooks = RunHooks()
logger = structlog.get_logger("audit")

@hooks.before_run
async def log_run_start(ctx):
    """Log every run attempt with user and agent context."""
    logger.info(
        "run_started",
        run_id=ctx.run_id,
        thread_id=ctx.thread_id,
        graph_id=ctx.graph_id,
        user_id=ctx.user.identity,
    )

@hooks.after_run
async def log_run_complete(ctx):
    """Log run completion with status."""
    logger.info(
        "run_completed",
        run_id=ctx.run_id,
        graph_id=ctx.graph_id,
        user_id=ctx.user.identity,
        status=ctx.status,
    )

@hooks.on_run_error
async def log_run_error(ctx):
    """Log run failures for alerting."""
    logger.error(
        "run_failed",
        run_id=ctx.run_id,
        graph_id=ctx.graph_id,
        user_id=ctx.user.identity,
        error=ctx.error,
        error_type=ctx.error_type,
    )
```

### Example: Subscription Gating

`before_run` hooks can **reject runs** by raising `RejectRun`. This is the gating mechanism.

```python
from aegra_api.hooks import RunHooks

hooks = RunHooks()
PAID_AGENTS = {"research-agent", "coding-assistant"}

@hooks.before_run
async def validate_subscription(ctx):
    """Check that the user has an active subscription before running paid agents."""
    if ctx.graph_id not in PAID_AGENTS:
        return

    has_access = await check_user_subscription(ctx.user.identity)
    if not has_access:
        raise hooks.RejectRun("Active subscription required", status_code=402)
```

### Example: Composing Multiple Hooks

Multiple hooks per event are supported and run in registration order. This lets you compose independent concerns:

```python
hooks = RunHooks()

@hooks.before_run
async def rate_limit(ctx):
    """Enforce per-user rate limits."""
    if await is_rate_limited(ctx.user.identity):
        raise hooks.RejectRun("Rate limit exceeded", status_code=429)

@hooks.after_run
async def report_to_analytics(ctx):
    """Send run metadata to analytics backend."""
    await analytics.track("run_completed", {
        "user_id": ctx.user.identity,
        "graph_id": ctx.graph_id,
        "status": ctx.status,
    })

@hooks.after_run
async def notify_on_completion(ctx):
    """Send webhook notification on run completion."""
    await webhook.send(event="run.completed", payload={
        "run_id": ctx.run_id,
        "thread_id": ctx.thread_id,
        "status": ctx.status,
    })
```

---

## RunContext Dataclass

```python
@dataclass(frozen=True)
class RunContext:
    """Immutable context passed to all run hook callbacks.

    All hooks receive the same fields. Fields that are only meaningful for
    specific hook points are None/empty when not applicable.
    """
    run_id: str
    thread_id: str
    assistant_id: str
    graph_id: str
    user: User                     # Full User object (identity, email, org_id, permissions, extras)
    config: dict[str, Any]         # Read-only snapshot of run config
    input: dict[str, Any] | None = None

    # Populated only in after_run:
    output: dict[str, Any] | None = None
    status: str | None = None      # "success" | "interrupted"

    # Populated only in on_run_error:
    error: str | None = None
    error_type: str | None = None  # e.g., "ValueError", "CancelledError"

    # Server-populated extensible data. Hooks read from this dict for data
    # that the server collects during execution (e.g., token usage, timing).
    # Keys are conventional strings — see docs for available keys.
    extras: dict[str, Any]         # default: {} (set via field default_factory)
```

#### `extras` — Extensible Server Data

The `extras` dict is populated by the server before passing `RunContext` to hooks. It provides a generic extension point so that hooks can access data collected during execution without requiring schema changes to `RunContext` for every new use case.

**Currently defined keys:**

| Key                | Type                          | When populated                  | Description                                                  |
| ------------------ | ----------------------------- | ------------------------------- | ------------------------------------------------------------ |
| `"usage_metadata"` | `dict[str, dict[str, int]]`   | `after_run`, `on_run_error`     | Per-model token counts from `get_usage_metadata_callback()`. Keyed by model name, values contain `input_tokens`, `output_tokens`, `total_tokens`, plus optional detail breakdowns. Only present if token tracking is enabled in `execute_run_async`. |

Users can also populate custom keys by modifying server code. The `extras` dict is part of the frozen dataclass but the dict itself is mutable (standard Python frozen dataclass behavior — the reference is frozen, not the contents). However, hooks **should not** mutate `extras`; it is intended as read-only data from the server.

---

## Hook Behavior Rules

| Rule                                                                        | Rationale                                                           |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `before_run` errors propagate (via `RejectRun`)                             | This is the gating mechanism                                        |
| `before_run` is observe-only (cannot modify config or return values)        | Hooks observe and gate — they don't mutate server state or run config |
| `after_run` errors are logged, not propagated                               | Run already succeeded — don't fail it because a hook failed          |
| `on_run_error` errors are logged, not propagated                            | Run already failed — don't mask the original error                  |
| Multiple hooks per event allowed, run in registration order                 | Composing concerns (logging + gating + analytics)                   |
| `RunContext` is frozen                                                      | Hooks observe, they don't mutate server state                       |
| All hook calls are wrapped in `asyncio.wait_for()` with configurable timeout | Prevents hanging external calls from blocking the server            |
| Hooks must NOT use the shared DB session                                    | Hooks that need DB access must create their own session             |

---

## How Interrupts Work

Each resume creates a **new run** with its own `run_id` and call to `execute_run_async`. Hooks fire per-run:

```
Run 1: user input -> LLM calls -> graph hits interrupt
  before_run fires
  after_run fires (status="interrupted")

Run 2: user sends resume command -> more LLM calls -> success
  before_run fires
  after_run fires (status="success")
```

Users who need to correlate across interrupt legs can aggregate by `thread_id` on their side.

## How Cancellation Works

When a client disconnects and `on_disconnect="cancel"`, the graph is forcibly stopped via `asyncio.CancelledError`. The `on_run_error` hook fires with `error_type="CancelledError"`.

---

## Coverage

All run creation paths flow through `execute_run_async` in `runs.py`:

- `POST /threads/{thread_id}/runs` -> `create_run()` -> `execute_run_async()`
- `POST /threads/{thread_id}/runs/stream` -> `create_and_stream_run()` -> `execute_run_async()`
- `POST /threads/{thread_id}/runs/wait` -> `wait_for_run()` -> `execute_run_async()`
- `POST /runs/*` (stateless) -> delegates to the above -> `execute_run_async()`

One function, all paths covered.

---

## Detailed Design

### New file: `src/aegra_api/core/hooks.py` (~150 lines)

```python
"""Run lifecycle hooks for aegra-api."""

import asyncio
import importlib
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import structlog

from aegra_api.models.auth import User

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RunContext:
    """Immutable context passed to all run hook callbacks.

    All hooks receive the same fields. Fields that are only meaningful for
    specific hook points are None when not applicable.

    The ``extras`` dict is populated by the server with data collected during
    execution (e.g., token usage metadata). Hooks should treat it as read-only.
    """
    run_id: str
    thread_id: str
    assistant_id: str
    graph_id: str
    user: User
    config: dict[str, Any] = field(default_factory=dict)
    input: dict[str, Any] | None = None
    output: dict[str, Any] | None = None
    status: str | None = None
    error: str | None = None
    error_type: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class RejectRun(Exception):
    """Raise inside a before_run hook to reject execution."""
    def __init__(self, message: str = "Run rejected by hook", status_code: int = 429):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class RunHooks:
    """Registry for run lifecycle hook callbacks.

    Hooks are purely observational — they can read run context and gate
    execution (via RejectRun) but cannot modify the run config or graph state.
    """

    # Class-level reference for user convenience
    RejectRun = RejectRun

    def __init__(self) -> None:
        self._before_run: list[Callable] = []
        self._after_run: list[Callable] = []
        self._on_run_error: list[Callable] = []

    def before_run(self, fn: Callable) -> Callable:
        self._before_run.append(fn)
        return fn

    def after_run(self, fn: Callable) -> Callable:
        self._after_run.append(fn)
        return fn

    def on_run_error(self, fn: Callable) -> Callable:
        self._on_run_error.append(fn)
        return fn

    async def fire_before_run(self, ctx: RunContext, timeout: float = 10.0) -> None:
        """Fire all before_run hooks.

        RejectRun exceptions propagate up. TimeoutError becomes RejectRun(504).
        Hook return values are ignored.
        """
        for fn in self._before_run:
            try:
                await asyncio.wait_for(fn(ctx), timeout=timeout)
            except asyncio.TimeoutError:
                raise RejectRun(
                    f"before_run hook '{fn.__name__}' timed out after {timeout}s",
                    status_code=504,  # Gateway Timeout — upstream hook dependency timed out
                )

    async def fire_after_run(self, ctx: RunContext, timeout: float = 10.0) -> None:
        """Fire all after_run hooks. Errors are logged, not propagated."""
        for fn in self._after_run:
            try:
                await asyncio.wait_for(fn(ctx), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(
                    f"after_run hook '{fn.__name__}' timed out after {timeout}s "
                    f"for run {ctx.run_id}"
                )
            except Exception:
                logger.exception(
                    f"after_run hook '{fn.__name__}' failed for run {ctx.run_id}"
                )

    async def fire_on_run_error(self, ctx: RunContext, timeout: float = 10.0) -> None:
        """Fire all on_run_error hooks. Errors are logged, not propagated.

        Note: This method catches BaseException (not just Exception) because it
        may be called inside a CancelledError handler. If the task is being
        cancelled (e.g., server shutdown), the await inside this method could
        itself raise CancelledError. We must not let that escape unhandled.
        """
        for fn in self._on_run_error:
            try:
                await asyncio.wait_for(fn(ctx), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(
                    f"on_run_error hook '{fn.__name__}' timed out after {timeout}s "
                    f"for run {ctx.run_id}"
                )
            except BaseException:
                # BaseException (not Exception) to also catch CancelledError,
                # which can occur if this method is called during task cancellation.
                logger.exception(
                    f"on_run_error hook '{fn.__name__}' failed for run {ctx.run_id}"
                )


# ---------- Global singleton ----------
_hooks: RunHooks | None = None
_hooks_timeout: float = 10.0


def get_hooks() -> RunHooks | None:
    return _hooks


def get_hooks_timeout() -> float:
    return _hooks_timeout


def set_hooks(hooks: RunHooks, timeout: float = 10.0) -> None:
    global _hooks, _hooks_timeout
    _hooks = hooks
    _hooks_timeout = timeout


# ---------- Loader ----------
def load_hooks(hooks_import: str, base_dir: Path | None = None) -> RunHooks:
    """Load RunHooks instance from import path.

    Supports: "./hooks.py:hooks" or "mypackage.hooks:hooks"
    Follows same pattern as core/app_loader.py.
    """
    if ":" not in hooks_import:
        raise ValueError(
            f"Invalid hooks import path '{hooks_import}'. "
            "Expected format: './file.py:variable' or 'module.path:variable'"
        )

    module_path, attr_name = hooks_import.rsplit(":", 1)

    # Handle relative file paths
    if module_path.startswith("."):
        if base_dir is None:
            base_dir = Path.cwd()
        file_path = (base_dir / module_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Hooks file not found: {file_path}")

        spec = importlib.util.spec_from_file_location("_user_hooks", file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load hooks from {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_path)

    hooks = getattr(module, attr_name, None)
    if hooks is None:
        raise AttributeError(
            f"Module '{module_path}' has no attribute '{attr_name}'"
        )
    if not isinstance(hooks, RunHooks):
        raise TypeError(
            f"Expected RunHooks instance, got {type(hooks).__name__}. "
            f"Make sure '{attr_name}' is an instance of RunHooks."
        )

    return hooks
```

### New file: `src/aegra_api/hooks.py` (~10 lines, public re-export)

```python
"""Public API for run lifecycle hooks.

Users should import from this module, not from aegra_api.core.hooks directly:

    from aegra_api.hooks import RunHooks, RunContext, RejectRun
"""

from aegra_api.core.hooks import RejectRun, RunContext, RunHooks

__all__ = ["RunHooks", "RunContext", "RejectRun"]
```

### Changes to `src/aegra_api/config.py` (+15 lines)

```python
class HooksConfig(TypedDict, total=False):
    path: str
    """Import path for hooks in format './file.py:variable'"""
    timeout: float
    """Max seconds a hook can run before being killed (default: 10)"""

def load_hooks_config() -> HooksConfig | None:
    config = load_config()
    if config is None:
        return None
    return config.get("hooks")
```

### Changes to `src/aegra_api/main.py` (+12 lines)

Add imports at the top of the file (not inline in the lifespan function):

```python
# Top of main.py (new imports)
from aegra_api.config import load_hooks_config
from aegra_api.core.hooks import load_hooks, set_hooks
```

In the `lifespan` function, after LangGraph service initialization (after line 95):

```python
# Load run hooks if configured
hooks_config = load_hooks_config()
if hooks_config and hooks_config.get("path"):
    config_dir = get_config_dir()  # or Path.cwd()
    timeout = hooks_config.get("timeout", 10.0)
    user_hooks = load_hooks(hooks_config["path"], base_dir=config_dir)
    set_hooks(user_hooks, timeout=timeout)
    logger.info("Run hooks loaded successfully")
```

### Changes to `src/aegra_api/services/langgraph_service.py` (+5 lines)

Fix existing code quality issues in `create_run_config()`:

1. **Add type annotation for `user`** — currently untyped, violates CLAUDE.md strict typing rules.
2. **Fix `additional_config: dict = None`** — misleading annotation; should be `dict | None = None`.
3. **Move `from copy import deepcopy`** — currently an inline import inside the function body (line 416); move to top of file per CLAUDE.md import rules.

```python
from copy import deepcopy  # Move to top of file (currently inline at line 416)

def create_run_config(
    run_id: str,
    thread_id: str,
    user: User,                                    # FIX: was untyped
    additional_config: dict | None = None,         # FIX: was `dict = None`
    checkpoint: dict | None = None,
) -> dict:
    # ... unchanged ...
```

### Changes to `src/aegra_api/api/runs.py` (~40 lines)

#### 1. Thread `assistant_id` through to `execute_run_async`

Add `assistant_id: str` parameter to `execute_run_async` signature. Update all 3 call sites (`create_run`, `create_and_stream_run`, `wait_for_run`) to pass it through. They already have `resolved_assistant_id` available.

#### 2. Wire hooks into `execute_run_async`

Pseudocode for the three injection points:

```python
# --- Top of runs.py (new imports) ---
from aegra_api.core.hooks import get_hooks, get_hooks_timeout, RejectRun, RunContext

# --- Inside execute_run_async ---

async def execute_run_async(
    run_id, thread_id, assistant_id, graph_id, input_data, user,
    config, context, stream_mode, session, checkpoint, command,
    interrupt_before, interrupt_after, _multitask_strategy, subgraphs,
):
    owns_session = session is None
    if session is None:
        maker = _get_session_maker()
        session = maker()

    hooks = get_hooks()
    timeout = get_hooks_timeout()

    try:
        await update_run_status(run_id, "running", session=session)
        langgraph_service = get_langgraph_service()

        run_config = create_run_config(run_id, thread_id, user, config or {}, checkpoint)

        # ... existing interrupt/command setup (unchanged) ...

        # === HOOK POINT 1: before_run ===
        if hooks:
            before_ctx = RunContext(
                run_id=run_id, thread_id=thread_id,
                assistant_id=assistant_id, graph_id=graph_id,
                user=user, config=run_config, input=input_data,
            )
            try:
                await hooks.fire_before_run(before_ctx, timeout=timeout)
            except RejectRun as e:
                await update_run_status(run_id, "error", output={}, error=e.message, session=session)
                # Thread goes back to "idle", not "error" — the run was denied before
                # execution started. The thread itself is fine; only the run failed.
                await set_thread_status(session, thread_id, "idle")
                await streaming_service.signal_run_error(run_id, e.message, "RejectRun")
                return

        # ... existing graph execution, wrapped in get_usage_metadata_callback() ...
        # (see Appendix: Token Tracking for how the context manager auto-injects
        #  the callback into all LLM calls within the block)

        # Build extras dict with any server-collected data (e.g., token usage)
        run_extras: dict[str, Any] = {}
        # usage_cb is from `with get_usage_metadata_callback() as usage_cb:` wrapping
        # the graph execution block. If token tracking is enabled, capture its data.
        if usage_cb is not None and usage_cb.usage_metadata:
            run_extras["usage_metadata"] = dict(usage_cb.usage_metadata)

        if has_interrupt:
            await update_run_status(run_id, "interrupted", output=..., session=session)
            await set_thread_status(session, thread_id, "interrupted")
            # === HOOK POINT 2a: after_run (interrupted) ===
            if hooks:
                after_ctx = RunContext(
                    run_id=run_id, thread_id=thread_id,
                    assistant_id=assistant_id, graph_id=graph_id,
                    user=user, config=run_config,
                    output=final_output or {}, status="interrupted",
                    extras=run_extras,
                )
                await hooks.fire_after_run(after_ctx, timeout=timeout)
        else:
            await update_run_status(run_id, "success", output=..., session=session)
            await set_thread_status(session, thread_id, "idle")
            # === HOOK POINT 2b: after_run (success) ===
            if hooks:
                after_ctx = RunContext(
                    run_id=run_id, thread_id=thread_id,
                    assistant_id=assistant_id, graph_id=graph_id,
                    user=user, config=run_config,
                    output=final_output or {}, status="success",
                    extras=run_extras,
                )
                await hooks.fire_after_run(after_ctx, timeout=timeout)

    except asyncio.CancelledError:
        # ... existing handling ...
        # Build extras — partial token usage may be available even on cancellation
        error_extras: dict[str, Any] = {}
        if usage_cb is not None and usage_cb.usage_metadata:
            error_extras["usage_metadata"] = dict(usage_cb.usage_metadata)

        # === HOOK POINT 3a: on_run_error (cancellation) ===
        if hooks:
            error_ctx = RunContext(
                run_id=run_id, thread_id=thread_id,
                assistant_id=assistant_id, graph_id=graph_id,
                user=user, config=run_config if run_config else {},
                error="Run was cancelled", error_type="CancelledError",
                extras=error_extras,
            )
            # Shield the hook call from the ongoing cancellation. Without this,
            # the await inside fire_on_run_error would immediately raise
            # CancelledError again, preventing the hook from running at all.
            try:
                await asyncio.shield(hooks.fire_on_run_error(error_ctx, timeout=timeout))
            except asyncio.CancelledError:
                logger.warning(f"on_run_error hooks cancelled during shutdown for run {run_id}")
        raise

    except RejectRun:
        pass  # Already handled above

    except Exception as e:
        # ... existing error handling ...
        # Build extras — partial token usage may be available even on error
        error_extras_exc: dict[str, Any] = {}
        if usage_cb is not None and usage_cb.usage_metadata:
            error_extras_exc["usage_metadata"] = dict(usage_cb.usage_metadata)

        # === HOOK POINT 3b: on_run_error (exception) ===
        if hooks:
            error_ctx = RunContext(
                run_id=run_id, thread_id=thread_id,
                assistant_id=assistant_id, graph_id=graph_id,
                user=user, config=run_config if run_config else {},
                error=str(e), error_type=type(e).__name__,
                extras=error_extras_exc,
            )
            await hooks.fire_on_run_error(error_ctx, timeout=timeout)

    finally:
        # ... existing cleanup (unchanged) ...
```

### Separate fix: Auth handler gap in `runs.py`

`create_and_stream_run` (after line 304) and `wait_for_run` (after line 631) are missing the `@auth.on.threads.create_run` authorization handler call that `create_run` has (lines 170-184). This is a bug where two of three run-creation endpoints skip auth handlers. Fix by copying the same auth handler pattern.

---

## Known Issues and Mitigations

### 1. Hook timeout — hanging external calls (HIGH)

**Problem:** If a `before_run` hook hangs (e.g., an external API is down), `execute_run_async` blocks indefinitely. This holds a DB pool connection, leaves the run in "running" status, and can exhaust the connection pool (blocking the entire server) if enough runs hang concurrently.

**Mitigation:** All hook calls are wrapped in `asyncio.wait_for()` with a configurable timeout (default 10s). If a `before_run` hook times out, the run is rejected with a 504 (Gateway Timeout). If `after_run` or `on_run_error` time out, the timeout is logged and execution continues.

### 2. Hooks must not use the shared DB session (HIGH)

**Problem:** `execute_run_async` creates a bare `AsyncSession` at startup that lives for the entire run. If a hook uses this session (e.g., to query the DB), it could interleave transactions or leave the session in an error state.

**Mitigation:** The `RunContext` does NOT include the session. Hooks that need DB access must create their own session. This should be documented explicitly.

### 3. `assistant_id` not in `execute_run_async` (MEDIUM)

**Problem:** The function receives `graph_id` but not `assistant_id`. All callers have it but don't pass it through.

**Mitigation:** Add `assistant_id` parameter and thread it from all 3 call sites.

### 4. `wait_for_run` doesn't return errors to client (MEDIUM)

**Problem:** `wait_for_run` returns `run_orm.output` after the task completes, but doesn't check if the run errored. If a `before_run` hook rejects, the client gets `{}` with no error message.

**Mitigation:** Separate bug fix — `wait_for_run` should check `run_orm.status == "error"` and raise an `HTTPException` with the error message. Same for `"interrupted"`. This is a pre-existing bug, not caused by hooks.

### 5. CancelledError can re-cancel hook execution (MEDIUM)

**Problem:** When `CancelledError` is caught in `execute_run_async` and we call `hooks.fire_on_run_error()`, the `await` inside that method could itself raise `CancelledError` (the task is still being cancelled). In Python 3.12+, `CancelledError` is a `BaseException`, not an `Exception`, so a standard `except Exception` handler won't catch it.

**Mitigation:** Two defenses:
1. The `fire_on_run_error` call in the `CancelledError` handler is wrapped in `asyncio.shield()` to protect it from the ongoing cancellation.
2. `fire_on_run_error` itself catches `BaseException` (not just `Exception`) so that if a `CancelledError` does slip through (e.g., server shutdown), it's logged and suppressed rather than propagating unhandled.

### 6. Cancellation race — double status write (LOW)

**Problem:** `cancel_run_endpoint` sets `status="interrupted"` in the DB, then the cancelled task's `CancelledError` handler also sets `status="interrupted"`. Both write to the same row.

**Mitigation:** This is idempotent (same value written twice). The `on_run_error` hook fires exactly once (in the `CancelledError` handler inside `execute_run_async`). Document that `error_type="CancelledError"` means client-initiated cancellation.

---

## Files to Create

| File                                                          | Lines (est.) | Purpose                                                                |
| ------------------------------------------------------------- | ------------ | ---------------------------------------------------------------------- |
| `libs/aegra-api/src/aegra_api/hooks.py`                      | ~10          | Public re-export module: `from aegra_api.hooks import RunHooks, RunContext, RejectRun` — users should never import from `core/hooks.py` directly |
| `libs/aegra-api/src/aegra_api/core/hooks.py`                 | ~150         | `RunHooks`, `RunContext`, `RejectRun`, loader, singleton |
| `libs/aegra-api/tests/unit/test_core/test_hooks.py`          | ~250         | Registration, loading, fire semantics, error handling, timeout, RejectRun |
| `libs/aegra-api/tests/integration/test_hooks_integration.py` | ~100         | Hooks wired through `execute_run_async` with mock graph                |
| `libs/aegra-api/tests/e2e/test_hooks/test_hooks_e2e.py`     | ~80          | E2E: configure hooks in aegra.json, run a graph, verify hooks fire with correct context |

## Files to Modify

| File                                                          | Lines changed (est.) | Change                                                                                  |
| ------------------------------------------------------------- | -------------------- | --------------------------------------------------------------------------------------- |
| `libs/aegra-api/src/aegra_api/config.py`                     | +15                  | `HooksConfig` TypedDict + `load_hooks_config()`                                        |
| `libs/aegra-api/src/aegra_api/main.py`                       | +12                  | Load hooks during lifespan startup                                                      |
| `libs/aegra-api/src/aegra_api/api/runs.py`                   | +40                  | Wire hooks into `execute_run_async` + thread `assistant_id` + auth fix   |
| `libs/aegra-api/src/aegra_api/services/langgraph_service.py` | +5                   | Fix `user` type annotation, fix `additional_config` default, move `deepcopy` import |
| `libs/aegra-api/pyproject.toml`                              | +1                   | Version bump (0.7.3 -> 0.7.4)                                                          |
| `libs/aegra-cli/pyproject.toml`                              | +1                   | Version bump (0.7.3 -> 0.7.4)                                                          |
| `docs/configuration.mdx`                                     | +30                  | Document new `hooks` config key with examples                                           |
| `docs/openapi.json`                                          | (regen)              | Regenerate via `make openapi` (new 504 error response from RejectRun)                   |
| `CLAUDE.md`                                                  | +5                   | Add hooks to architecture overview if needed                                            |
| `README.md`                                                  | +5                   | Mention run lifecycle hooks in feature list                                             |

**Total: ~230 lines production code, ~430 lines tests**

> **Note on loader duplication:** `load_hooks()` in `core/hooks.py` duplicates ~40 lines of
> dynamic import logic from `core/app_loader.py`. Consider extracting a shared
> `load_module_attr(import_path: str, base_dir: Path | None, expected_type: type, label: str)`
> utility that both loaders can delegate to. This is not blocking but would reduce
> maintenance burden if more loaders are added in the future.

---

## Design Decisions

| Decision                                  | Choice                           | Rationale                                                                 |
| ----------------------------------------- | -------------------------------- | ------------------------------------------------------------------------- |
| Hooks are observe-only                    | No config modification, no return values | Clean contract: hooks observe and gate, they never modify run state  |
| Loader pattern                            | Same as `core/app_loader.py`     | Consistent with `auth.path` and `http.app`                                |
| Global singleton                          | Same as `db_manager`, `broker_manager` | Consistent with existing patterns                                    |
| `RunContext` frozen                       | Yes                              | Hooks observe, they don't mutate server state                             |
| Token tracking via `get_usage_metadata_callback()` | Context manager wrapping graph execution in `execute_run_async` | Uses `langchain-core`'s built-in `ContextVar`-based auto-injection rather than manually appending to `config["callbacks"]`. Zero graph code changes required. Opt-in at the server level. |
| `extras: dict[str, Any]` on `RunContext`  | Generic extension dict           | Avoids adding domain-specific fields (e.g., `usage_metadata`) directly to `RunContext`. Server populates with data collected during execution; hooks read via conventional string keys. Trades type safety for extensibility. |
| Hook timeout default                      | 10 seconds                       | Generous for an API call but prevents indefinite hangs                    |
| `after_run` / `on_run_error` timeouts     | Log and continue                 | Don't fail a completed run because a hook call was slow                   |
| Hook timeout status code                  | 504 (Gateway Timeout)            | More precise than 503 — communicates "upstream dependency timed out"       |
| Thread status on `RejectRun`              | Reset to `"idle"`                | Run was denied, thread itself is fine — don't mark it as errored          |
| `on_run_error` in `CancelledError` path   | `asyncio.shield()` + `BaseException` catch | Prevents re-cancellation from killing the hook call                |
| Public import path                        | `aegra_api.hooks` (re-export)    | Users import from stable public API, not internal `core/hooks.py`         |

---

## PR Structure

### Commit 1: Fix auth handler gap
`runs.py` only — adds missing `build_auth_context` / `handle_event` calls to `create_and_stream_run` and `wait_for_run`.

### Commit 2: Fix code quality issues in `langgraph_service.py`
- Add `User` type annotation to `create_run_config` `user` param
- Fix `additional_config: dict = None` -> `dict | None = None`
- Move `from copy import deepcopy` to top of file

### Commit 3: Add run lifecycle hooks
- `hooks.py` (new — public re-export module)
- `core/hooks.py` (new — implementation)
- `config.py` (add HooksConfig)
- `main.py` (load hooks at startup)
- `runs.py` (wire hooks into `execute_run_async`, thread `assistant_id`)

### Commit 4: Tests
- Unit tests for hooks registration, loading, fire semantics, timeout behavior
- Integration tests with mock graph execution
- E2E test with hooks configured in aegra.json

### Commit 5: Documentation and version bump
- Bump version 0.7.3 -> 0.7.4 in both `pyproject.toml` files
- Add `hooks` config documentation to `docs/configuration.mdx`
- Create `docs/run-hooks.mdx` guide with usage examples (gating, audit logging, analytics)
- Update `README.md` feature list
- Regenerate `docs/openapi.json` via `make openapi`

---

## Appendix: Token Tracking and `extras["usage_metadata"]`

Token tracking is **not built into hooks** — hooks are general-purpose and don't couple to LangChain internals. However, the server can optionally use `langchain-core`'s `get_usage_metadata_callback()` context manager to collect per-model token usage during graph execution and populate `extras["usage_metadata"]` on the `RunContext`.

### How `get_usage_metadata_callback()` Works

`langchain-core` (>= 0.3.49) provides a context manager that uses `ContextVar` + `register_configure_hook` to **auto-inject** a `UsageMetadataCallbackHandler` into every LLM call within its scope. No manual `config["callbacks"].append(...)` is needed.

```python
from langchain_core.callbacks import get_usage_metadata_callback

with get_usage_metadata_callback() as cb:
    # Any LLM call inside this block — including those deep inside
    # a LangGraph graph — will have its token usage captured in `cb`.
    async for event_type, event_data in stream_graph_events(graph, ...):
        ...

# After the block exits:
cb.usage_metadata  # dict keyed by model name with token counts
```

The mechanism:
1. `get_usage_metadata_callback()` calls `register_configure_hook()` with `inheritable=True`
2. This registers a `ContextVar` that LangChain's `_configure()` checks on **every** callback manager creation
3. Every LLM call within the `with` block (including nested graph nodes) automatically gets the handler injected
4. `ContextVar` values propagate through `await` and `asyncio.create_task()`, so it works correctly in async code
5. The handler's `on_llm_end` extracts `AIMessage.usage_metadata` and `response_metadata["model_name"]`, accumulating per-model totals with thread-safe locking

This is much simpler than manually appending to `config["callbacks"]` — and it requires **zero changes** to graph code.

### Server-Side Integration in `execute_run_async`

Wrap the graph execution block in the context manager:

```python
from langchain_core.callbacks import get_usage_metadata_callback

# Inside execute_run_async, wrap the graph execution:
with get_usage_metadata_callback() as usage_cb:
    async with langgraph_service.get_graph(graph_id) as graph:
        async for event_type, event_data in stream_graph_events(...):
            ...

# After execution, populate extras:
run_extras: dict[str, Any] = {}
if usage_cb.usage_metadata:
    run_extras["usage_metadata"] = dict(usage_cb.usage_metadata)
```

This is a **server-level change** in `execute_run_async`, not a hooks change. The hooks system itself has no knowledge of token tracking.

### What hooks receive

When `extras["usage_metadata"]` is populated, it contains a dict keyed by model name:

```python
# ctx.extras["usage_metadata"] example:
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

Hooks access it via `ctx.extras.get("usage_metadata")`. If token tracking is not enabled, the key is absent and `.get()` returns `None`.

### Prerequisite: `model_name` in response metadata

`UsageMetadataCallbackHandler` requires both `AIMessage.usage_metadata` and `response_metadata["model_name"]` to be non-None. If either is missing, that LLM call's tokens are silently skipped. `ChatOpenAI`, `ChatAnthropic`, and most major providers populate both.
