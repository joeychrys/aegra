"""Stateless (thread-free) run endpoints.

These endpoints accept POST /runs/stream, /runs/wait, and /runs without a
thread_id. They generate an ephemeral thread, delegate to the existing threaded
endpoint functions, and clean up the thread afterward (unless the caller
explicitly sets ``on_completion="keep"``).
"""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.api.runs import (
    active_runs,
    create_and_stream_run,
    create_run,
    wait_for_run,
)
from aegra_api.core.auth_deps import auth_dependency, get_current_user
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import _get_session_maker, get_session
from aegra_api.models import RunCreate, User
from aegra_api.services.streaming_service import streaming_service

router = APIRouter(tags=["Stateless Runs"], dependencies=auth_dependency)
logger = structlog.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _delete_thread_by_id(thread_id: str, user_id: str) -> None:
    """Delete an ephemeral thread and cascade-delete its runs.

    Opens its own DB session so it can be called after the request session has
    been closed (e.g. in a ``finally`` block or background task).
    """
    maker = _get_session_maker()
    async with maker() as session:
        # Cancel any still-active runs on this thread
        active_runs_stmt = select(RunORM).where(
            RunORM.thread_id == thread_id,
            RunORM.user_id == user_id,
            RunORM.status.in_(["pending", "running"]),
        )
        active_runs_list = (await session.scalars(active_runs_stmt)).all()

        for run in active_runs_list:
            run_id = run.run_id
            await streaming_service.cancel_run(run_id)
            task = active_runs.pop(run_id, None)
            if task and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

        # Delete thread (cascade deletes runs via FK)
        thread = await session.scalar(
            select(ThreadORM).where(
                ThreadORM.thread_id == thread_id,
                ThreadORM.user_id == user_id,
            )
        )
        if thread:
            await session.delete(thread)
            await session.commit()


async def _cleanup_after_background_run(run_id: str, thread_id: str, user_id: str) -> None:
    """Wait for a background run task to finish, then delete the ephemeral thread."""
    task = active_runs.get(run_id)
    if task:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    await _delete_thread_by_id(thread_id, user_id)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/runs/wait")
async def stateless_wait_for_run(
    request: RunCreate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Create a stateless run and wait for completion.

    Generates an ephemeral thread, delegates to the threaded ``wait_for_run``
    endpoint, and deletes the thread on completion (unless
    ``on_completion="keep"``).
    """
    thread_id = str(uuid4())
    should_delete = (request.on_completion or "delete") == "delete"

    try:
        result = await wait_for_run(thread_id, request, user, session)
        return result
    finally:
        if should_delete:
            await _delete_thread_by_id(thread_id, user.identity)


@router.post("/runs/stream")
async def stateless_stream_run(
    request: RunCreate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    """Create a stateless run and stream its execution.

    Generates an ephemeral thread, delegates to the threaded
    ``create_and_stream_run`` endpoint, and deletes the thread after the
    stream finishes (unless ``on_completion="keep"``).
    """
    thread_id = str(uuid4())
    should_delete = (request.on_completion or "delete") == "delete"

    response = await create_and_stream_run(thread_id, request, user, session)

    if not should_delete:
        return response

    # Wrap the body_iterator so cleanup happens after the stream ends
    original_iterator = response.body_iterator

    async def _wrapped_iterator() -> AsyncIterator[str]:
        try:
            async for chunk in original_iterator:
                yield chunk
        finally:
            await _delete_thread_by_id(thread_id, user.identity)

    return StreamingResponse(
        _wrapped_iterator(),
        media_type=response.media_type,
        headers=dict(response.headers),
    )


@router.post("/runs")
async def stateless_create_run(
    request: RunCreate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Any:
    """Create a stateless background run.

    Generates an ephemeral thread, delegates to the threaded ``create_run``
    endpoint, and schedules cleanup as a background task (unless
    ``on_completion="keep"``).
    """
    thread_id = str(uuid4())
    should_delete = (request.on_completion or "delete") == "delete"

    result = await create_run(thread_id, request, user, session)

    if should_delete:
        asyncio.create_task(_cleanup_after_background_run(result.run_id, thread_id, user.identity))

    return result
