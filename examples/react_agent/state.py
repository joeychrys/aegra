"""Define the state structures for the agent."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep


def _merge_usage_metadata(
    current: dict[str, Any] | None,
    update: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge per-model token usage dicts across multiple LLM calls.

    Each call to ``call_model`` produces a snapshot of cumulative usage from the
    ``get_usage_metadata_callback()`` context manager.  Since the context manager
    accumulates across the entire run, the latest snapshot is always a superset
    of previous ones — so we simply replace with the update.
    """
    if update is not None:
        return update
    return current or {}


@dataclass
class InputState:
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)
    """
    Messages tracking the primary execution state of the agent.

    Typically accumulates a pattern of:
    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    4. AIMessage without .tool_calls - agent responding in unstructured format to the user
    5. HumanMessage - user responds with the next conversational turn

    Steps 2-5 may repeat as needed.

    The `add_messages` annotation ensures that new messages are merged with existing ones,
    updating by ID to maintain an "append-only" state unless a message with the same ID is provided.
    """


@dataclass
class State(InputState):
    """Represents the complete state of the agent, extending InputState with additional attributes.

    This class can be used to store any information needed throughout the agent's lifecycle.
    """

    is_last_step: IsLastStep = field(default=False)
    """
    Indicates whether the current step is the last one before the graph raises an error.

    This is a 'managed' variable, controlled by the state machine rather than user code.
    It is set to 'True' when the step count reaches recursion_limit - 1.
    """

    usage_metadata: Annotated[dict[str, Any], _merge_usage_metadata] = field(default_factory=dict)
    """
    Per-model token usage accumulated across all LLM calls in this run.

    Populated by ``call_model`` using ``get_usage_metadata_callback()``.
    Keyed by model name, values contain ``input_tokens``, ``output_tokens``,
    ``total_tokens``, and optional detail breakdowns.
    """
