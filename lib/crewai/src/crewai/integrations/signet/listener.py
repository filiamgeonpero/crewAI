"""Signet event listener that signs paired CrewAI action events.

The listener subscribes to the paired ``Started``/``Completed`` events emitted
for structured tool calls, MCP tool executions, and A2A delegations. When a
``Completed`` event fires, it correlates the two payloads via the CrewAI event
scope (``event.started_event_id``) and produces a single Ed25519-signed Signet
receipt covering the input and the output.

The ``signet_auth`` dependency is **lazy-imported**. If the ``crewai[signet]``
extra is not installed and no ``signing_agent`` is injected, a clear
:class:`ImportError` is raised the first time a matching event fires. Users
who do not opt in pay no import cost.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from crewai.events.base_event_listener import BaseEventListener
from crewai.events.types.a2a_events import (
    A2ADelegationCompletedEvent,
    A2ADelegationStartedEvent,
)
from crewai.events.types.mcp_events import (
    MCPToolExecutionCompletedEvent,
    MCPToolExecutionFailedEvent,
    MCPToolExecutionStartedEvent,
)
from crewai.events.types.tool_usage_events import (
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)
from crewai.integrations.signet.config import SignetConfig


if TYPE_CHECKING:
    from crewai.events.base_events import BaseEvent
    from crewai.events.event_bus import CrewAIEventsBus


_StartedEventT = TypeVar("_StartedEventT")


_SIGNET_INSTALL_HINT: str = (
    "The Signet integration requires the `signet-auth` package. Install the "
    "optional extra with `pip install 'crewai[signet]'` or inject a "
    "`signing_agent` with a `.sign(action, params=...)` method."
)


@runtime_checkable
class SigningAgentProtocol(Protocol):
    """Minimal protocol satisfied by ``signet_auth.SigningAgent``.

    Any object exposing a ``sign(action, params=...)`` method returning a
    receipt (typically a mapping) is accepted. This keeps the listener
    decoupled from ``signet-auth`` for testing and for alternative backends.
    """

    def sign(
        self, action: str, *, params: dict[str, Any]
    ) -> Any:  # pragma: no cover - protocol
        ...


@dataclass
class Receipt:
    """A signed receipt produced by the listener.

    Attributes:
        kind: One of ``"tool"``, ``"mcp_tool"``, ``"a2a_delegation"``.
        action: Action name used when signing (e.g. the tool name).
        payload: Canonical dict passed to the signing agent covering both the
            input (from the ``Started`` event) and the output (from the
            ``Completed`` event).
        receipt: The raw object returned by the signing agent.
        error: ``True`` if the receipt was produced from an error/failed event.
    """

    kind: str
    action: str
    payload: dict[str, Any]
    receipt: Any
    error: bool = False


class SignetEventListener(BaseEventListener):
    """Event listener that produces Signet receipts for governed actions.

    Args:
        config: :class:`SignetConfig` controlling which event surfaces are
            signed and how the ``SigningAgent`` is built.
        signing_agent: Optional pre-built signing agent. When provided the
            ``signet_auth`` package is not imported. Must expose
            ``sign(action, params=...)``.
    """

    verbose: bool = False

    def __init__(
        self,
        config: SignetConfig,
        *,
        signing_agent: Any | None = None,
    ) -> None:
        self.config = config
        self._injected_signing_agent = signing_agent
        self._signing_agent: Any | None = signing_agent
        self._pending: dict[str, BaseEvent] = {}
        self._pending_lock = threading.Lock()
        self.receipts: list[Receipt] = []
        self._receipts_lock = threading.Lock()
        super().__init__()

    def _get_signing_agent(self) -> Any:
        """Return the active signing agent, lazily building one if needed."""
        if self._signing_agent is not None:
            return self._signing_agent
        try:
            from signet_auth import SigningAgent  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised via stubbed test
            raise ImportError(_SIGNET_INSTALL_HINT) from exc

        kwargs: dict[str, Any] = {}
        if self.config.audit:
            kwargs["audit"] = True
        if self.config.policy_path is not None:
            kwargs["policy_path"] = self.config.policy_path

        if self.config.create_if_missing and hasattr(SigningAgent, "create"):
            create_kwargs = dict(kwargs)
            if self.config.owner is not None:
                create_kwargs["owner"] = self.config.owner
            self._signing_agent = SigningAgent.create(
                self.config.key_name, **create_kwargs
            )
        else:
            self._signing_agent = SigningAgent(self.config.key_name, **kwargs)
        return self._signing_agent

    def setup_listeners(self, crewai_event_bus: CrewAIEventsBus) -> None:
        """Register handlers for each enabled event surface."""
        if self.config.tool_events:
            self._register_tool_handlers(crewai_event_bus)
        if self.config.mcp_events:
            self._register_mcp_handlers(crewai_event_bus)
        if self.config.a2a_events:
            self._register_a2a_handlers(crewai_event_bus)

    def _register_tool_handlers(self, bus: CrewAIEventsBus) -> None:
        @bus.on(ToolUsageStartedEvent)
        def _on_tool_start(source: Any, event: ToolUsageStartedEvent) -> None:
            self._remember_start(event)

        @bus.on(ToolUsageFinishedEvent)
        def _on_tool_finish(source: Any, event: ToolUsageFinishedEvent) -> None:
            started = self._consume_start(event.started_event_id, ToolUsageStartedEvent)
            if started is None:
                return
            payload = _tool_payload(started, output=event.output, error=None)
            self._sign_and_record("tool", started.tool_name, payload, error=False)

        @bus.on(ToolUsageErrorEvent)
        def _on_tool_error(source: Any, event: ToolUsageErrorEvent) -> None:
            started = self._consume_start(event.started_event_id, ToolUsageStartedEvent)
            if started is None:
                return
            payload = _tool_payload(started, output=None, error=str(event.error))
            self._sign_and_record("tool", started.tool_name, payload, error=True)

    def _register_mcp_handlers(self, bus: CrewAIEventsBus) -> None:
        @bus.on(MCPToolExecutionStartedEvent)
        def _on_mcp_start(source: Any, event: MCPToolExecutionStartedEvent) -> None:
            self._remember_start(event)

        @bus.on(MCPToolExecutionCompletedEvent)
        def _on_mcp_complete(
            source: Any, event: MCPToolExecutionCompletedEvent
        ) -> None:
            started = self._consume_start(
                event.started_event_id, MCPToolExecutionStartedEvent
            )
            if started is None:
                return
            payload = _mcp_payload(started, result=event.result, error=None)
            self._sign_and_record("mcp_tool", started.tool_name, payload, error=False)

        @bus.on(MCPToolExecutionFailedEvent)
        def _on_mcp_failed(source: Any, event: MCPToolExecutionFailedEvent) -> None:
            started = self._consume_start(
                event.started_event_id, MCPToolExecutionStartedEvent
            )
            if started is None:
                return
            payload = _mcp_payload(started, result=None, error=event.error)
            self._sign_and_record("mcp_tool", started.tool_name, payload, error=True)

    def _register_a2a_handlers(self, bus: CrewAIEventsBus) -> None:
        @bus.on(A2ADelegationStartedEvent)
        def _on_a2a_start(source: Any, event: A2ADelegationStartedEvent) -> None:
            self._remember_start(event)

        @bus.on(A2ADelegationCompletedEvent)
        def _on_a2a_complete(source: Any, event: A2ADelegationCompletedEvent) -> None:
            started = self._consume_start(
                event.started_event_id, A2ADelegationStartedEvent
            )
            if started is None:
                return
            payload = _a2a_payload(started, completed=event)
            action = f"a2a:{started.agent_id}"
            is_error = event.status.lower() not in {"completed", "ok", "success"}
            self._sign_and_record("a2a_delegation", action, payload, error=is_error)

    def _remember_start(self, event: BaseEvent) -> None:
        with self._pending_lock:
            self._pending[event.event_id] = event

    def _consume_start(
        self,
        started_event_id: str | None,
        expected_type: type[_StartedEventT],
    ) -> _StartedEventT | None:
        if not started_event_id:
            return None
        with self._pending_lock:
            started = self._pending.pop(started_event_id, None)
        if not isinstance(started, expected_type):
            return None
        return started

    def _sign_and_record(
        self,
        kind: str,
        action: str,
        payload: dict[str, Any],
        *,
        error: bool,
    ) -> None:
        agent = self._get_signing_agent()
        receipt = agent.sign(action, params=payload)
        with self._receipts_lock:
            self.receipts.append(
                Receipt(
                    kind=kind,
                    action=action,
                    payload=payload,
                    receipt=receipt,
                    error=error,
                )
            )


def _tool_payload(
    started: ToolUsageStartedEvent,
    *,
    output: Any,
    error: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": "tool",
        "tool_name": started.tool_name,
        "tool_class": started.tool_class,
        "tool_args": started.tool_args,
        "agent_id": started.agent_id,
        "agent_role": started.agent_role,
        "task_id": started.task_id,
        "started_event_id": started.event_id,
    }
    if error is None:
        payload["output"] = output
    else:
        payload["error"] = error
    return payload


def _mcp_payload(
    started: MCPToolExecutionStartedEvent,
    *,
    result: Any,
    error: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": "mcp_tool",
        "tool_name": started.tool_name,
        "tool_args": started.tool_args,
        "server_name": started.server_name,
        "server_url": started.server_url,
        "transport_type": started.transport_type,
        "agent_id": started.agent_id,
        "agent_role": started.agent_role,
        "task_id": started.task_id,
        "started_event_id": started.event_id,
    }
    if error is None:
        payload["result"] = result
    else:
        payload["error"] = error
    return payload


def _a2a_payload(
    started: A2ADelegationStartedEvent,
    *,
    completed: A2ADelegationCompletedEvent,
) -> dict[str, Any]:
    return {
        "kind": "a2a_delegation",
        "endpoint": started.endpoint,
        "task_description": started.task_description,
        "a2a_agent_id": started.agent_id,
        "a2a_agent_name": started.a2a_agent_name,
        "context_id": started.context_id,
        "is_multiturn": started.is_multiturn,
        "turn_number": started.turn_number,
        "skill_id": started.skill_id,
        "started_event_id": started.event_id,
        "status": completed.status,
        "result": completed.result,
        "error": completed.error,
    }
