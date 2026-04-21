"""Tests for the optional Signet integration.

These tests use a lightweight ``FakeSigningAgent`` that satisfies the same
contract as ``signet_auth.SigningAgent`` so the listener can be exercised
without installing the ``crewai[signet]`` extra.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from crewai.events.event_bus import crewai_event_bus
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
from crewai.integrations.signet import SignetConfig, SignetEventListener, install
from crewai.integrations.signet.listener import _SIGNET_INSTALL_HINT, Receipt


class FakeSigningAgent:
    """Test double matching the ``signet_auth.SigningAgent`` contract."""

    def __init__(self, name: str = "fake") -> None:
        self.name = name
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def sign(self, action: str, *, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((action, params))
        return {
            "action": action,
            "params": params,
            "signature": f"sig-{len(self.calls)}",
            "signed_by": self.name,
        }


def _install(**kwargs: Any) -> tuple[SignetEventListener, FakeSigningAgent]:
    agent = FakeSigningAgent()
    listener = install(key_name="test-key", signing_agent=agent, **kwargs)
    return listener, agent


def _wait() -> None:
    """Block until all pending event handlers have finished."""
    crewai_event_bus.flush(timeout=5.0)


def _emit_tool_pair(
    tool_name: str = "some_tool",
    *,
    with_error: bool = False,
    output: Any = "ok",
) -> tuple[ToolUsageStartedEvent, ToolUsageFinishedEvent | ToolUsageErrorEvent]:
    started = ToolUsageStartedEvent(
        tool_name=tool_name,
        tool_args={"x": 1},
        tool_class="SomeTool",
        agent_id="agent-1",
        agent_role="analyst",
    )
    crewai_event_bus.emit(source=None, event=started)

    finished: ToolUsageFinishedEvent | ToolUsageErrorEvent
    if with_error:
        finished = ToolUsageErrorEvent(
            tool_name=tool_name,
            tool_args={"x": 1},
            tool_class="SomeTool",
            agent_id="agent-1",
            agent_role="analyst",
            error="boom",
            started_event_id=started.event_id,
        )
    else:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        finished = ToolUsageFinishedEvent(
            tool_name=tool_name,
            tool_args={"x": 1},
            tool_class="SomeTool",
            agent_id="agent-1",
            agent_role="analyst",
            started_at=now,
            finished_at=now,
            output=output,
            started_event_id=started.event_id,
        )
    crewai_event_bus.emit(source=None, event=finished)
    _wait()
    return started, finished


def _emit_mcp_pair(
    *,
    with_error: bool = False,
    tool_name: str = "mcp_echo",
) -> tuple[
    MCPToolExecutionStartedEvent,
    MCPToolExecutionCompletedEvent | MCPToolExecutionFailedEvent,
]:
    started = MCPToolExecutionStartedEvent(
        server_name="server-a",
        server_url="http://localhost:8080",
        transport_type="http",
        tool_name=tool_name,
        tool_args={"q": "hi"},
    )
    crewai_event_bus.emit(source=None, event=started)

    completed: MCPToolExecutionCompletedEvent | MCPToolExecutionFailedEvent
    if with_error:
        completed = MCPToolExecutionFailedEvent(
            server_name="server-a",
            tool_name=tool_name,
            tool_args={"q": "hi"},
            error="server crashed",
            started_event_id=started.event_id,
        )
    else:
        completed = MCPToolExecutionCompletedEvent(
            server_name="server-a",
            tool_name=tool_name,
            tool_args={"q": "hi"},
            result={"echo": "hi"},
            started_event_id=started.event_id,
        )
    crewai_event_bus.emit(source=None, event=completed)
    _wait()
    return started, completed


def _emit_a2a_pair(
    *,
    status: str = "completed",
) -> tuple[A2ADelegationStartedEvent, A2ADelegationCompletedEvent]:
    started = A2ADelegationStartedEvent(
        endpoint="https://remote/agent",
        task_description="summarize",
        agent_id="remote-agent-1",
        context_id="ctx-1",
    )
    crewai_event_bus.emit(source=None, event=started)

    completed = A2ADelegationCompletedEvent(
        status=status,
        result="done" if status == "completed" else None,
        error=None if status == "completed" else "refused",
        context_id="ctx-1",
        endpoint="https://remote/agent",
        started_event_id=started.event_id,
    )
    crewai_event_bus.emit(source=None, event=completed)
    _wait()
    return started, completed


class TestSignetConfig:
    def test_defaults(self) -> None:
        cfg = SignetConfig(key_name="k")
        assert cfg.key_name == "k"
        assert cfg.audit is True
        assert cfg.create_if_missing is True
        assert cfg.tool_events and cfg.mcp_events and cfg.a2a_events

    def test_key_name_required(self) -> None:
        with pytest.raises(ValueError):
            SignetConfig(key_name="")

    def test_frozen(self) -> None:
        cfg = SignetConfig(key_name="k")
        with pytest.raises(Exception):
            cfg.key_name = "other"  # type: ignore[misc]


class TestToolSigning:
    def test_signs_one_receipt_per_tool_call(self) -> None:
        with crewai_event_bus.scoped_handlers():
            listener, agent = _install()
            started, finished = _emit_tool_pair(output={"answer": 42})

        assert len(listener.receipts) == 1
        rec = listener.receipts[0]
        assert isinstance(rec, Receipt)
        assert rec.kind == "tool"
        assert rec.action == "some_tool"
        assert rec.error is False
        assert rec.payload["tool_name"] == "some_tool"
        assert rec.payload["tool_args"] == {"x": 1}
        assert rec.payload["output"] == {"answer": 42}
        assert rec.payload["started_event_id"] == started.event_id
        assert rec.payload["agent_id"] == "agent-1"
        assert rec.receipt["signature"] == "sig-1"
        assert agent.calls[0][0] == "some_tool"

    def test_signs_error_event(self) -> None:
        with crewai_event_bus.scoped_handlers():
            listener, _ = _install()
            _emit_tool_pair(with_error=True)

        assert len(listener.receipts) == 1
        rec = listener.receipts[0]
        assert rec.error is True
        assert rec.payload["error"] == "boom"
        assert "output" not in rec.payload

    def test_finished_without_started_is_skipped(self) -> None:
        with crewai_event_bus.scoped_handlers():
            listener, _ = _install()
            from datetime import datetime, timezone

            now = datetime.now(timezone.utc)
            orphan = ToolUsageFinishedEvent(
                tool_name="orphan",
                tool_args={},
                started_at=now,
                finished_at=now,
                output="x",
                started_event_id="does-not-exist",
            )
            crewai_event_bus.emit(source=None, event=orphan)

        assert listener.receipts == []

    def test_pairs_are_independent_across_calls(self) -> None:
        with crewai_event_bus.scoped_handlers():
            listener, _ = _install()
            _emit_tool_pair(tool_name="t1", output="a")
            _emit_tool_pair(tool_name="t2", output="b")

        actions = [r.action for r in listener.receipts]
        outputs = [r.payload["output"] for r in listener.receipts]
        assert actions == ["t1", "t2"]
        assert outputs == ["a", "b"]


class TestMCPSigning:
    def test_signs_mcp_tool_execution(self) -> None:
        with crewai_event_bus.scoped_handlers():
            listener, _ = _install()
            _emit_mcp_pair()

        assert len(listener.receipts) == 1
        rec = listener.receipts[0]
        assert rec.kind == "mcp_tool"
        assert rec.action == "mcp_echo"
        assert rec.payload["server_name"] == "server-a"
        assert rec.payload["result"] == {"echo": "hi"}
        assert rec.error is False

    def test_signs_mcp_failure(self) -> None:
        with crewai_event_bus.scoped_handlers():
            listener, _ = _install()
            _emit_mcp_pair(with_error=True)

        assert len(listener.receipts) == 1
        rec = listener.receipts[0]
        assert rec.error is True
        assert rec.payload["error"] == "server crashed"


class TestA2ASigning:
    def test_signs_successful_a2a_delegation(self) -> None:
        with crewai_event_bus.scoped_handlers():
            listener, _ = _install()
            _emit_a2a_pair(status="completed")

        assert len(listener.receipts) == 1
        rec = listener.receipts[0]
        assert rec.kind == "a2a_delegation"
        assert rec.action == "a2a:remote-agent-1"
        assert rec.payload["status"] == "completed"
        assert rec.payload["endpoint"] == "https://remote/agent"
        assert rec.error is False

    def test_flags_failed_a2a_delegation_as_error(self) -> None:
        with crewai_event_bus.scoped_handlers():
            listener, _ = _install()
            _emit_a2a_pair(status="failed")

        assert len(listener.receipts) == 1
        assert listener.receipts[0].error is True
        assert listener.receipts[0].payload["status"] == "failed"


class TestSurfaceToggles:
    def test_disabling_tool_events_skips_tool_signing(self) -> None:
        with crewai_event_bus.scoped_handlers():
            listener, _ = _install(tool_events=False)
            _emit_tool_pair()
            _emit_mcp_pair()

        assert len(listener.receipts) == 1
        assert listener.receipts[0].kind == "mcp_tool"

    def test_disabling_mcp_events_skips_mcp_signing(self) -> None:
        with crewai_event_bus.scoped_handlers():
            listener, _ = _install(mcp_events=False)
            _emit_tool_pair()
            _emit_mcp_pair()

        assert len(listener.receipts) == 1
        assert listener.receipts[0].kind == "tool"

    def test_disabling_a2a_events_skips_a2a_signing(self) -> None:
        with crewai_event_bus.scoped_handlers():
            listener, _ = _install(a2a_events=False)
            _emit_a2a_pair()

        assert listener.receipts == []


class TestLazyImport:
    def test_missing_signet_auth_raises_clear_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When no signing_agent is injected and signet_auth isn't installed,
        ``_get_signing_agent`` must raise a clear ImportError — not at import
        or registration time.
        """
        monkeypatch.setitem(sys.modules, "signet_auth", None)

        with crewai_event_bus.scoped_handlers():
            listener = install(key_name="real-key")
            assert listener.receipts == []
            with pytest.raises(ImportError, match="signet-auth"):
                listener._get_signing_agent()

    def test_builds_signing_agent_from_fake_signet_auth(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If signet_auth is importable, SigningAgent.create is used with the
        config-derived kwargs."""
        captured: dict[str, Any] = {}

        class _SigningAgent:
            def __init__(self, name: str, **kwargs: Any) -> None:
                self.name = name
                self.kwargs = kwargs

            @classmethod
            def create(cls, name: str, **kwargs: Any) -> "_SigningAgent":
                captured["create"] = (name, kwargs)
                return cls(name, **kwargs)

            def sign(self, action: str, *, params: dict[str, Any]) -> dict[str, Any]:
                return {"action": action, "params": params, "by": self.name}

        fake_module = types.ModuleType("signet_auth")
        fake_module.SigningAgent = _SigningAgent  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "signet_auth", fake_module)

        with crewai_event_bus.scoped_handlers():
            listener = install(
                key_name="team-agent",
                owner="alice",
                audit=True,
                policy_path="/tmp/policy.yaml",
            )
            agent = listener._get_signing_agent()

        assert captured["create"][0] == "team-agent"
        assert captured["create"][1] == {
            "audit": True,
            "policy_path": "/tmp/policy.yaml",
            "owner": "alice",
        }
        assert isinstance(agent, _SigningAgent)
        # Subsequent calls must reuse the same instance.
        assert listener._get_signing_agent() is agent

    def test_loads_existing_identity_when_create_if_missing_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When ``create_if_missing=False`` the listener instantiates
        ``SigningAgent`` directly instead of calling ``create``."""
        init_args: dict[str, Any] = {}

        class _SigningAgent:
            def __init__(self, name: str, **kwargs: Any) -> None:
                init_args["name"] = name
                init_args["kwargs"] = kwargs

            def sign(self, action: str, *, params: dict[str, Any]) -> dict[str, Any]:
                return {"action": action, "params": params}

        fake_module = types.ModuleType("signet_auth")
        fake_module.SigningAgent = _SigningAgent  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "signet_auth", fake_module)

        with crewai_event_bus.scoped_handlers():
            listener = install(
                key_name="existing-agent",
                create_if_missing=False,
                audit=False,
            )
            listener._get_signing_agent()

        assert init_args["name"] == "existing-agent"
        assert init_args["kwargs"] == {}

    def test_hint_message_mentions_extra(self) -> None:
        assert "crewai[signet]" in _SIGNET_INSTALL_HINT
