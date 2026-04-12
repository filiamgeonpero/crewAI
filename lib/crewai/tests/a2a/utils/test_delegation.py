"""Tests for A2A delegation utilities."""

from __future__ import annotations

from contextlib import asynccontextmanager, ExitStack
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from a2a.client import Client
from a2a.types import AgentCapabilities, AgentCard

from crewai.a2a.updates.polling.handler import PollingHandler
from crewai.a2a.updates.streaming.handler import StreamingHandler
from crewai.a2a.utils.delegation import get_handler, _aexecute_a2a_delegation_impl
from crewai.a2a.utils.transport import NegotiatedTransport


class TestGetHandler:
    """Tests for the get_handler helper."""

    def test_returns_streaming_handler_when_config_is_none(self) -> None:
        assert get_handler(None) is StreamingHandler

    def test_returns_polling_handler_for_polling_config(self) -> None:
        from crewai.a2a.updates import PollingConfig

        assert get_handler(PollingConfig()) is PollingHandler


def _make_agent_card(streaming: bool | None) -> AgentCard:
    """Build a minimal AgentCard with the given streaming capability."""
    capabilities = AgentCapabilities(streaming=streaming)
    return AgentCard(
        name="test-agent",
        description="A test agent",
        url="http://localhost:9999/",
        version="1.0.0",
        capabilities=capabilities,
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain"],
        skills=[],
    )


_TASK_RESULT = {"status": "completed", "result": "done", "history": []}


def _make_shared_patches(agent_card: AgentCard) -> list:
    """Return the common patches used across the delegation tests."""
    mock_client = MagicMock(spec=Client)

    @asynccontextmanager
    async def _fake_client_ctx(*args: Any, **kwargs: Any):
        yield mock_client

    negotiated = NegotiatedTransport(
        transport="JSONRPC",
        url=agent_card.url,
        source="server_card",
    )

    return [
        patch(
            "crewai.a2a.utils.delegation._afetch_agent_card_cached",
            new=AsyncMock(return_value=agent_card),
        ),
        patch("crewai.a2a.utils.delegation.validate_auth_against_agent_card"),
        patch(
            "crewai.a2a.utils.delegation.validate_required_extensions",
            return_value=[],
        ),
        patch(
            "crewai.a2a.utils.delegation.negotiate_transport",
            return_value=negotiated,
        ),
        patch(
            "crewai.a2a.utils.delegation.negotiate_content_types",
            return_value=MagicMock(output_modes=None),
        ),
        patch(
            "crewai.a2a.utils.delegation._prepare_auth_headers",
            new=AsyncMock(return_value=({}, None)),
        ),
        patch("crewai.a2a.utils.delegation.crewai_event_bus"),
        patch(
            "crewai.a2a.utils.delegation._create_a2a_client",
            side_effect=_fake_client_ctx,
        ),
    ]


async def _call_impl(agent_card: AgentCard, updates=None) -> None:
    await _aexecute_a2a_delegation_impl(
        endpoint="http://localhost:9999/",
        auth=None,
        timeout=30,
        task_description="test task",
        context=None,
        context_id=None,
        task_id=None,
        reference_task_ids=None,
        metadata=None,
        extensions=None,
        conversation_history=[],
        is_multiturn=False,
        turn_number=1,
        agent_branch=None,
        agent_id=None,
        agent_role=None,
        response_model=None,
        updates=updates,
    )


class TestStreamingFallback:
    """Tests that the delegation respects the agent card's streaming capability."""

    @pytest.mark.asyncio
    async def test_uses_polling_when_agent_card_says_no_streaming(self) -> None:
        """When streaming=False and updates is None, PollingHandler should be used."""
        agent_card = _make_agent_card(streaming=False)

        with ExitStack() as stack:
            for p in _make_shared_patches(agent_card):
                stack.enter_context(p)
            mock_polling = stack.enter_context(
                patch.object(PollingHandler, "execute", new=AsyncMock(return_value=_TASK_RESULT))
            )
            mock_streaming = stack.enter_context(
                patch.object(StreamingHandler, "execute", new=AsyncMock(return_value=_TASK_RESULT))
            )
            await _call_impl(agent_card, updates=None)

        mock_polling.assert_called_once()
        mock_streaming.assert_not_called()

    @pytest.mark.asyncio
    async def test_uses_streaming_when_agent_card_says_streaming(self) -> None:
        """When streaming=True and updates is None, StreamingHandler should be used."""
        agent_card = _make_agent_card(streaming=True)

        with ExitStack() as stack:
            for p in _make_shared_patches(agent_card):
                stack.enter_context(p)
            mock_polling = stack.enter_context(
                patch.object(PollingHandler, "execute", new=AsyncMock(return_value=_TASK_RESULT))
            )
            mock_streaming = stack.enter_context(
                patch.object(StreamingHandler, "execute", new=AsyncMock(return_value=_TASK_RESULT))
            )
            await _call_impl(agent_card, updates=None)

        mock_streaming.assert_called_once()
        mock_polling.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicit_streaming_config_overrides_agent_card(self) -> None:
        """Explicitly passing StreamingConfig keeps StreamingHandler even when agent card says no streaming."""
        from crewai.a2a.updates.streaming.config import StreamingConfig

        agent_card = _make_agent_card(streaming=False)

        with ExitStack() as stack:
            for p in _make_shared_patches(agent_card):
                stack.enter_context(p)
            mock_polling = stack.enter_context(
                patch.object(PollingHandler, "execute", new=AsyncMock(return_value=_TASK_RESULT))
            )
            mock_streaming = stack.enter_context(
                patch.object(StreamingHandler, "execute", new=AsyncMock(return_value=_TASK_RESULT))
            )
            await _call_impl(agent_card, updates=StreamingConfig())

        # explicit config overrides agent card; streaming handler is used
        mock_streaming.assert_called_once()
        mock_polling.assert_not_called()
