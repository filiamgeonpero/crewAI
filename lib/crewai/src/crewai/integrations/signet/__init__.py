"""Optional Signet integration for CrewAI.

`Signet <https://github.com/Prismer-AI/signet>`_ produces Ed25519-signed,
hash-chained receipts for AI agent tool calls. This integration registers a
:class:`BaseEventListener` that signs a receipt for every governed action
(structured tool, MCP tool execution, A2A delegation) using the paired
``Started``/``Completed`` events emitted by the CrewAI event bus.

The integration is installed as an optional extra::

    pip install 'crewai[signet]'

Then enabled with a single call::

    from crewai.integrations.signet import install

    listener = install(key_name="my-crew-agent")

After installation, every tool call, MCP tool execution, and A2A delegation
produces a signed receipt stored on the returned listener and (optionally)
appended to a local hash-chained audit log by ``signet-auth``.
"""

from __future__ import annotations

from crewai.integrations.signet.config import SignetConfig
from crewai.integrations.signet.listener import Receipt, SignetEventListener


__all__ = ["Receipt", "SignetConfig", "SignetEventListener", "install"]


def install(
    key_name: str,
    *,
    owner: str | None = None,
    audit: bool = True,
    policy_path: str | None = None,
    create_if_missing: bool = True,
    tool_events: bool = True,
    mcp_events: bool = True,
    a2a_events: bool = True,
    signing_agent: object | None = None,
) -> SignetEventListener:
    """Install the Signet event listener on the CrewAI event bus.

    Args:
        key_name: Signet ``SigningAgent`` identity name. If the identity does
            not exist yet and ``create_if_missing=True`` (the default), a new
            Ed25519 keypair is created and stored under ``~/.signet/keys/``.
        owner: Optional owner string used when creating a new identity.
        audit: If ``True``, ``signet-auth`` appends every receipt to its local
            hash-chained audit log at ``~/.signet/audit/``.
        policy_path: Optional path to a Signet policy file that is co-signed
            with every receipt.
        create_if_missing: If ``True``, create the identity on first use when
            no matching key is found. If ``False``, load the existing key only
            and raise if it cannot be found.
        tool_events: If ``True``, sign structured tool calls
            (``tool_usage_started`` / ``tool_usage_finished``).
        mcp_events: If ``True``, sign MCP tool executions
            (``mcp_tool_execution_started`` / ``mcp_tool_execution_completed``).
        a2a_events: If ``True``, sign A2A delegations
            (``a2a_delegation_started`` / ``a2a_delegation_completed``).
        signing_agent: Optional pre-built ``SigningAgent`` (or a test double
            exposing a ``sign(action, params=...)`` method). When provided,
            ``signet-auth`` is not imported and the extra is not required.

    Returns:
        The registered :class:`SignetEventListener`. Receipts can be inspected
        via ``listener.receipts``.
    """
    config = SignetConfig(
        key_name=key_name,
        owner=owner,
        audit=audit,
        policy_path=policy_path,
        create_if_missing=create_if_missing,
        tool_events=tool_events,
        mcp_events=mcp_events,
        a2a_events=a2a_events,
    )
    return SignetEventListener(config=config, signing_agent=signing_agent)
