"""Configuration for the optional Signet integration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SignetConfig(BaseModel):
    """User-facing configuration for the Signet listener.

    Attributes:
        key_name: Signet ``SigningAgent`` identity name.
        owner: Optional owner string used when creating a new identity.
        audit: Whether signet-auth should append receipts to its hash-chained
            audit log.
        policy_path: Optional path to a Signet policy file that is co-signed
            with every receipt.
        create_if_missing: Whether to create the identity on first use when no
            matching key is found.
        tool_events: Whether to sign structured tool calls.
        mcp_events: Whether to sign MCP tool executions.
        a2a_events: Whether to sign A2A delegations.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    key_name: str = Field(..., min_length=1)
    owner: str | None = None
    audit: bool = True
    policy_path: str | None = None
    create_if_missing: bool = True
    tool_events: bool = True
    mcp_events: bool = True
    a2a_events: bool = True
