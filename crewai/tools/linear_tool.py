import os
from enum import Enum
from typing import Any, Type

import httpx
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

LINEAR_API_URL = "https://api.linear.app/graphql"


class LinearAction(str, Enum):
    MY_ISSUES = "my_issues"
    LIST_TEAMS = "list_teams"
    LIST_PROJECTS = "list_projects"


class LinearToolInput(BaseModel):
    action: LinearAction = Field(
        description=(
            "Action to perform: "
            "'my_issues' — fetch issues assigned to the authenticated user; "
            "'list_teams' — list all teams in the workspace; "
            "'list_projects' — list all projects in the workspace."
        )
    )
    first: int = Field(
        default=25,
        ge=1,
        le=250,
        description="Maximum number of records to return (1–250).",
    )


_QUERIES: dict[LinearAction, str] = {
    LinearAction.MY_ISSUES: """
        query MyIssues($first: Int!) {
          viewer {
            assignedIssues(first: $first, orderBy: updatedAt) {
              nodes {
                id
                identifier
                title
                state { name }
                priority
                url
                updatedAt
              }
            }
          }
        }
    """,
    LinearAction.LIST_TEAMS: """
        query ListTeams($first: Int!) {
          teams(first: $first) {
            nodes {
              id
              name
              key
              description
            }
          }
        }
    """,
    LinearAction.LIST_PROJECTS: """
        query ListProjects($first: Int!) {
          projects(first: $first, orderBy: updatedAt) {
            nodes {
              id
              name
              description
              state
              url
              updatedAt
            }
          }
        }
    """,
}


def _extract(action: LinearAction, data: dict) -> list[dict]:
    if action == LinearAction.MY_ISSUES:
        return data["viewer"]["assignedIssues"]["nodes"]
    if action == LinearAction.LIST_TEAMS:
        return data["teams"]["nodes"]
    if action == LinearAction.LIST_PROJECTS:
        return data["projects"]["nodes"]
    return []


class LinearTool(BaseTool):
    name: str = "Linear API Tool"
    description: str = (
        "Interact with the Linear project management API. "
        "Supports fetching your assigned issues, listing teams, and listing projects."
    )
    args_schema: Type[BaseModel] = LinearToolInput

    def _run(self, action: LinearAction, first: int = 25) -> Any:
        api_key = os.environ.get("LINEAR_API_KEY", "")
        if not api_key:
            raise EnvironmentError("LINEAR_API_KEY environment variable is not set.")

        query = _QUERIES[action]
        payload = {"query": query, "variables": {"first": first}}
        headers = {
            "Authorization": api_key,
            "Content-Type": "application/json",
        }

        response = httpx.post(
            LINEAR_API_URL,
            json=payload,
            headers=headers,
            timeout=15,
        )
        response.raise_for_status()

        body = response.json()
        if "errors" in body:
            raise RuntimeError(f"Linear API errors: {body['errors']}")

        return _extract(action, body["data"])
