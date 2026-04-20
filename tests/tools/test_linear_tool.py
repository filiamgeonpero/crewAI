"""
Test script for LinearTool — runs against the real Linear API.

Usage:
    LINEAR_API_KEY=lin_api_xxxxxxxxxxxx python tests/tools/test_linear_tool.py

Set LINEAR_API_KEY to your actual Personal API key from:
  https://linear.app/settings/api  (Profile → API → Personal API keys)
"""

import json
import os
import sys

from crewai.tools.linear_tool import LinearAction, LinearTool


def pretty(data: object) -> str:
    return json.dumps(data, indent=2, default=str)


def run_test(tool: LinearTool, label: str, action: LinearAction, first: int = 5) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    try:
        result = tool._run(action=action, first=first)
        if not result:
            print("  (no records returned)")
        else:
            print(pretty(result))
    except Exception as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)


def main() -> None:
    if not os.environ.get("LINEAR_API_KEY"):
        print(
            "ERROR: Set LINEAR_API_KEY before running.\n"
            "  export LINEAR_API_KEY=lin_api_xxxxxxxxxxxx",
            file=sys.stderr,
        )
        sys.exit(1)

    tool = LinearTool()

    run_test(tool, "My assigned issues (up to 5)", LinearAction.MY_ISSUES, first=5)
    run_test(tool, "Teams (up to 10)", LinearAction.LIST_TEAMS, first=10)
    run_test(tool, "Projects (up to 10)", LinearAction.LIST_PROJECTS, first=10)

    print(f"\n{'─' * 60}")
    print("  All tests complete.")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    main()
