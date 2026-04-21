"""First-party integrations for CrewAI.

Each subpackage is opt-in and must lazily import any third-party dependencies
so that importing ``crewai.integrations`` has no runtime cost for users who
have not installed the corresponding extra.
"""
