"""Regression tests for third-party dependency pins that fix known advisories.

These tests guard against accidentally loosening or rolling back security-driven
version constraints declared in ``lib/crewai/pyproject.toml`` and in the
workspace-level ``pyproject.toml``. Each test references the advisory it exists
to defend against so future maintainers understand why a bound is required.
"""

from __future__ import annotations

import sys
from pathlib import Path


if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version


CREWAI_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _find_workspace_pyproject() -> Path | None:
    """Walk upward to locate the workspace root's ``pyproject.toml``.

    Returns ``None`` when the crewai package is being tested outside of the
    monorepo (e.g. installed into a separate virtualenv), in which case the
    workspace-level override is not applicable.
    """
    for candidate in Path(__file__).resolve().parents:
        pyproject = candidate / "pyproject.toml"
        if not pyproject.exists() or pyproject == CREWAI_PYPROJECT:
            continue
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        if "workspace" in data.get("tool", {}).get("uv", {}):
            return pyproject
    return None


def _load_dependencies(pyproject_path: Path) -> list[Requirement]:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    raw_deps: list[str] = data.get("project", {}).get("dependencies", [])
    return [Requirement(dep) for dep in raw_deps]


def _load_override_dependencies(pyproject_path: Path) -> list[Requirement]:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    raw_overrides: list[str] = (
        data.get("tool", {}).get("uv", {}).get("override-dependencies", [])
    )
    return [Requirement(dep) for dep in raw_overrides]


def _requirement_for(requirements: list[Requirement], name: str) -> Requirement:
    for req in requirements:
        if req.name == name:
            return req
    raise AssertionError(
        f"Expected dependency {name!r} in {pyproject_names(requirements)}"
    )


def pyproject_names(reqs: list[Requirement]) -> list[str]:
    return [r.name for r in reqs]


def _specifier_excludes_versions_below(
    specifier: SpecifierSet, threshold: Version
) -> bool:
    """Return True when no version strictly below ``threshold`` satisfies the set.

    We check a handful of representative versions near and below the threshold
    rather than probing every version that ever existed.
    """
    probes = [
        Version("0.0.0"),
        Version("0.9.13"),
        Version("0.9.30"),
        Version("0.10.0"),
        Version("0.11.0"),
        Version("0.11.5"),
    ]
    # Only probe versions strictly below the threshold.
    return not any(p in specifier for p in probes if p < threshold)


class TestUvDependencyPin:
    """Protects against regressing the fix for GHSA-pjjw-68hj-v9mw.

    ``uv`` versions prior to 0.11.6 are affected by a wheel ``RECORD`` path
    traversal vulnerability that can delete files outside the install prefix on
    uninstall. See https://github.com/advisories/GHSA-pjjw-68hj-v9mw and crewAI
    issue #5520.
    """

    PATCHED_VERSION = Version("0.11.6")

    def test_crewai_package_pins_patched_uv(self) -> None:
        requirements = _load_dependencies(CREWAI_PYPROJECT)
        uv_req = _requirement_for(requirements, "uv")

        assert self.PATCHED_VERSION in uv_req.specifier, (
            f"uv specifier {uv_req.specifier!s} must allow {self.PATCHED_VERSION} "
            "(the GHSA-pjjw-68hj-v9mw fix version)."
        )
        assert _specifier_excludes_versions_below(
            uv_req.specifier, self.PATCHED_VERSION
        ), (
            f"uv specifier {uv_req.specifier!s} still permits versions below "
            f"{self.PATCHED_VERSION}, which are affected by GHSA-pjjw-68hj-v9mw."
        )

    def test_workspace_override_pins_patched_uv(self) -> None:
        workspace_pyproject = _find_workspace_pyproject()
        if workspace_pyproject is None:
            pytest.skip(
                "Workspace pyproject.toml not found; running outside the monorepo."
            )

        overrides = _load_override_dependencies(workspace_pyproject)
        if "uv" not in pyproject_names(overrides):
            # The workspace override is belt-and-suspenders; if it disappears we
            # still rely on the package-level pin validated above. Don't fail
            # solely on its absence, but make the skip explicit so a future
            # maintainer who re-adds it gets full coverage automatically.
            pytest.skip(
                "Workspace does not declare a uv override-dependency; relying on "
                "the package-level pin."
            )

        uv_override = _requirement_for(overrides, "uv")
        assert self.PATCHED_VERSION in uv_override.specifier, (
            f"Workspace override {uv_override!s} must allow "
            f"{self.PATCHED_VERSION} (the GHSA-pjjw-68hj-v9mw fix version)."
        )
        assert _specifier_excludes_versions_below(
            uv_override.specifier, self.PATCHED_VERSION
        ), (
            f"Workspace override {uv_override!s} still permits versions below "
            f"{self.PATCHED_VERSION}, which are affected by GHSA-pjjw-68hj-v9mw."
        )
