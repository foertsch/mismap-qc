"""Packaging metadata guards.

These tests exist because of two real defects. (1) `pyproject.toml` and
`mismap_qc.__version__` drifted apart, so the version the package reported at
runtime was not the version being packaged. (2) The published PyPI metadata
pointed at a repository URL that did not resolve, and described the package as a
missing-data *matrix* rather than *validation* — the framing the project is
reviewed under. Both are invisible to the rest of the test suite because neither
affects behaviour, only what users and reviewers see.

Skipped on Python 3.10, which has no stdlib TOML parser.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import mismap_qc

tomllib = pytest.importorskip("tomllib", reason="stdlib TOML parser needs 3.11+")

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

pytestmark = pytest.mark.skipif(
    not PYPROJECT.exists(),
    reason="running against an installed package, not a source checkout",
)


@pytest.fixture(scope="module")
def pyproject() -> dict:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


@pytest.fixture(scope="module")
def project(pyproject) -> dict:
    return pyproject["project"]


def test_version_matches_package(project):
    assert project["version"] == mismap_qc.__version__


def test_author_has_email(project):
    # pyOpenSci requires contactable author metadata (authors, emails, URLs).
    authors = project["authors"]
    assert authors, "at least one author required"
    assert all(author.get("email") for author in authors)


def test_required_project_urls(project):
    urls = project["urls"]
    for key in ("Repository", "Documentation", "Issues", "Changelog"):
        assert key in urls, f"missing [project.urls] entry: {key}"
    # The published 0.1.0 metadata pointed at a nonexistent account.
    assert urls["Repository"] == "https://github.com/foertsch/mismap-qc"


def test_scope_framing_is_validation_not_visualization(project):
    """The package is reviewed as data validation. Metadata must not say otherwise."""
    assert "validation" in project["description"].lower()
    assert "visualization" not in project["keywords"]
    assert not any("Visualization" in c for c in project["classifiers"])


def test_sdist_is_explicitly_scoped(pyproject):
    """Hatchling's default sdist sweeps in the whole working tree (2.2 MB of demo
    PNGs, notebooks, CLAUDE.md, .claude/). Keep the allowlist, and keep tests in it
    so downstream packagers can verify a build."""
    include = pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    assert "mismap_qc/" in include
    assert "tests/" in include


def test_classifiers_cover_supported_pythons(project):
    """Every version in requires-python has a matching classifier."""
    declared = {
        c.rsplit(" :: ", 1)[-1]
        for c in project["classifiers"]
        if c.startswith("Programming Language :: Python :: 3.")
    }
    assert {"3.10", "3.11", "3.12", "3.13"} <= declared
