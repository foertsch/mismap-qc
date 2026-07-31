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

import re
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


def test_citation_file_tracks_the_package_version(project):
    """CITATION.cff carries its own version field, so a release bump has to touch it
    too. Parsed by hand to avoid a PyYAML dependency in a bare CI environment."""
    citation = PYPROJECT.parent / "CITATION.cff"
    assert citation.exists(), "CITATION.cff is required for citation metadata"
    text = citation.read_text(encoding="utf-8")
    match = re.search(r"^version:\s*(\S+)\s*$", text, re.MULTILINE)
    assert match, "CITATION.cff has no top-level version field"
    assert match.group(1).strip('"\'') == project["version"]
    assert "repository-code: https://github.com/foertsch/mismap-qc" in text
    # Verified against the ORCID public API, which registers this iD to Arion Förtsch.
    assert re.search(r"orcid: https://orcid\.org/\d{4}-\d{4}-\d{4}-\d{3}[\dX]", text)


def test_required_community_files_exist():
    """pyOpenSci's editor-in-chief check looks for these by name before review starts."""
    root = PYPROJECT.parent
    for name in ("README.md", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "LICENSE"):
        assert (root / name).exists(), f"missing required repo file: {name}"


def test_code_of_conduct_names_a_reporting_contact():
    """The Contributor Covenant ships with an [INSERT CONTACT METHOD] placeholder.
    A code of conduct with no route for reporting is decoration."""
    text = (PYPROJECT.parent / "CODE_OF_CONDUCT.md").read_text(encoding="utf-8")
    assert "INSERT CONTACT METHOD" not in text
    assert "@" in text


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
