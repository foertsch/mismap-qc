# Use of generative AI

mismap-qc was authored by Arion Förtsch, including the scope, the design and the statistical choices.

The implementation was written largely by Claude Code (Anthropic), working from written specifications and reviewed by the author before it landed. The models were Claude Opus 4.6 and Claude Sonnet 4.6 in March and April 2026, and Claude Opus 5 from July 2026, recorded per commit in the `Co-Authored-By` trailers.

## What the tool produced

Every module in the package was wholly or primarily generated: `_core.py`, `stats.py`, `validation.py`, `plots.py`, `io.py`, `lod.py`, `__init__.py`. So was the test suite. So were the NumPy-style docstrings that the API reference is built from, the README, and the pages of this site. The code of conduct is the Contributor Covenant, adopted verbatim from upstream.

The tool was used agentically, not as line completion. It reads the repository, proposes changes, runs `pytest` and `ruff` itself, and iterates until they pass. Each working session had its scope set in advance, and what landed was decided at review.

## What the author decided

- The scope: a validation package with plots attached, not a visualization package. Held through the API and the metadata.
- The three entry points. `qc()` returns a report, `report.passes()` checks without raising, `assert_qc()` gates a pipeline.
- A `return_data=True` counterpart on the plot functions, so the numbers behind a figure stay usable.
- Error, warning and info severity tiers on the threshold rules.
- One-sided Mann-Whitney U as the basis for MNAR/MAR classification, and the rule defaults.
- The 50% group-membership threshold in `missing_upset()`, and its 50-intersection figure cap with the full table still returned so nothing is dropped silently.
- Pinning the ruff rule set, after a ruff release changed what CI enforced with no code change.
- A generated documentation site over hand-maintained README tables, which duplicate the docstrings and drift.

## Review

The author read the diffs before merging, uses mismap-qc on real proteomics datasets at the Functional Genomics Center Zurich, and owns the statistical choices above. Agent proposals were declined on disagreement: `upsetplot` was taken as an optional dependency instead of hand-rolling the UpSet layout, and a proposed script that would have inflated this repository's view-count badge was refused.

Correctness is checked by the test suite on Python 3.10 through 3.13 across Linux and macOS, a ruff lint job, and a strict documentation build that fails on an unparseable docstring. Every example in the public docstrings is executed as part of that suite.

## Commit trailers

Commits written by the agent carry a `Co-Authored-By` trailer, a convention written down in `CONTRIBUTING.md`. It is reliable from 2026-07-30 onward and patchy before: four earlier commits lack the trailer despite being agent-assisted, `5cd2f0a`, the v0.2.0 release, most significantly. A missing trailer on an early commit is a gap in the record, not evidence of human authorship. The trailers have not been backfilled, because rewriting fourteen commits would move every release tag and discard the public development history to make metadata look tidy.

Development began on 2026-03-11 and is ongoing; the commit history is the authoritative record. If you find a bug, please open an issue.
