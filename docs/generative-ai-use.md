# Use of generative AI

mismap-qc was authored by Arion Förtsch, including the scope, the design and the statistical choices.

The implementation was written largely by Claude Code (Anthropic), working from written specifications and reviewed by the author before it landed. The models were Claude Opus 4.6 and Claude Sonnet 4.6 in March and April 2026, and Claude Opus 5 from July 2026, recorded per commit in the `Co-Authored-By` trailers.

Concretely, those decisions include framing this as a validation package with plots attached rather than a visualization package, and holding that line through the API and the metadata. The three entry points: `qc()` returning a report, `report.passes()` as the non-raising check, `assert_qc()` as the pipeline gate. Giving the plot functions a `return_data=True` counterpart, so the numbers behind a figure are usable rather than locked in an image. The error, warning and info severity tiers. Mann-Whitney U for MNAR/MAR classification. The specific parameters: the 50% group-membership threshold in `missing_upset()`, and its 50-intersection figure cap with the full table still returned so nothing is silently dropped. Pinning the ruff rule set explicitly after a ruff release changed what CI enforced without a line of code changing. Choosing a generated documentation site over hand-maintained README tables, because those tables duplicate the docstrings and drift.

The tool was used agentically rather than as autocomplete. It reads the repository, proposes changes, runs `pytest` and `ruff` itself, and iterates until they pass. Its output covers the library code, the test suite, the docstrings, the README and the pages of this site. The code of conduct is the Contributor Covenant, adopted verbatim from upstream, not generated. Each working session had a scope set in advance, and what landed was decided at review.

Correctness is checked by the test suite on Python 3.10 through 3.13 across Linux and macOS, a ruff lint job, and a strict documentation build that fails on an unparseable docstring. Every example in the public docstrings is executed as part of that suite.

Commits written by the agent carry a `Co-Authored-By` trailer for it, a convention written down in `CONTRIBUTING.md`. It is reliable from 2026-07-30 onward and patchy before: four earlier commits lack the trailer despite being agent-assisted, `5cd2f0a`, the v0.2.0 release, most significantly. Read a missing trailer on an early commit as a gap in the record rather than as human authorship. The trailers have not been backfilled, because rewriting fourteen commits would move every release tag and throw away the public development history to make metadata look tidy.

Development ran from 2026-03-11 to 2026-08-11. If you find a bug, please open an issue.
