# Use of generative AI

Most of this package was written by Claude Code (Anthropic) working from my specifications. That covers the library code, the test suite, the docstrings, the README, and the pages of this site. The code of conduct is the Contributor Covenant, adopted verbatim from upstream, not generated.

The tool was used agentically rather than as autocomplete. It reads the repository, proposes changes, runs `pytest` and `ruff` itself, and iterates until they pass. I set the scope for each session, reviewed what came back, and decided what landed.

The design is mine. That includes framing the package as validation rather than visualization, the three entry points (`qc()` returning a report, `report.passes()` as the non-raising check, `assert_qc()` as the pipeline gate), giving every plot function a `return_data=True` counterpart so the numbers behind a figure are usable, the error/warning/info severity tiers, Mann-Whitney U for MNAR/MAR classification, and the specific parameters: the 50% group-membership threshold in `missing_upset()`, and its 50-intersection figure cap with the full table still returned so nothing is silently dropped.

Correctness is checked by 152 tests on Python 3.10 through 3.13 across Linux and macOS, a ruff lint job, and a strict documentation build that fails on an unparseable docstring.

Commits written by the agent carry a `Co-Authored-By` trailer for it. That convention is reliable from 2026-07-30 onward and patchy before: four earlier commits lack the trailer despite being agent-assisted, `5cd2f0a`, the v0.2.0 release, most significantly. Read a missing trailer on an early commit as a gap in the record rather than as human authorship. The convention is written down in `CONTRIBUTING.md` now, and I have not backfilled it, because rewriting fourteen commits would move every release tag and throw away the public development history to make metadata look tidy.

Code written this way can carry subtle errors that review does not catch. If you find one, please open an issue.

Development ran from 2026-03-11 to 2026-08-11.

<!-- TODO (Arion): two statements the policy asks for, which are yours to make and
     not the agent's. Write them in your own words and delete this comment.

     1. What your review actually consisted of. Reading diffs before merge,
        running the checks locally, using the package on real FGCZ proteomics
        data. Be specific, do not overstate it.
     2. That all your correspondence during the review process is written by you.

     Also check the design paragraph above. The severity tiers and the
     Mann-Whitney choice predate this session, so I took them from
     docs/PLAN_validation_scope.md. If either was the agent's suggestion that you
     accepted rather than your own call, move it out of that paragraph. -->
