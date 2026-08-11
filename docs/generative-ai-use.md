# Use of generative AI

pyOpenSci's [generative AI policy](https://www.pyopensci.org/software-peer-review/our-process/generative-ai-policy.html) asks authors to disclose which parts of a package were produced with generative AI tools, at what scale, and by what method. This page is that disclosure. It is published rather than kept in the submission thread because the answer is substantial and anyone evaluating the package should be able to find it.

## Short version

Nearly all of the code, tests, and prose in mismap-qc was written by an AI coding agent, working from my specifications, under my review, in an agentic loop rather than as autocomplete. The design decisions, the scope, the statistical choices, and the judgement calls are mine. The typing is not.

## Which tool, and how it was used

**Claude Code** (Anthropic), used agentically: the model reads the repository, proposes edits, runs `pytest` and `ruff` itself, and iterates until the checks pass. That is materially different from tab-completion or from pasting snippets out of a chat window, and it is the mode the policy asks to be named specifically.

Development ran from the first commit on **2026-03-11** to **2026-08-11**, across roughly a dozen working sessions.

## What was generated, by component

| Component | Extent | Notes |
|---|---|---|
| Library code (`mismap_qc/`, 3,380 lines) | Nearly all agent-written | From my specifications and design decisions. Several rounds of restructuring, including the split from a single 3,031-line module into seven submodules. |
| Test suite (`tests/`, 1,746 lines, 152 tests) | Nearly all agent-written | The testing conventions in `CLAUDE.md` (the five cases every plot function must cover) were set by me and applied by the agent. |
| Docstrings | Nearly all agent-written | NumPy style. The API reference on this site generates from them. |
| README, `CONTRIBUTING.md`, this site's pages | Agent-drafted, edited by me | |
| `CODE_OF_CONDUCT.md` | Not generated | Contributor Covenant 2.1, adopted verbatim from upstream. |
| Design and planning documents (`docs/PLAN_*.md`) | Agent-drafted from my requirements | These record decisions we worked through in conversation. |
| Commit messages and pull request descriptions | Agent-drafted, reviewed by me | Commits from sessions where the agent authored the message carry a `Co-Authored-By` trailer. |

Across the whole history: **13,286 insertions, 1,269 deletions, 15 commits on `main`**.

## An honest caveat about the commit trailers

Eleven of the fifteen commits on `main` carry a `Co-Authored-By: Claude` trailer. **Do not read the other four as human-written.** The trailer convention was applied inconsistently across sessions, and one of the four without it (`5cd2f0a`, the v0.2.0 release, 4,074 insertions) was in fact one of the most heavily agent-assisted changes in the project. The trailers undercount rather than overcount. Assume agent involvement throughout unless a file says otherwise.

## What was mine

The parts a reviewer should hold me to:

- **Scope and framing.** Deciding this is a data *validation* package with plots attached, rather than a visualization package, and holding that line through the API and the metadata.
- **API design.** The three entry points (`qc()` returning a report, `report.passes()` as the non-raising check, `assert_qc()` as the pipeline gate), the frozen `MismapReport`, the decision to give every plot a `return_data=True` counterpart, and the error/warning/info severity tiers.
- **Statistical choices.** Mann-Whitney U for MNAR/MAR classification, the threshold rules and their defaults, and the specific parameters: the 50% group-membership threshold in `missing_upset()`, the 50-intersection figure cap with the full table still returned so nothing is silently hidden.
- **Tooling decisions.** Pinning the ruff rule set explicitly after a ruff release silently changed what CI enforced. Choosing a generated documentation site over hand-maintained README tables, because the tables duplicate the docstrings and drift.
- **Judgement calls that went against the agent's suggestion**, including declining a script that would have inflated the repository's view-count badge.

## Review

<!-- Arion: confirm or correct both statements below before submitting. They are
     yours to make, not the agent's, and the policy asks for them explicitly. -->

- [ ] I have reviewed the code in this package. *(Add here, in your own words, what your review actually consisted of: reading diffs before merge, running the checks, exercising the functions on real FGCZ data, and so on. Be specific and do not overstate it.)*
- [ ] All correspondence in the review process is written by me, not generated.

## Why this is disclosed at this length

The policy exists partly to keep reviewers from spending volunteer time on code the author did not engage with. The honest position is that this package would not exist in this form without an AI agent, and that the engineering judgement, the domain decisions, and the responsibility for it are mine. Reviewers should weigh it knowing that.
