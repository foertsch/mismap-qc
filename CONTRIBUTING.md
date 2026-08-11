# Contributing to mismap-qc

Bug reports, feature requests, and pull requests are all welcome. The package is maintained by one person, so please open an issue before starting a large change, to avoid duplicated work.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Development install

The project uses [uv](https://docs.astral.sh/uv/). Plain pip works too.

```bash
git clone https://github.com/foertsch/mismap-qc.git
cd mismap-qc
uv sync --extra dev
```

With pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The `dev` extra pulls in `pytest`, `ruff`, `plotly`, and `anndata`. `plotly` is needed for `missing_matrix_html()` and `anndata` for `from_anndata()`; both are optional at runtime, and their tests skip when absent.

## Running the checks

Both of these run in CI and both must pass:

```bash
pytest tests/ -q
ruff check .
```

Ruff is pinned to 0.16.0 in the workflow, and the enabled rules are declared explicitly in `pyproject.toml` under `[tool.ruff.lint]`. This is deliberate. Ruff's default rule set widened in 0.16 and turned a green build into 45 errors without a line of code changing. If you want to widen the selection, do it as its own pull request with the resulting fixes, not as a side effect of another change.

Expect around 130 passing tests with the `dev` extra installed, fewer in a bare environment where the optional-dependency tests skip. Tests named `*_realdata.py` need CPTAC data that is not in the repo and skip without it.

## Where code goes

| Module | Contents |
|---|---|
| `mismap_qc/validation.py` | `qc()`, `assert_qc()`, `MismapReport`, the threshold rule registry |
| `mismap_qc/plots.py` | Plot functions and their `_data_*` helpers |
| `mismap_qc/stats.py` | Pure-numeric helpers shared between `qc()` and the plots |
| `mismap_qc/io.py` | Readers such as `from_anndata()` |
| `mismap_qc/lod.py` | Limit-of-detection estimation |
| `mismap_qc/_core.py` | Constants and layout helpers |

Figure-returning functions go in `plots.py`, numeric-returning ones in `stats.py`. Analytical logic is shared through `stats.py` rather than duplicated between `qc()` and a plot wrapper.

New public names get re-exported in `mismap_qc/__init__.py`.

## Conventions

Read `CLAUDE.md` for the full set. The ones that matter most:

- `df` is always the first parameter, features as rows and samples as columns, `NaN` meaning missing.
- Everything after `df` is keyword-only.
- Include a `feature_type` parameter on anything that labels features, and get label text from `_get_feature_labels()`. Do not hardcode "gene" or "protein".
- Plot functions support `return_data=True`, returning `(Figure, DataFrame)` with the schema registered in `_RETURN_DATA_SCHEMAS` and covered by `tests/test_return_data_schemas.py`.
- Do not break the public API without a deprecation warning first.

## Tests

Every new function needs, at minimum:

1. returns the expected type
2. works with a `MultiIndex` column index
3. survives all-missing input
4. survives all-present input
5. writes a file when `save=` is given

Close figures with `plt.close("all")` at the end of each test.

## Pull requests

1. Branch off `main`.
2. Keep the change scoped to one concern. Separate concerns go in separate pull requests.
3. Run `pytest` and `ruff check .` before pushing.
4. Update `CHANGELOG.md` under the unreleased or upcoming version heading.
5. Open the pull request against `main`. CI runs pytest on Python 3.10 to 3.13 across Ubuntu and macOS, plus the ruff job.

Pull requests are squash-merged.

## If you used an AI coding tool

Say so, and mark it in the commit. This project uses the standard trailer:

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

Substitute whichever tool you used. Add it to commits the tool wrote, and leave it off commits you wrote yourself.

No judgement attached to using one. Most of this package was written that way, and the project-level disclosure is at [Use of generative AI](https://foertsch.github.io/mismap-qc/generative-ai-use/). What matters is that the record is accurate, so please do not apply the trailer by default to everything, and do not omit it on a change a tool actually wrote.

## Releases

Maintainer only:

1. Bump the version in `pyproject.toml` and `mismap_qc/__init__.py`. They are checked against each other by `tests/test_package_metadata.py`, so both have to move.
2. Update `CHANGELOG.md` and the `version` field in `CITATION.cff`.
3. Merge, then tag `vX.Y.Z` and create the GitHub release.
4. `uv build`, then `uv publish`.

The sdist contents are an explicit allowlist in `[tool.hatch.build.targets.sdist]`. Anything new that has to ship in the source distribution must be added there, otherwise it is silently left out.

## Reporting bugs

Open an issue with the mismap-qc version, your Python version, and a minimal example. A small DataFrame that reproduces the problem is worth more than a description of it.
