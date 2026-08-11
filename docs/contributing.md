# Contributing

Bug reports and pull requests are welcome. The package has one maintainer, so please open an issue before starting a large change.

The full guide lives in the repository:

- [CONTRIBUTING.md](https://github.com/foertsch/mismap-qc/blob/main/CONTRIBUTING.md): development install, running the checks, where code belongs, conventions, the pull request flow, release steps
- [Code of Conduct](https://github.com/foertsch/mismap-qc/blob/main/CODE_OF_CONDUCT.md)
- [Issue tracker](https://github.com/foertsch/mismap-qc/issues)

## Development install

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

## Running the checks

Both run in CI and both must pass:

```bash
pytest tests/ -q
ruff check .
```

## Building these docs

```bash
uv run --extra docs mkdocs serve
```

That serves the site at `http://127.0.0.1:8000` with live reload. `mkdocs build --strict` is what CI runs, and it treats warnings such as a broken internal link as errors.

## Citation

If you use mismap-qc in published work, see [CITATION.cff](https://github.com/foertsch/mismap-qc/blob/main/CITATION.cff), which GitHub also renders as "Cite this repository".
