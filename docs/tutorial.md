# Tutorial

## CPTAC lung adenocarcinoma proteomics

The worked tutorial runs against public CPTAC LUAD data, roughly 100 tumour and normal samples, and shows how missingness clusters by tumour/normal status.

[Open the notebook on GitHub :material-open-in-new:](https://github.com/foertsch/mismap-qc/blob/main/examples/cptac_proteomics.ipynb)

It covers:

1. Loading the CPTAC matrix into the features-by-samples layout mismap-qc expects
2. The nullity matrix with tumour/normal annotation strips
3. Per-group completeness
4. The detection threshold curve
5. Abundance density of detected against missing values
6. The validation report from `qc()`

The data is not in the repository. The notebook's first cell documents where to get it.

## Toy data

For a quick look with no download, the repository ships a generator:

```bash
uv run make_toy_data.py
```

That writes `data/toy_rnaseq.csv`: 80 genes by 30 samples with structured missingness across six groups. `demo.py` renders the full plot set from it:

```bash
uv run demo.py
```

Both scripts use [PEP 723](https://peps.python.org/pep-0723/) inline dependencies, so neither needs a virtual environment when run with uv.
