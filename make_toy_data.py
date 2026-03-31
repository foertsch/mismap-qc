# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pandas",
# ]
# ///
"""
Generate a toy RNA-Seq dataset with structured missingness for testing
the mismap_qc visualization.

Output: data/toy_rnaseq.csv (MultiIndex columns preserved via header rows)
"""
import numpy as np
import pandas as pd
from pathlib import Path


def make_toy_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Sample metadata
    medium_types = ["Fresh", "Fresh", "Fresh", "Conditioned", "Conditioned", "Conditioned"]
    medium_conds = ["SF", "FBS", "AS", "SF", "FBS", "AS"]
    samples_per_group = 5
    oms_ids = [f"GS.{1290 + i}" for i in range(samples_per_group)]

    col_tuples = []
    for mt, mc in zip(medium_types, medium_conds):
        for oid in oms_ids:
            col_tuples.append((mt, mc, oid))

    columns = pd.MultiIndex.from_tuples(
        col_tuples, names=["Medium_Type", "Medium_Condition", "OMS_ID"]
    )

    # Gene names (mix of realistic-looking names)
    n_genes = 80
    gene_prefixes = [
        "PDGF-BB", "YKL-40", "G-CSF", "VEGF", "M-CSF", "MMP-2", "CXCL-9",
        "IL-6", "TNF-a", "IFN-g", "CCL2", "CXCL10", "IL-1b", "IL-10",
        "TGF-b1", "HGF", "FGF-2", "EGF", "ANGPT1", "ANGPT2",
    ]
    genes = []
    for i in range(n_genes):
        if i < len(gene_prefixes):
            genes.append(gene_prefixes[i])
        else:
            genes.append(f"Gene_{i + 1:03d}")

    # Simulate log-normal expression
    data = rng.lognormal(mean=3.0, sigma=1.5, size=(n_genes, len(col_tuples)))

    # Introduce structured missingness per group
    miss_rates = {
        ("Fresh", "SF"): 0.35,
        ("Fresh", "FBS"): 0.12,
        ("Fresh", "AS"): 0.22,
        ("Conditioned", "SF"): 0.30,
        ("Conditioned", "FBS"): 0.10,
        ("Conditioned", "AS"): 0.20,
    }

    for j, (mt, mc, _) in enumerate(col_tuples):
        rate = miss_rates.get((mt, mc), 0.15)
        mask = rng.random(n_genes) < rate
        data[mask, j] = np.nan

    # A few genes that are always missing in SF (structured block)
    sf_cols = [j for j, (_, mc, _) in enumerate(col_tuples) if mc == "SF"]
    always_missing_genes = list(range(5, 12))
    for j in sf_cols:
        for g in always_missing_genes:
            if rng.random() < 0.85:
                data[g, j] = np.nan

    df = pd.DataFrame(data, index=genes, columns=columns)
    return df


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)

    df = make_toy_data()
    df.to_csv(out_dir / "toy_rnaseq.csv")
    print(f"Wrote {df.shape[0]} genes x {df.shape[1]} samples to data/toy_rnaseq.csv")
    print(f"Overall missingness: {df.isnull().sum().sum() / df.size:.1%}")
