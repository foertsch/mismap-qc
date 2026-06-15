# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "matplotlib",
#     "scipy",
#     "pandas",
#     "plotly",
# ]
# ///
"""Demo: showcases the v0.2.0 mismap_qc API on a toy RNA-Seq matrix.

Covers the validation entry points (qc, assert_qc, MismapReport), a couple
of plot functions, and the return_data=True flag.
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from mismap_qc import (
    MismapQCFailure,
    MismapQCWarning,
    assert_qc,
    completeness_bars,
    detection_waterfall,
    missing_matrix,
    missing_matrix_html,
    qc,
)

data_path = Path(__file__).parent / "data" / "toy_rnaseq.csv"
out_dir = Path(__file__).parent / "output"
out_dir.mkdir(exist_ok=True)

df = pd.read_csv(data_path, index_col=0, header=[0, 1, 2])
n_genes, n_samples = df.shape
overall = df.isnull().sum().sum() / df.size

# ---------------------------------------------------------------------------
# 1. Run the full validation battery
# ---------------------------------------------------------------------------
report = qc(df, group_level="Medium_Condition")
print(repr(report))
print()
print(report.summary())
print()

# ---------------------------------------------------------------------------
# 2. Gate against threshold rules
#    Loose thresholds pass; the strict block raises.
# ---------------------------------------------------------------------------
assert_qc(df, thresholds={
    "min_sample_completeness": 0.1,
    "max_mnar_fraction": 0.9,
})
print("assert_qc passed with loose thresholds")

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MismapQCWarning)
        assert_qc(df, thresholds={"min_sample_completeness": 0.99})
except MismapQCFailure as e:
    first_line = str(e).splitlines()[0]
    print(f"assert_qc raised with strict thresholds: {first_line}")
print()

# ---------------------------------------------------------------------------
# 3. Static missing-data matrix
# ---------------------------------------------------------------------------
fig = missing_matrix(
    df,
    title="Gene Detection Matrix -- RNA-Seq QC",
    subtitle=f"{n_genes} genes x {n_samples} samples | {overall:.0%} missing overall",
    completeness_threshold=0.5,
    group_summary="Medium_Condition",
    legend_loc="upper right",
    save=str(out_dir / "demo_full.png"),
)
plt.close(fig)
print("Saved demo_full.png")

# ---------------------------------------------------------------------------
# 4. Detection waterfall with return_data=True -- get the figure AND the numbers
# ---------------------------------------------------------------------------
fig, waterfall_df = detection_waterfall(df, return_data=True)
fig.savefig(out_dir / "demo_waterfall.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(
    f"Saved demo_waterfall.png; top 3 most-detected genes: "
    f"{list(waterfall_df.head(3)['feature'])}"
)

# ---------------------------------------------------------------------------
# 5. Per-group completeness bars
# ---------------------------------------------------------------------------
fig, comp_df = completeness_bars(
    df, group_level="Medium_Condition", return_data=True,
)
fig.savefig(out_dir / "demo_completeness.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved demo_completeness.png; per-group completeness:")
print(comp_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 6. Split-by-condition matrix and interactive HTML
# ---------------------------------------------------------------------------
fig = missing_matrix(
    df,
    title="Gene Detection -- Split by Condition",
    split_by="Medium_Condition",
    annotation_levels=[0],
    save=str(out_dir / "demo_split.png"),
)
plt.close(fig)
print("Saved demo_split.png")

missing_matrix_html(
    df,
    title="Gene Detection Matrix (Interactive)",
    subtitle=f"{n_genes} genes x {n_samples} samples | {overall:.0%} missing overall",
    completeness_threshold=0.5,
    save=str(out_dir / "demo_interactive.html"),
)
print("Saved demo_interactive.html")
