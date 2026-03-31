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
"""Demo: showcases all major features of mismap_qc."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from mismap_qc import missing_matrix, missing_matrix_html

data_path = Path(__file__).parent / "data" / "toy_rnaseq.csv"
out_dir = Path(__file__).parent / "output"
out_dir.mkdir(exist_ok=True)

df = pd.read_csv(data_path, index_col=0, header=[0, 1, 2])
n_genes, n_samples = df.shape
overall = df.isnull().sum().sum() / df.size

# 1. Full-featured static plot with group summary printed to console
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

# 2. Split by Medium_Condition
fig = missing_matrix(
    df,
    title="Gene Detection -- Split by Condition",
    split_by="Medium_Condition",
    annotation_levels=[0],
    save=str(out_dir / "demo_split.png"),
)
plt.close(fig)
print("Saved demo_split.png")

# 3. Interactive HTML
missing_matrix_html(
    df,
    title="Gene Detection Matrix (Interactive)",
    subtitle=f"{n_genes} genes x {n_samples} samples | {overall:.0%} missing overall",
    completeness_threshold=0.5,
    save=str(out_dir / "demo_interactive.html"),
)
print("Saved demo_interactive.html")
