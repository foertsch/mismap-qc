"""Shared fixtures for mismap-qc tests.

Real-data tests use CPTAC LUAD proteomics files in examples/data/.
Those files are gitignored; tests are skipped automatically when absent.
"""

from pathlib import Path

import pandas as pd
import pytest

CPTAC_TUMOR = (
    Path(__file__).parent.parent
    / "examples/data/HS_CPTAC_LUAD_proteome_ratio_NArm_TUMOR.cct"
)
CPTAC_NORMAL = (
    Path(__file__).parent.parent
    / "examples/data/HS_CPTAC_LUAD_proteome_ratio_NArm_NORMAL.cct"
)


@pytest.fixture(scope="session")
def cptac_df():
    """Load CPTAC LUAD combined proteomics matrix. Skip if data not present."""
    if not CPTAC_TUMOR.exists() or not CPTAC_NORMAL.exists():
        return None
    tumor = pd.read_csv(CPTAC_TUMOR, sep="\t", index_col=0)
    normal = pd.read_csv(CPTAC_NORMAL, sep="\t", index_col=0)
    tumor.columns = [f"{c}_T" for c in tumor.columns]
    normal.columns = [f"{c}_N" for c in normal.columns]
    return pd.concat([tumor, normal], axis=1)


@pytest.fixture(scope="session")
def cptac_groups(cptac_df):
    if cptac_df is None:
        return None
    return ["Tumor" if c.endswith("_T") else "Normal" for c in cptac_df.columns]
