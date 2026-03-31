"""Backwards compatibility shim. Use mismap_qc instead."""
from mismap_qc import (  # noqa: F401
    __version__,
    missing_matrix,
    missing_matrix_html,
    rna_missing_matrix,
)
