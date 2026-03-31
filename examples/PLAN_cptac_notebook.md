# Plan: CPTAC Proteomics Tutorial Notebook

## Goal

A fully functional Jupyter notebook demonstrating mismap-qc on real public proteomics data.
Target audience: computational biologists / proteomics people who currently use Perseus, DEP, or protti.
The notebook should show something they cannot easily do with existing tools.

---

## Dataset

**CPTAC LUAD** (Lung Adenocarcinoma) via the `cptac` Python package.

- ~100 samples (tumor + matched normal)
- ~11,000 proteins quantified
- Clinical metadata: tumor/normal status, stage, sex, smoking history
- Well-known in the cancer proteomics community — reviewers will recognise it
- Free, no authentication required
- Download ~200MB on first run, cached locally afterwards

---

## Notebook Structure

### Section 1 — Setup
- pip install instructions (commented out)
- Imports: cptac, pandas, numpy, matplotlib, mismap_qc
- Note on download size and caching

### Section 2 — Load & Prepare Data
- `cptac.download(dataset="luad")` + `cptac.Luad()`
- `get_proteomics()` → samples × proteins DataFrame
- `get_clinical()` → clinical metadata
- Print shape, show first few rows
- Flatten protein column MultiIndex if present
- Align samples between proteomics and clinical tables

### Section 3 — Data Subsetting
- Filter to proteins with 10–90% missingness (the interesting ones)
- Take top 200 by missingness variance (captures structured patterns, not random noise)
- Explain why: all-present and all-absent proteins tell you nothing about QC patterns

### Section 4 — Build MultiIndex
- Transpose to proteins × samples
- Attach clinical metadata as column MultiIndex levels:
  - Level 0: Tumor_Normal (primary biological grouping)
  - Level 1: Stage (I / II / III / IV, simplified)
  - Level 2: Sex
- Handle missing clinical values with 'Unknown' fill

### Section 5 — Plot 1: Overview
- `missing_matrix()` with all three annotation levels
- `group_summary="Tumor_Normal"` to print per-group completeness to console
- `completeness_threshold=0.7` to flag low-quality samples
- Key question: does missingness cluster by biology or by batch?

### Section 6 — Plot 2: Split by Tumor/Normal
- `missing_matrix(df, split_by="Tumor_Normal")`
- Each panel independently clustered
- Shows which proteins are systematically absent in one condition
- This is the biological insight: MNAR (missing not at random) proteins

### Section 7 — Interactive Export
- `missing_matrix_html()` with hover tooltips
- Save to `output/cptac_luad_interactive.html`
- Note: hover shows gene name, sample ID, tumor/normal status, stage, sex, detection status

### Section 8 — Comparison with Existing Tools
- Short markdown cell explaining what Perseus, DEP, FragPipe-Analyst show vs what we show
- Perseus: binary heatmap, no clustering, no annotations
- DEP: no clustering, no annotations
- ComplexHeatmap (R): can do this but requires ~50 lines across 3-4 packages
- mismap-qc: one function call

### Section 9 — Biological Interpretation
- Proteins with high missingness in Normal but not Tumor → tumour-specific expression
- Proteins with high missingness in Tumor → potentially lost in cancer
- Samples that cluster away from their group → potential QC outliers
- This is the decision point before imputation: MNAR vs MAR matters for method choice

---

## Decisions

1. **cptac API** — verify exact method names and clinical column names by installing and inspecting before writing notebook code. Exact API output is saved in the section below.
2. **Protein subset strategy** — top 200 by missingness variance (captures structured patterns better than missingness rate alone).
3. **Stage simplification** — YES, simplify to I/II/III/IV. Granular stages (IA, IB, IIB...) clutter the annotation legend without adding insight for a missingness QC plot. The biological question is disease progression, not substage.
4. **Output location** — save plots to `examples/output/`, gitignore the folder.
5. **HTML export** — inline (self-contained), saved to `examples/output/`.
6. **Fallback dataset** — not needed. Assume cptac remains available.
7. **Requirements file** — YES, add `examples/requirements.txt` with pinned versions.

---

## cptac API Reference

> To be filled in by running the inspection script below before writing the notebook.

Run this first and paste output here:

```python
import cptac
cptac.download(dataset="luad")
luad = cptac.Luad()

prot = luad.get_proteomics()
print("Proteomics shape:", prot.shape)
print("Columns type:", type(prot.columns))
print("First 5 columns:", list(prot.columns[:5]))
print("Index type:", type(prot.index))
print("First 5 samples:", list(prot.index[:5]))

clin = luad.get_clinical()
print("\nClinical shape:", clin.shape)
print("Clinical columns:", list(clin.columns))
print("\nSample_Tumor_Normal values:", clin['Sample_Tumor_Normal'].value_counts() if 'Sample_Tumor_Normal' in clin.columns else "COLUMN NOT FOUND")
print("\nStage column - candidates:", [c for c in clin.columns if 'stage' in c.lower()])
print("Sex column - candidates:", [c for c in clin.columns if 'sex' in c.lower() or 'gender' in c.lower()])
print("Smoking column - candidates:", [c for c in clin.columns if 'smok' in c.lower()])
```

**Paste output here before writing the notebook.**

---

## Files to Create

```
mismap-qc/
└── examples/
    ├── PLAN_cptac_notebook.md        ← this file
    ├── cptac_proteomics.ipynb        ← the notebook
    └── output/                       ← gitignored
        ├── cptac_overview.png
        ├── cptac_split.png
        └── cptac_interactive.html
```

---

## Next Steps for Whoever Picks This Up

1. Run the inspection script in the **cptac API Reference** section above. Paste the output into the plan.
2. Based on the output, confirm exact column names for: tumor/normal status, stage, sex.
3. Write `examples/cptac_proteomics.ipynb` following the structure in **Notebook Structure**.
4. Write `examples/requirements.txt` with: `mismap-qc`, `cptac`, `pandas`, `numpy`, `matplotlib`, `plotly`, `jupyter` — pin versions after confirming the notebook runs.
5. Add `examples/output/` to `.gitignore`.
6. Run the notebook end-to-end and save outputs.
7. Commit everything and update the mismap-qc README to link to the example.

## Status

- [x] Plan written
- [x] cptac API inspected — column names confirmed
- [x] Notebook written
- [x] Notebook runs end-to-end
- [x] requirements.txt written
- [x] README updated with link to example
