# Shared utilities

Custom Python implementations and helper scripts used across both adhtn-prs and adt2d-prs. **Identical copies live in the adt2d-prs repo** to keep each repo self-contained.

## Scripts

| File | Purpose |
|---|---|
| `ldsc_rg_fast.py` | Custom Python 3 implementation of LDSC bivariate regression. Replaces the Python 2 LDSC dependency for genetic correlation. ~2 min per pair |
| `fast_gene_burden.py` | Vectorized numpy gene-level PRS burden computation across ~23 K genes. ~14 seconds wall time |
| `twas_cross_disease.py` | Combine per-tissue TWAS outputs (S-PrediXcan CSVs) into a long-format gene × tissue × disease table for downstream comparison |
| `twas_fix_varid.py` | Fix variant-ID format mismatches between GWAS and GTEx prediction models (rs## vs chr_pos_ref_alt_b38) |
| `pathway_enrichr_quadrants.py` | Pathway enrichment via Enrichr API, run separately for each cross-disease quadrant (AD↑SBP↑, AD↑SBP↓, AD↓SBP↑, AD↓SBP↓) |
| `quartile_enrichment.py` | PRS quartile enrichment statistics for case-control validation |

## Why duplicated rather than shared

These started as project-specific scripts and were generalized as both projects matured. Treating them as part of each repo (rather than as a separate utility library) keeps each repo self-contained and reproducible without external dependencies.
