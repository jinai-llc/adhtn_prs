# Figures — manuscript composites

Three final-stage plotting scripts for the AD × SBP manuscript.

## Scripts

### `make_fig1_AD_SBP.py` (29 KB)

Primary composite figure: PRS architecture + bidirectional MR. Adapted from the AD × T2D `make_merged_fig_prs_mr` template, with T2D → SBP throughout.

- **Row 1a**: SNP weight distribution histogram + per-chromosome PRS burden
- **Row 1b**: Venn (|β| > 0.001) + concordance donut
- **Row 1c**: Miami plot (AD top / SBP bottom inverted, chromosome stripes)
- **Row 2**: Bidirectional MR (forward / reverse / APOE-excluded / forest plot)

### `make_fig2_AD_SBP.py` (92 KB)

Cross-disease TWAS / pathway / cell-type composite. 13 brain tissue panels with top discordant/concordant genes labeled, pathway enrichment per quadrant, and HPA single-nucleus cell-type overlay.

### `make_fig2_AD_SBP_TWAS.py` (15 KB)

Compact TWAS-only alternate of Fig 2. Same scatter grid + heatmap strip but without the pathway and cell-type panels. Useful for presentations where the larger figure is too dense.

## Run commands

```bash
python3 make_fig1_AD_SBP.py \
    --prs-dir   /path/to/results \
    --mr-dir    /path/to/results_htn/phase6_mr \
    --out       ./paper_figs/adhtn

python3 make_fig2_AD_SBP.py \
    --base-dir  /path/to/results \
    --out       ./paper_figs/adhtn

python3 make_fig2_AD_SBP_TWAS.py \
    --twas-long /path/to/results_htn/phase3_multiomics/twas_cross_disease_all_tissues.tsv \
    --out       ./paper_figs/adhtn
```

## Style conventions

- AD = `#7B1FA2` (purple)
- SBP = `#1565C0` (blue, distinct from concordant green)
- Concordant = `#2E7D32` (green)
- Discordant = `#C62828` (red)
- Chromosome stripes = `#f2f2f2` (light grey)
- 14 pt Arial throughout
