# adhtn-prs

Cross-disease polygenic risk score (PRS) and multi-omics analysis of **Alzheimer's disease (AD) × Hypertension / Systolic Blood Pressure (SBP)**. Parallel companion to `adt2d-prs`; same scientific framework, different secondary trait.

## Overview

Multi-phase pipeline that consumes summary-statistic GWAS for AD and BP (SBP / DBP / PP) and produces gene-level cross-disease evidence integrating PRS, TWAS, MR, and pathway analysis.

| Phase | What it does |
|---|---|
| 1 — GWAS | QC and harmonization, LDSC munging, h², genetic correlation, locus identification |
| 2 — PRS | Pre-computed SBayesRC weights (Keaton 2024 PGS Catalog), liftOver hg19 → hg38 |
| 3 — Multi-omics | S-PrediXcan TWAS across 13 GTEx brain tissues (AD + SBP separately) |
| 4 — Classification | Locus classification, pathway enrichment per quadrant |
| 5 — MR | AD ↔ SBP bidirectional Mendelian randomization with APOE-region exclusion |
| Figures | Composite manuscript figures (PRS architecture, TWAS multi-tissue) |

## Why simpler than adt2d

This study uses ready-made, author-tuned SBayesRC weights from PGS Catalog rather than running PRS-CS-auto ourselves. SBayesRC on 7.36 M variants is the current state-of-the-art Bayesian method (vs. PRS-CS-auto on ~1.1 M HM3 variants), and it was trained and tuned by Keaton et al. on the full n ≈ 1 M meta-analysis. This skips the PRS-CS computation step entirely, plus the painful liftOver-based rsID recovery that dominated the AD/T2D effort, since PGS Catalog scoring files come pre-formatted with rsID + chr:pos + effect allele + beta.

## Data sources

| Resource | Reference / URL |
|---|---|
| AD GWAS — Bellenguez 2022 | GWAS Catalog GCST90027158 |
| BP GWAS — Keaton 2024 | n ≈ 1,028,980 EUR meta-analysis (CHARGE / ICBP / MVP / Lifelines) |
| SBP PRS weights | PGS Catalog **PGS004603** (SBayesRC, hg19, ~7.36 M) |
| DBP PRS weights | PGS Catalog PGS004604 |
| PP PRS weights | PGS Catalog PGS004605 |
| GTEx v8 prediction models (TWAS) | https://predictdb.org |
| Reference build | GRCh38 (hg38). Keaton SBayesRC weights lifted from hg19 |

## Software requirements

### Python (≥ 3.10)

```bash
pip install numpy pandas polars scipy matplotlib openpyxl h5py
```

### External tools

| Tool | Used in | Source |
|---|---|---|
| LDSC | phase1 (also custom Python in `shared_utils/ldsc_rg_fast.py`) | https://github.com/bulik/ldsc |
| MetaXcan / S-PrediXcan | phase3 | https://github.com/hakyimlab/MetaXcan |
| UCSC liftOver | phase2 | https://genome.ucsc.edu/cgi-bin/hgLiftOver |
| PLINK 1.9 | (downstream Wake Forest scoring) | https://www.cog-genomics.org/plink/ |

## Repository layout

```
adhtn_prs/
├── htn_ad_pipeline.py               # Main HTN pipeline
├── run_htn_ad.sh                    # Orchestration: pull → harmonize → liftOver
├── post_fast.sh                     # Post-processing (gene burden, LDSC, pathway)
├── phase3_multiomics/               # 26 S-PrediXcan launchers (13 tissues × AD + SBP)
├── mendelian_randomization/         # AD ↔ SBP bidirectional MR (Python)
├── shared_utils/                    # Custom LDSC, gene burden, TWAS helpers (also in adt2d_prs)
└── figures/                         # Three composite manuscript figures
```

## Usage

### Full pipeline

```bash
bash run_htn_ad.sh pull       # rsync Keaton GWAS files
bash run_htn_ad.sh full       # harmonize → liftOver → PRS scoring setup
bash post_fast.sh             # gene burden + LDSC + pathway analysis
```

### S-PrediXcan TWAS (per tissue)

```bash
# AD across 13 brain tissues
for f in phase3_multiomics/run_spredixcan_ad_Brain_*.sh; do bash "$f"; done

# SBP across 13 brain tissues
for f in phase3_multiomics/run_spredixcan_sbp_Brain_*.sh; do bash "$f"; done
```

### Bidirectional MR

```bash
python3 mendelian_randomization/bidir_mr_AD_SBP.py
```

### Figure regeneration

```bash
python3 figures/make_fig1_AD_SBP.py \
    --prs-dir   /path/to/results \
    --mr-dir    /path/to/results_htn/phase6_mr \
    --out       ./paper_figs/adhtn

python3 figures/make_fig2_AD_SBP.py \
    --base-dir  /path/to/results \
    --out       ./paper_figs/adhtn

python3 figures/make_fig2_AD_SBP_TWAS.py \
    --twas-long /path/to/results_htn/phase3_multiomics/twas_cross_disease_all_tissues.tsv \
    --out       ./paper_figs/adhtn
```

## Contact

For questions about the code or manuscript (in preparation):

- Heng Du <Heng.Du@wfusm.edu>
- Guangxu Jin <Guangxu.Jin@wfusm.edu>

## License

To be determined before public release.
