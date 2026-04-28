# Phase 3 — Multi-omics S-PrediXcan launchers

26 per-tissue S-PrediXcan launcher scripts: 13 GTEx brain tissues × {AD, SBP}.

## Tissues covered

| Tissue | AD launcher | SBP launcher |
|---|---|---|
| Amygdala | `run_spredixcan_ad_Brain_Amygdala.sh` | `run_spredixcan_sbp_Brain_Amygdala.sh` |
| Anterior cingulate cortex BA24 | `run_spredixcan_ad_Brain_Anterior_cingulate_cortex_BA24.sh` | `run_spredixcan_sbp_...` |
| Caudate basal ganglia | ↓ | ↓ |
| Cerebellar Hemisphere | | |
| Cerebellum | | |
| Cortex | | |
| Frontal Cortex BA9 | | |
| Hippocampus | | |
| Hypothalamus | | |
| Nucleus accumbens basal ganglia | | |
| Putamen basal ganglia | | |
| Spinal cord cervical c-1 | | |
| Substantia nigra | | |

## Run all

```bash
# AD across all 13 brain tissues
for f in run_spredixcan_ad_Brain_*.sh; do
    bash "$f"
done

# SBP across all 13 brain tissues
for f in run_spredixcan_sbp_Brain_*.sh; do
    bash "$f"
done
```

## Method

S-PrediXcan with GTEx v8 mashr brain prediction models. Inputs:
- Harmonized GWAS summary statistics (AD: Bellenguez 2022; SBP: Keaton 2024)
- Prediction models: `eqtl/mashr/mashr_<tissue>.db` and `mashr_<tissue>.txt.gz` covariance files

Outputs: per-tissue z-scores combined downstream by `shared_utils/twas_cross_disease.py` into the long table consumed by figures.
