# Mendelian randomization — AD ↔ SBP

Single Python script implementing bidirectional MR for AD vs SBP, built on the same instrument-selection / harmonization pattern as the adt2d MR code but without the per-method R script split.

## Scripts

| File | Purpose |
|---|---|
| `bidir_mr_AD_SBP.py` | Bidirectional MR for AD ↔ SBP. Forward (SBP → AD), reverse (AD → SBP), and APOE-excluded reverse. IVW, MR-Egger, weighted-median, weighted-mode |

## Inputs

- AD GWAS summary statistics (Bellenguez 2022, harmonized hg38)
- SBP GWAS summary statistics (Keaton 2024, harmonized hg38)
- Instrument selection: p < 5e-8 in exposure GWAS, LD-clumped at r² < 0.001 / 10 Mb

## Outputs

- `forward_harmonized_data.csv`, `forward_mr_sbp_to_ad.csv`
- `reverse_harmonized_data.csv`, `reverse_mr_ad_to_sbp.csv`
- `reverse_harmonized_data_no_apoe.csv`, `reverse_mr_ad_to_sbp_no_apoe.csv`

Consumed by `figures/make_fig1_AD_SBP.py` (Row 2 of the figure, when included).

## Note on midlife vs late-life BP

The literature is genuinely mixed for AD/hypertension: midlife hypertension increases AD risk while late-life low BP also associates with AD (likely reverse causation through cerebral hypoperfusion). MR cannot distinguish these without age-stratified GWAS, so we report the genome-wide MR estimate as a baseline and discuss the temporal nuance qualitatively in the manuscript.
