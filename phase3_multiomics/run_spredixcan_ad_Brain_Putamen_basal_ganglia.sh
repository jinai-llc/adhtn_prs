#!/bin/bash
set -euo pipefail
python3 "/home/guangxujin/prs_otherdata/omics_resources/twas/tools/MetaXcan/MetaXcan-master/software/SPrediXcan.py" \
  --model_db_path "/home/guangxujin/prs_otherdata/omics_resources/twas/predixcan_models/eqtl/mashr/mashr_Brain_Putamen_basal_ganglia.db" \
  --covariance     "/home/guangxujin/prs_otherdata/omics_resources/twas/predixcan_models/eqtl/mashr/mashr_Brain_Putamen_basal_ganglia.txt.gz" \
  --gwas_file      "/home/guangxujin/prs_otherdata/pipeline/results_htn/phase1_gwas/ad_gwas_harmonized_varid.tsv.gz" \
  --snp_column     varID \
  --effect_allele_column A1 --non_effect_allele_column A2 \
  --beta_column BETA --se_column SE --pvalue_column P \
  --keep_non_rsid --additional_output --throw \
  --output_file    "/home/guangxujin/prs_otherdata/pipeline/results_htn/phase3_multiomics/spredixcan_ad_Brain_Putamen_basal_ganglia.csv"
