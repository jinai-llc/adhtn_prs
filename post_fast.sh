#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  post_fast.sh — LDSC rg + S-PrediXcan TWAS for AD × SBP
#
#  Assumes:
#    - AD harmonized hg38 file exists in results_htn/phase1_gwas/
#    - SBP harmonized hg38 file exists in results_htn/phase1_gwas/
#    - (both produced by `htn_ad_pipeline.py --step harmonize/liftover`)
#
#  Produces:
#    - results_htn/phase4_classification/ldsc_rg_AD_SBP.log     (the rg number + SE + p)
#    - results_htn/phase3_multiomics/spredixcan_sbp_Brain_*.csv (13 tissues)
#
#  Usage:
#    bash post_fast.sh ldsc         # just LDSC rg (~3 min)
#    bash post_fast.sh twas         # just TWAS (~15 min on 4-way parallel)
#    bash post_fast.sh all          # both (runs TWAS in background, LDSC in foreground)
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

BASE="$HOME/prs_otherdata"
PIPELINE="$BASE/pipeline"
PHASE1="$PIPELINE/results_htn/phase1_gwas"
PHASE3="$PIPELINE/results_htn/phase3_multiomics"
PHASE4="$PIPELINE/results_htn/phase4_classification"

# LDSC tool + references
LDSC_DIR="$BASE/omics_resources/mr_tools/ldsc3"
LDSC_PY="$LDSC_DIR/ldsc.py"
MUNGE_PY="$LDSC_DIR/munge_sumstats.py"
EUR_LD="$BASE/omics_resources/annotations/ldsc_annotations/eur_w_ld_chr/"
HM3_SNPLIST="$BASE/omics_resources/ld_reference/w_hm3.snplist"

# S-PrediXcan
SPX_PY="$BASE/omics_resources/twas/tools/MetaXcan/MetaXcan-master/software/SPrediXcan.py"
MASHR_DIR="$BASE/omics_resources/twas/predixcan_models/eqtl/mashr"

AD_HG38="$PHASE1/ad_gwas_harmonized.tsv.gz"
SBP_HG38="$PHASE1/sbp_gwas_harmonized_hg38.tsv.gz"
AD_N=487511
SBP_N=1028980

mkdir -p "$PHASE3" "$PHASE4"

# ── LDSC rg ──────────────────────────────────────────────────────────────────
run_ldsc () {
    echo "══════════════════════════════════════════════════════════════════"
    echo "  LDSC rg(AD, SBP)"
    echo "══════════════════════════════════════════════════════════════════"

    cd "$PHASE4"

    # Decompress for munge (ldsc3 reads plain text; keep .gz untouched upstream)
    local AD_TMP="$PHASE4/ad_for_munge.tsv"
    local SBP_TMP="$PHASE4/sbp_for_munge.tsv"
    [[ -f "$AD_TMP"  ]] || zcat "$AD_HG38"  > "$AD_TMP"
    [[ -f "$SBP_TMP" ]] || zcat "$SBP_HG38" > "$SBP_TMP"

    # Munge: LDSC expects columns SNP A1 A2 and some effect measure
    echo "── munge AD ──"
    python3 "$MUNGE_PY" \
        --sumstats "$AD_TMP" \
        --N "$AD_N" \
        --out "$PHASE4/ad_munged" \
        --merge-alleles "$HM3_SNPLIST" \
        --snp SNP --a1 A1 --a2 A2 --signed-sumstats BETA,0 --p P \
        --chunksize 500000

    echo "── munge SBP ──"
    python3 "$MUNGE_PY" \
        --sumstats "$SBP_TMP" \
        --N "$SBP_N" \
        --out "$PHASE4/sbp_munged" \
        --merge-alleles "$HM3_SNPLIST" \
        --snp SNP --a1 A1 --a2 A2 --signed-sumstats BETA,0 --p P \
        --chunksize 500000

    # Genetic correlation
    echo "── rg(AD, SBP) ──"
    python3 "$LDSC_PY" \
        --rg "$PHASE4/ad_munged.sumstats.gz,$PHASE4/sbp_munged.sumstats.gz" \
        --ref-ld-chr "$EUR_LD" \
        --w-ld-chr   "$EUR_LD" \
        --out        "$PHASE4/ldsc_rg_AD_SBP"

    # Tail of the log carries the rg table
    echo ""
    echo "──── rg result ────"
    grep -A4 "Genetic Correlation" "$PHASE4/ldsc_rg_AD_SBP.log" || true
    grep -A2 "Summary of Genetic"   "$PHASE4/ldsc_rg_AD_SBP.log" || true
    echo "── full log: $PHASE4/ldsc_rg_AD_SBP.log"
}

# ── S-PrediXcan across 13 brain tissues ──────────────────────────────────────
run_twas () {
    echo "══════════════════════════════════════════════════════════════════"
    echo "  S-PrediXcan TWAS for SBP across 13 brain tissues"
    echo "══════════════════════════════════════════════════════════════════"

    local TISSUES=(
        Brain_Amygdala
        Brain_Anterior_cingulate_cortex_BA24
        Brain_Caudate_basal_ganglia
        Brain_Cerebellar_Hemisphere
        Brain_Cerebellum
        Brain_Cortex
        Brain_Frontal_Cortex_BA9
        Brain_Hippocampus
        Brain_Hypothalamus
        Brain_Nucleus_accumbens_basal_ganglia
        Brain_Putamen_basal_ganglia
        Brain_Spinal_cord_cervical_c-1
        Brain_Substantia_nigra
    )

    # Rewrite the 13 launchers so they point at the REAL S-PrediXcan + MASHR paths
    for T in "${TISSUES[@]}"; do
        local OUT="$PHASE3/spredixcan_sbp_${T}.csv"
        cat > "$PHASE3/run_spredixcan_sbp_${T}.sh" <<EOF
#!/bin/bash
set -euo pipefail
python3 "$SPX_PY" \\
  --model_db_path "$MASHR_DIR/mashr_${T}.db" \\
  --covariance     "$MASHR_DIR/mashr_${T}.txt.gz" \\
  --gwas_file      "$SBP_HG38" \\
  --snp_column     SNP --chromosome_column CHR --position_column BP \\
  --effect_allele_column A1 --non_effect_allele_column A2 \\
  --beta_column BETA --se_column SE --pvalue_column P \\
  --keep_non_rsid --additional_output --throw \\
  --output_file    "$OUT"
EOF
        chmod +x "$PHASE3/run_spredixcan_sbp_${T}.sh"
    done

    # Launch 4 at a time
    ls "$PHASE3"/run_spredixcan_sbp_*.sh | xargs -n1 -P4 bash
    echo "── done. tissue output counts:"
    for T in "${TISSUES[@]}"; do
        local f="$PHASE3/spredixcan_sbp_${T}.csv"
        if [[ -f "$f" ]]; then
            echo "  $T : $(( $(wc -l < "$f") - 1 )) genes"
        else
            echo "  $T : MISSING"
        fi
    done
}

case "${1:-all}" in
    ldsc) run_ldsc ;;
    twas) run_twas ;;
    all)
        echo "── launching TWAS in background ──"
        (run_twas > "$PHASE3/twas.log" 2>&1 &)
        echo "── running LDSC in foreground ──"
        run_ldsc
        echo ""
        echo "TWAS running in background. Check with:"
        echo "  tail -20 $PHASE3/twas.log"
        echo "  ls $PHASE3/spredixcan_sbp_*.csv | wc -l   # should reach 13"
        ;;
    *) echo "Usage: $0 {ldsc|twas|all}"; exit 1 ;;
esac
