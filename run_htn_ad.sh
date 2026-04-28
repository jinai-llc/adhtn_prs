#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  run_htn_ad.sh — orchestrator for the Hypertension × AD PRS pipeline
#
#  Bucket: gs://data-shared/jing/gwas_summary/Hypertension
#
#  Usage:
#     bash run_htn_ad.sh ls             # list what's in the bucket
#     bash run_htn_ad.sh pull           # rsync bucket → ~/gwas_summary/HTN/
#                                       #   (prints each file's header for ID)
#     bash run_htn_ad.sh inspect        # verify rename_map vs. real headers
#     bash run_htn_ad.sh full           # harmonize + liftOver + write PRS-CS launchers
#     nohup bash <launcher>.sh &        # actually run PRS-CS (~4 h / trait)
#     bash run_htn_ad.sh post           # gene burden + LDSC + TWAS after PRS-CS
#     bash run_htn_ad.sh sensitivity    # also run DBP and PP arms
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

BUCKET="gs://data-shared/jing/gwas_summary/Hypertension"
LOCAL_DIR="$HOME/gwas_summary/HTN"
PIPELINE_PY="$HOME/prs_otherdata/pipeline/htn_ad_pipeline.py"

mkdir -p "$LOCAL_DIR"

cmd="${1:-full}"

case "$cmd" in
    ls)
        echo "═══ listing $BUCKET ═══"
        gsutil ls -lh "$BUCKET/" 2>/dev/null || gcloud storage ls -l "$BUCKET/"
        ;;
    pull)
        echo "═══ mirroring $BUCKET → $LOCAL_DIR ═══"
        # Mirror everything; whatever you uploaded (SBP only, all three, or a tarball) comes down.
        gsutil -m rsync -r "$BUCKET/" "$LOCAL_DIR/" 2>/dev/null || \
            gcloud storage rsync -r "$BUCKET/" "$LOCAL_DIR/"
        echo ""
        echo "── Local inventory ──"
        ls -lh "$LOCAL_DIR/"
        echo ""
        echo "── Detected files (first line of each) ──"
        shopt -s nullglob
        for f in "$LOCAL_DIR"/*.gz "$LOCAL_DIR"/*.tsv "$LOCAL_DIR"/*.txt "$LOCAL_DIR"/*.csv; do
            [[ -f "$f" ]] || continue
            size=$(du -h "$f" | cut -f1)
            if [[ "$f" == *.gz ]]; then
                hdr=$(zcat "$f" 2>/dev/null | head -1 | cut -c1-200)
            else
                hdr=$(head -1 "$f" | cut -c1-200)
            fi
            echo "[$size] $(basename "$f")"
            echo "       $hdr"
        done
        shopt -u nullglob
        echo ""
        echo "── Next: edit TRAIT_CONFIG in $PIPELINE_PY so each gwas_file path"
        echo "        points at the right file above, then: bash $0 inspect"
        ;;
    inspect)
        python3 "$PIPELINE_PY" --step inspect --secondary SBP
        ;;
    full)
        bash "$0" pull
        python3 "$PIPELINE_PY" --step full --secondary SBP
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "  SBayesRC weights downloaded (Keaton 2024, PGS004603, ~7M SNPs)."
        echo "  No 4-h PRS-CS wait needed — go straight to post-processing:"
        echo "═══════════════════════════════════════════════════════════════"
        echo "     bash $0 post"
        ;;
    post)
        echo "═══ post-PRS-CS: gene burden, LDSC, TWAS ═══"
        python3 "$PIPELINE_PY" --step gene_burden --secondary SBP
        python3 "$PIPELINE_PY" --step ldsc        --secondary SBP
        python3 "$PIPELINE_PY" --step twas        --secondary SBP
        ;;
    sensitivity)
        echo "═══ DBP and PP sensitivity runs ═══"
        for T in DBP PP; do
            python3 "$PIPELINE_PY" --step full --secondary "$T"
        done
        ;;
    *)
        echo "Usage: $0 {pull|inspect|full|post|sensitivity}"
        exit 1
        ;;
esac
