#!/usr/bin/env python3
"""
Fix the MASHR TWAS SNP-matching problem for both AD and SBP.

Problem: MASHR stores variants as "chr{CHR}_{BP}_{A1}_{A2}_b38" (with "chr"
prefix and "_b38" suffix). Neither our AD file (chr1_594445_C_T) nor our SBP
file (rsIDs) match this format, leading to 1.6 SNPs/gene in AD and 0 in SBP.

Fix: add a 'varID' column in MASHR format to each harmonized file, regenerate
the S-PrediXcan launcher scripts to use --snp_column varID, and re-run.

Allele orientation: S-PrediXcan auto-handles A1↔A2 swaps (flips beta sign
when the varID matches the swapped form). So we just use A1_A2 order
consistently; SNPs where the genome-reference allele order is A2_A1 will
still match via S-PrediXcan's internal flip logic.
"""
from __future__ import annotations
import os, stat, subprocess
from pathlib import Path
import polars as pl

HOME = Path.home()
BASE = HOME / "prs_otherdata/pipeline"
PHASE1_NEW = BASE / "results_htn/phase1_gwas"
PHASE1_OLD = BASE / "results/phase1_gwas"
PHASE3_NEW = BASE / "results_htn/phase3_multiomics"
OMICS = HOME / "prs_otherdata/omics_resources"
MASHR_DIR = OMICS / "twas/predixcan_models/eqtl/mashr"
SPX_PY = OMICS / "twas/tools/MetaXcan/MetaXcan-master/software/SPrediXcan.py"

TISSUES = [
    "Brain_Amygdala", "Brain_Anterior_cingulate_cortex_BA24",
    "Brain_Caudate_basal_ganglia", "Brain_Cerebellar_Hemisphere",
    "Brain_Cerebellum", "Brain_Cortex", "Brain_Frontal_Cortex_BA9",
    "Brain_Hippocampus", "Brain_Hypothalamus",
    "Brain_Nucleus_accumbens_basal_ganglia", "Brain_Putamen_basal_ganglia",
    "Brain_Spinal_cord_cervical_c-1", "Brain_Substantia_nigra",
]

# ── 1. Build varID files ──────────────────────────────────────────────────────
def add_varid(src: Path, dst: Path, label: str):
    if dst.exists() and dst.stat().st_size > 0:
        print(f"  [skip] {label}: {dst.name} already exists")
        return
    print(f"  [+] reading {src.name}")
    df = pl.read_csv(str(src), separator="\t", ignore_errors=True)
    print(f"      raw: {len(df):,} rows, cols: {df.columns[:8]}...")
    # Synthesize varID = chr{CHR}_{BP}_{A1}_{A2}_b38
    df = df.with_columns(
        pl.concat_str([
            pl.lit("chr"),
            pl.col("CHR").cast(pl.Utf8),
            pl.lit("_"),
            pl.col("BP").cast(pl.Utf8),
            pl.lit("_"),
            pl.col("A2"),
            pl.lit("_"),
            pl.col("A1"),
            pl.lit("_b38"),
        ]).alias("varID")
    )
    tmp = dst.with_suffix("")   # strip .gz for polars write
    df.write_csv(str(tmp), separator="\t")
    subprocess.run(["gzip", "-f", str(tmp)], check=True)
    print(f"  [✓] {label}: wrote {dst.name} ({len(df):,} rows, varID added)")


# ── 2. Regenerate launcher scripts ────────────────────────────────────────────
def write_launcher(trait: str, tissue: str, gwas_file: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    launcher = out_dir / f"run_spredixcan_{trait.lower()}_{tissue}.sh"
    output   = out_dir / f"spredixcan_{trait.lower()}_{tissue}.csv"
    with open(launcher, "w") as f:
        f.write(f"""#!/bin/bash
set -euo pipefail
python3 "{SPX_PY}" \\
  --model_db_path "{MASHR_DIR}/mashr_{tissue}.db" \\
  --covariance     "{MASHR_DIR}/mashr_{tissue}.txt.gz" \\
  --gwas_file      "{gwas_file}" \\
  --snp_column     varID \\
  --effect_allele_column A1 --non_effect_allele_column A2 \\
  --beta_column BETA --se_column SE --pvalue_column P \\
  --keep_non_rsid --additional_output --throw \\
  --output_file    "{output}"
""")
    os.chmod(launcher, 0o755)
    return launcher


# ── 3. Driver ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  TWAS fix: add MASHR-format varID, regenerate launchers")
    print("=" * 72)

    # AD harmonized hg38 file — try new tree first, fall back to legacy
    ad_src = None
    for p in [PHASE1_NEW / "ad_gwas_harmonized.tsv.gz",
              PHASE1_OLD / "ad_gwas_harmonized.tsv.gz"]:
        if p.exists():
            ad_src = p; break
    if ad_src is None:
        raise FileNotFoundError("No AD harmonized file found")
    ad_dst = PHASE1_NEW / "ad_gwas_harmonized_varid.tsv.gz"

    sbp_src = PHASE1_NEW / "sbp_gwas_harmonized_hg38.tsv.gz"
    sbp_dst = PHASE1_NEW / "sbp_gwas_harmonized_hg38_varid.tsv.gz"

    print("\n[1/3] Build varID-augmented harmonized files")
    add_varid(ad_src, ad_dst, "AD")
    add_varid(sbp_src, sbp_dst, "SBP")

    print("\n[2/3] Regenerate 26 S-PrediXcan launchers (13 tissues × 2 traits)")
    phase3 = PHASE3_NEW
    for T in TISSUES:
        write_launcher("ad",  T, ad_dst,  phase3)
        write_launcher("sbp", T, sbp_dst, phase3)
    print(f"  wrote 26 launchers in {phase3}")

    print("\n[3/3] Done. To run:")
    print(f"  ls {phase3}/run_spredixcan_*.sh | xargs -n1 -P4 bash &")
    print(f"  echo \"TWAS PID: $!\"")
    print(f"  # ~25 min for all 26 jobs at 4-way parallel")
    print(f"\nAfter running, verify SNP-match quality with:")
    print(f'  for T in Brain_Cortex Brain_Hippocampus; do')
    print(f'    for X in ad sbp; do')
    print(f'      f="{phase3}/spredixcan_${{X}}_${{T}}.csv"')
    print(f'      echo -n "$X $T "')
    print(f'      awk -F, \'NR>1 && $6>0 {{sum+=$6; n++}} END {{printf "avg=%.1f over %d genes\\n", sum/n, n}}\' "$f"')
    print(f'    done')
    print(f'  done')
    print(f"  # avg SNPs/gene should be 10-100, not ~1.6")


if __name__ == "__main__":
    main()
