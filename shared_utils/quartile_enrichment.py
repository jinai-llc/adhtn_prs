#!/usr/bin/env python3
"""
Quadrant enrichment test: does the LDSC rg = -0.124 (negative, highly significant)
manifest at the gene-burden level when we restrict to high-burden genes?

Logic:
  - Genome-wide: 50.2% concordant, consistent with near-zero rg at per-gene level
  - Prediction: if rg is truly -0.12, then at *high-burden extremes* the negative
    correlation should become visible — i.e. genes with large AD burden AND large
    SBP burden should show discordance more often than chance

Tests performed:
  1. Raw concordance in all genes (sanity check: 50.2%)
  2. Concordance in top-decile AD burden genes  (how does AD-heavy set break?)
  3. Concordance in top-decile SBP burden genes  (how does SBP-heavy set break?)
  4. Concordance in the INTERSECTION of top-decile AD AND top-decile SBP
     (the rg-negative prediction: should be < 50% if rg story is real)
  5. Binomial test for each of the above

Run: python3 quartile_enrichment.py
"""
import json
from pathlib import Path
from math import sqrt
import polars as pl
from scipy.stats import binomtest

HOME = Path.home()
BURDEN = HOME / "prs_otherdata/pipeline/results_htn/phase2_prs/gene_burden_AD_SBP.tsv"
OUT = HOME / "prs_otherdata/pipeline/results_htn/phase2_prs/quadrant_enrichment.json"

def test(name, n_conc, n_total, null=0.5):
    if n_total == 0:
        return {"name": name, "n_total": 0, "n_conc": 0, "frac": None, "p": None}
    frac = n_conc / n_total
    bt = binomtest(n_conc, n_total, p=null, alternative="two-sided")
    return {
        "name": name, "n_total": n_total, "n_conc": n_conc,
        "frac_concordant": round(frac, 4),
        "enrichment_vs_0.5": round((frac - 0.5), 4),
        "p_binom_two_sided": float(bt.pvalue),
    }

def main():
    df = pl.read_csv(str(BURDEN), separator="\t")
    # Keep only genes scored in both (SNPs present in both panels)
    df = df.filter((pl.col("AD_nsnp") > 0) & (pl.col("SBP_nsnp") > 0))
    n_all = len(df)

    # Top decile thresholds on |burden|
    ad_q90  = df["AD_burden"].quantile(0.90)
    sbp_q90 = df["SBP_burden"].quantile(0.90)

    tests = []
    # 1. All genes
    c = df.filter(pl.col("concordant")).height
    tests.append(test("all_genes", c, n_all))

    # 2. Top-decile AD
    top_ad = df.filter(pl.col("AD_burden") >= ad_q90)
    c = top_ad.filter(pl.col("concordant")).height
    tests.append(test("top10pct_AD_burden", c, len(top_ad)))

    # 3. Top-decile SBP
    top_sbp = df.filter(pl.col("SBP_burden") >= sbp_q90)
    c = top_sbp.filter(pl.col("concordant")).height
    tests.append(test("top10pct_SBP_burden", c, len(top_sbp)))

    # 4. Intersection: top-decile in BOTH
    top_both = df.filter((pl.col("AD_burden") >= ad_q90) &
                         (pl.col("SBP_burden") >= sbp_q90))
    c = top_both.filter(pl.col("concordant")).height
    tests.append(test("top10pct_AD_AND_SBP", c, len(top_both)))

    # 5. Stricter: top-5% in both
    ad_q95  = df["AD_burden"].quantile(0.95)
    sbp_q95 = df["SBP_burden"].quantile(0.95)
    top_both_5 = df.filter((pl.col("AD_burden") >= ad_q95) &
                           (pl.col("SBP_burden") >= sbp_q95))
    c = top_both_5.filter(pl.col("concordant")).height
    tests.append(test("top5pct_AD_AND_SBP", c, len(top_both_5)))

    # 6. Joint-z top decile (z_AD + z_SBP)
    df_jz = df.with_columns((pl.col("AD_z") + pl.col("SBP_z")).alias("joint_z"))
    jz_q90 = df_jz["joint_z"].quantile(0.90)
    top_jz = df_jz.filter(pl.col("joint_z") >= jz_q90)
    c = top_jz.filter(pl.col("concordant")).height
    tests.append(test("top10pct_joint_z", c, len(top_jz)))

    # Print in a table
    print("=" * 80)
    print(f"  Quadrant enrichment for AD × SBP gene burden")
    print(f"  LDSC rg = -0.124 (p = 3.7e-32) predicts discordant enrichment at extremes")
    print("=" * 80)
    print(f"{'subset':<28} {'n':>7} {'conc':>6} {'frac':>7} {'Δ0.5':>7} {'p':>12}")
    print("-" * 80)
    for t in tests:
        p_str = f"{t['p_binom_two_sided']:.2e}" if t['p_binom_two_sided'] else "-"
        frac  = f"{t['frac_concordant']:.3f}" if t['frac_concordant'] else "-"
        delta = f"{t['enrichment_vs_0.5']:+.3f}" if t['enrichment_vs_0.5'] is not None else "-"
        print(f"{t['name']:<28} {t['n_total']:>7,} {t['n_conc']:>6,} {frac:>7} {delta:>7} {p_str:>12}")
    print()
    print("Interpretation:")
    print("  - 'Δ0.5' is the deviation of concordance rate from 50% null.")
    print("  - Negative Δ = discordance-enriched (consistent with rg = -0.124).")
    print("  - Positive Δ = concordance-enriched (would contradict the rg sign).")
    print()

    # Save for the report
    summary = {
        "rg_AD_SBP": -0.1237,
        "rg_p": 3.71e-32,
        "genome_wide_concordance_frac": tests[0]["frac_concordant"],
        "tests": tests,
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"saved: {OUT}")

if __name__ == "__main__":
    main()
