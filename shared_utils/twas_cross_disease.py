#!/usr/bin/env python3
"""
Cross-disease TWAS integration: AD × SBP × 13 GTEx brain tissues.

For each tissue:
  - Load AD  S-PrediXcan output (ad_spredixcan_Brain_<T>.csv)
  - Load SBP S-PrediXcan output (spredixcan_sbp_Brain_<T>.csv)
  - Inner-join on gene
  - Classify each gene into one of 4 quadrants by sign of AD.zscore and SBP.zscore
  - Count tissue-specific concordant/discordant hits at several |z| thresholds

Outputs:
  results_htn/phase3_multiomics/twas_cross_disease_all_tissues.tsv
    long-format: one row per (tissue, gene) with AD and SBP z-scores, p-values, quadrant
  results_htn/phase3_multiomics/twas_cross_disease_summary.tsv
    per-tissue summary: n_genes, n_concordant_at_|z|>=4, n_discordant_at_|z|>=4, etc.
  results_htn/phase3_multiomics/twas_top_cross_hits.tsv
    genes with |z|>=4 in both traits in at least 1 tissue (candidates for deep dive)

Runtime ~5 s.
"""
from __future__ import annotations
from pathlib import Path
import polars as pl
import numpy as np

HOME = Path.home()
BASE = HOME / "prs_otherdata/pipeline"
AD_DIR  = BASE / "results"       / "phase3_multiomics"   # ad_spredixcan_*.csv
SBP_DIR = BASE / "results_htn"   / "phase3_multiomics"   # spredixcan_sbp_*.csv
OUT_DIR = SBP_DIR

TISSUES = [
    "Brain_Amygdala",
    "Brain_Anterior_cingulate_cortex_BA24",
    "Brain_Caudate_basal_ganglia",
    "Brain_Cerebellar_Hemisphere",
    "Brain_Cerebellum",
    "Brain_Cortex",
    "Brain_Frontal_Cortex_BA9",
    "Brain_Hippocampus",
    "Brain_Hypothalamus",
    "Brain_Nucleus_accumbens_basal_ganglia",
    "Brain_Putamen_basal_ganglia",
    "Brain_Spinal_cord_cervical_c-1",
    "Brain_Substantia_nigra",
]

Z_THRESHOLDS = [2.0, 3.0, 4.0, 5.0]   # |z| cutoffs for enrichment tests


def load_twas(path: Path, trait: str, tissue: str) -> pl.DataFrame | None:
    if not path.exists():
        print(f"  [miss] {path.name}")
        return None
    df = pl.read_csv(str(path), ignore_errors=True, null_values=["NA","nan",""])
    # S-PrediXcan output standard columns: gene, gene_name, zscore, pvalue, effect_size, ...
    needed = ["gene", "gene_name", "zscore", "pvalue"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"  [bad cols] {path.name} — missing {missing}; has {df.columns}")
        return None
    df = df.select(needed).with_columns(
        pl.col("zscore").cast(pl.Float64, strict=False),
        pl.col("pvalue").cast(pl.Float64, strict=False),
    ).filter(pl.col("zscore").is_not_null() & pl.col("pvalue").is_not_null())
    df = df.rename({
        "zscore":  f"{trait}_z",
        "pvalue":  f"{trait}_p",
    })
    df = df.with_columns(pl.lit(tissue).alias("tissue"))
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_tissues_long = []
    per_tissue_summary = []

    print("=" * 88)
    print("  Cross-disease TWAS integration: AD × SBP × 13 brain tissues")
    print("=" * 88)

    for T in TISSUES:
        ad_path  = AD_DIR  / f"ad_spredixcan_{T}.csv"
        sbp_path = SBP_DIR / f"spredixcan_sbp_{T}.csv"
        ad  = load_twas(ad_path,  "AD",  T)
        sbp = load_twas(sbp_path, "SBP", T)
        if ad is None or sbp is None:
            per_tissue_summary.append({"tissue": T, "n_ad": 0, "n_sbp": 0, "n_shared": 0})
            continue

        # Inner-join on gene (ENSG ID)
        merged = ad.select(["gene", "gene_name", "AD_z", "AD_p"]).join(
            sbp.select(["gene", "SBP_z", "SBP_p"]), on="gene", how="inner"
        ).with_columns(pl.lit(T).alias("tissue"))
        # Quadrant label (by sign of z-score)
        merged = merged.with_columns([
            pl.when((pl.col("AD_z") > 0) & (pl.col("SBP_z") > 0)).then(pl.lit("AD+SBP+"))
              .when((pl.col("AD_z") < 0) & (pl.col("SBP_z") < 0)).then(pl.lit("AD-SBP-"))
              .when((pl.col("AD_z") > 0) & (pl.col("SBP_z") < 0)).then(pl.lit("AD+SBP-"))
              .when((pl.col("AD_z") < 0) & (pl.col("SBP_z") > 0)).then(pl.lit("AD-SBP+"))
              .otherwise(pl.lit("zero"))
              .alias("quadrant"),
        ])
        all_tissues_long.append(merged)

        # Per-tissue summary with multiple |z| cutoffs
        row = {
            "tissue": T,
            "n_ad":   len(ad),
            "n_sbp":  len(sbp),
            "n_shared": len(merged),
        }
        for zc in Z_THRESHOLDS:
            sig_both = merged.filter((pl.col("AD_z").abs() >= zc) & (pl.col("SBP_z").abs() >= zc))
            ns = len(sig_both)
            if ns > 0:
                nc = sig_both.filter(
                    (pl.col("AD_z") * pl.col("SBP_z")) > 0
                ).height
            else:
                nc = 0
            row[f"n_both_|z|>={zc}"]   = ns
            row[f"concordant_|z|>={zc}"] = nc
            row[f"discordant_|z|>={zc}"] = ns - nc
            row[f"frac_conc_|z|>={zc}"]  = round(nc / ns, 3) if ns else None
        per_tissue_summary.append(row)
        print(f"  {T:<42} AD={len(ad):>6,}  SBP={len(sbp):>6,}  shared={len(merged):>6,}  "
              f"|z|>=4 both={row['n_both_|z|>=4.0']:>4}  "
              f"conc={row['concordant_|z|>=4.0']:>3}  disc={row['discordant_|z|>=4.0']:>3}")

    # ── write long-format per-(tissue, gene) table ───────────────────────────
    if all_tissues_long:
        long_df = pl.concat(all_tissues_long)
        out_long = OUT_DIR / "twas_cross_disease_all_tissues.tsv"
        long_df.write_csv(out_long, separator="\t")
        print(f"\nlong table saved: {out_long}  ({len(long_df):,} tissue×gene rows)")
    else:
        long_df = None

    # ── write per-tissue summary ──────────────────────────────────────────────
    sum_df = pl.DataFrame(per_tissue_summary)
    out_sum = OUT_DIR / "twas_cross_disease_summary.tsv"
    sum_df.write_csv(out_sum, separator="\t")
    print(f"summary saved   : {out_sum}")

    # ── top cross-disease hits: |z|>=4 in both, in at least one tissue ────────
    if long_df is not None:
        strong = long_df.filter((pl.col("AD_z").abs() >= 4) & (pl.col("SBP_z").abs() >= 4))
        # Collapse to unique gene with tissue list, min p, etc.
        if len(strong):
            agg = strong.group_by("gene_name").agg([
                pl.col("tissue").n_unique().alias("n_tissues"),
                pl.col("tissue").unique().alias("tissues"),
                pl.col("AD_z").mean().alias("AD_z_mean"),
                pl.col("SBP_z").mean().alias("SBP_z_mean"),
                pl.col("AD_p").min().alias("AD_p_min"),
                pl.col("SBP_p").min().alias("SBP_p_min"),
                pl.col("quadrant").mode().first().alias("modal_quadrant"),
            ]).sort("n_tissues", descending=True)
            out_top = OUT_DIR / "twas_top_cross_hits.tsv"
            agg.write_csv(out_top, separator="\t")
            print(f"top hits saved  : {out_top}  ({len(agg):,} unique genes with |z|>=4 in both traits in >=1 tissue)")

            # Print top 20 by tissue count
            print("\nTop cross-disease TWAS hits (|z|>=4 in both AD and SBP), ranked by n_tissues:")
            print(f"{'gene':<15} {'n_tiss':>6} {'AD_z_mean':>10} {'SBP_z_mean':>10} {'modal_quad':>12}")
            print("-" * 60)
            for row in agg.head(20).iter_rows(named=True):
                print(f"{row['gene_name']:<15} {row['n_tissues']:>6} "
                      f"{row['AD_z_mean']:>+10.2f} {row['SBP_z_mean']:>+10.2f} "
                      f"{row['modal_quadrant']:>12}")

    # ── overall concordance across all tissue-gene pairs ──────────────────────
    if long_df is not None:
        print("\n" + "=" * 88)
        print("Global concordance across all (tissue, gene) rows:")
        print("=" * 88)
        for zc in Z_THRESHOLDS:
            sig = long_df.filter((pl.col("AD_z").abs() >= zc) & (pl.col("SBP_z").abs() >= zc))
            n = len(sig)
            if n:
                c = sig.filter(pl.col("AD_z") * pl.col("SBP_z") > 0).height
                print(f"  |z|>={zc}: n={n:>6,}  concordant={c:>5,} ({100*c/n:.1f}%)  "
                      f"discordant={n-c:>5,} ({100*(n-c)/n:.1f}%)")
            else:
                print(f"  |z|>={zc}: n=0")


if __name__ == "__main__":
    main()
