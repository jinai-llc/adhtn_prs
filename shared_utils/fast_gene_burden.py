#!/usr/bin/env python3
"""
Fast gene-level PRS burden: AD (PRS-CS-auto) vs SBP (SBayesRC).

Algorithm: for each chromosome, sort SNPs by BP and use numpy.searchsorted to
find the [gene_start, gene_end] slice in O(log N) per gene instead of scanning
the full SNP table. ~23K genes × log(~500K SNPs/chr) ≈ half a million ops total.
Expected runtime: ~5 seconds vs ~15 minutes for the naive version.

Outputs:
  results_htn/phase2_prs/gene_burden_AD_SBP.tsv
  results_htn/phase2_prs/gene_burden_AD_SBP_summary.txt

Columns in the TSV:
  gene_name, CHR, start, end,
  AD_nsnp, AD_burden, AD_signed,
  SBP_nsnp, SBP_burden, SBP_signed,
  concordant (True/False based on sign of signed-burden product),
  AD_z, SBP_z  (within-trait z-scores of |burden|, for apples-to-apples plotting)
"""
from __future__ import annotations
import time
import numpy as np
import polars as pl
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
HOME = Path.home()
BASE = HOME / "prs_otherdata"
PHASE2_LEGACY = BASE / "pipeline" / "results" / "phase2_prs"
PHASE2_NEW    = BASE / "pipeline" / "results_htn" / "phase2_prs"
GENE_POS      = BASE / "ADsum_GCST_cortex_Tigar_TWAS.xlsx"
OUT_TSV       = PHASE2_NEW / "gene_burden_AD_SBP.tsv"
OUT_SUM       = PHASE2_NEW / "gene_burden_AD_SBP_summary.txt"

AUTOSOMES = [str(i) for i in range(1, 23)]


# ── helpers ───────────────────────────────────────────────────────────────────
def load_weights(subdir: Path, prefix: str) -> pl.DataFrame:
    """Load 22 per-chromosome PRS-CS-style weight files: CHR SNP BP A1 A2 BETA_PRS."""
    parts = []
    for chr_ in range(1, 23):
        f = subdir / f"{prefix}_pst_eff_a1_b0.5_phiauto_chr{chr_}.txt"
        if not f.exists() or f.stat().st_size == 0:
            continue
        parts.append(pl.read_csv(
            f, separator="\t", has_header=False,
            new_columns=["CHR", "SNP", "BP", "A1", "A2", "BETA_PRS"]
        ))
    if not parts:
        raise RuntimeError(f"no weight files in {subdir}")
    w = pl.concat(parts).with_columns(
        pl.col("CHR").cast(pl.Utf8).str.replace(r"^chr", "").alias("CHR"),
        pl.col("BP").cast(pl.Int64),
        pl.col("BETA_PRS").cast(pl.Float64),
    ).filter(pl.col("CHR").is_in(AUTOSOMES))
    return w


def index_by_chrom(w: pl.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """For each chromosome: (sorted BP array, BETA_PRS array in matching order)."""
    idx: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for chr_ in AUTOSOMES:
        sub = w.filter(pl.col("CHR") == chr_).sort("BP")
        if len(sub) == 0:
            continue
        idx[chr_] = (sub["BP"].to_numpy(), sub["BETA_PRS"].to_numpy())
    return idx


def burden(idx: dict[str, tuple[np.ndarray, np.ndarray]],
           genes: pl.DataFrame, label: str) -> pl.DataFrame:
    """Per-gene sum of |BETA_PRS| and signed sum via binary search."""
    out_n, out_abs, out_signed = [], [], []
    gene_names = genes["gene_name"].to_list()
    chrs       = genes["CHR"].to_list()
    starts     = genes["start"].to_numpy()
    ends       = genes["end"].to_numpy()
    for k in range(len(gene_names)):
        chr_s = str(chrs[k])
        if chr_s not in idx:
            out_n.append(0); out_abs.append(0.0); out_signed.append(0.0); continue
        bp, beta = idx[chr_s]
        i = np.searchsorted(bp, int(starts[k]), side="left")
        j = np.searchsorted(bp, int(ends[k]),   side="right")
        if j > i:
            b = beta[i:j]
            out_n.append(j - i)
            out_abs.append(float(np.abs(b).sum()))
            out_signed.append(float(b.sum()))
        else:
            out_n.append(0); out_abs.append(0.0); out_signed.append(0.0)
    return pl.DataFrame({
        f"{label}_nsnp":   out_n,
        f"{label}_burden": out_abs,
        f"{label}_signed": out_signed,
    })


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 70)
    print("  Fast gene burden:  AD (PRS-CS-auto)  vs  SBP (SBayesRC)")
    print("=" * 70)

    # Weights
    print("\n[1/4] Load PRS weights")
    ad  = load_weights(PHASE2_LEGACY / "ad_prscs",  "ad")
    print(f"  AD : {len(ad):,} SNPs")
    sbp = load_weights(PHASE2_NEW / "sbp_sbayesrc", "sbp")
    print(f"  SBP: {len(sbp):,} SNPs")

    # Per-chromosome index
    print("\n[2/4] Build per-chromosome binary-search index")
    ad_idx  = index_by_chrom(ad)
    sbp_idx = index_by_chrom(sbp)
    print(f"  AD  chromosomes: {sorted(ad_idx)}")
    print(f"  SBP chromosomes: {sorted(sbp_idx)}")

    # Genes
    print("\n[3/4] Load gene positions")
    raw = pl.read_excel(str(GENE_POS))
    rename = {}
    for canon, cands in {
        "gene_name": ["GeneName", "gene_name", "gene", "symbol"],
        "CHR":       ["CHROM", "CHR", "chr", "chromosome"],
        "start":     ["GeneStart", "start", "Start"],
        "end":       ["GeneEnd", "end", "End"],
    }.items():
        for c in cands:
            if c in raw.columns:
                rename[c] = canon
                break
    genes = (raw.rename(rename)
                .select(["gene_name", "CHR", "start", "end"])
                .unique("gene_name")
                .with_columns(
                    pl.col("CHR").cast(pl.Utf8).str.replace(r"^chr", "").alias("CHR"),
                    pl.col("start").cast(pl.Int64),
                    pl.col("end").cast(pl.Int64),
                )
                .filter(
                    pl.col("CHR").is_in(AUTOSOMES) &
                    pl.col("start").is_not_null() & pl.col("end").is_not_null()
                ))
    print(f"  genes: {len(genes):,}  (autosomal, deduped)")

    # Burden
    print("\n[4/4] Compute burden (binary search)")
    t1 = time.time()
    ad_b  = burden(ad_idx,  genes, "AD")
    sbp_b = burden(sbp_idx, genes, "SBP")
    print(f"  done in {time.time()-t1:.1f}s")

    # Merge and enrich
    df = pl.concat([genes, ad_b, sbp_b], how="horizontal")

    # Concordance: based on sign of signed burden
    df = df.with_columns(
        (pl.col("AD_signed") * pl.col("SBP_signed") > 0).alias("concordant"),
    )

    # Within-trait z-scores of |burden| (for apples-to-apples cross-trait plots)
    for trait in ("AD", "SBP"):
        bcol = f"{trait}_burden"
        mean = df[bcol].mean()
        std  = df[bcol].std()
        df = df.with_columns(
            ((pl.col(bcol) - mean) / std).alias(f"{trait}_z")
        )

    # Restrict to genes with at least 1 SNP in BOTH traits (usable for concordance)
    df_both = df.filter((pl.col("AD_nsnp") > 0) & (pl.col("SBP_nsnp") > 0))

    # Save
    df.write_csv(OUT_TSV, separator="\t")
    print(f"\nsaved: {OUT_TSV}")

    # Summary
    n_total     = len(df)
    n_ad        = (df["AD_nsnp"] > 0).sum()
    n_sbp       = (df["SBP_nsnp"] > 0).sum()
    n_both      = len(df_both)
    n_conc      = df_both.filter(pl.col("concordant")).height
    n_disc      = n_both - n_conc
    top_ad      = df.sort("AD_burden",  descending=True).head(10)["gene_name"].to_list()
    top_sbp     = df.sort("SBP_burden", descending=True).head(10)["gene_name"].to_list()
    top_both    = df_both.with_columns(
        (pl.col("AD_z") + pl.col("SBP_z")).alias("joint_z")
    ).sort("joint_z", descending=True).head(10)["gene_name"].to_list()

    summary = f"""Gene-level PRS burden  —  AD (PRS-CS-auto, HM3)  vs  SBP (SBayesRC, 7M)
================================================================================
Runtime           : {time.time()-t0:.1f} s
Genes total       : {n_total:,}
  with any AD SNP : {n_ad:,}
  with any SBP SNP: {n_sbp:,}
  with SNPs in both (usable for concordance): {n_both:,}

Concordance (among genes with SNPs in both traits, sign of signed burden):
  concordant : {n_conc:,}  ({100*n_conc/max(n_both,1):.1f}%)
  discordant : {n_disc:,}  ({100*n_disc/max(n_both,1):.1f}%)

Top 10 AD-burden genes:
  {", ".join(top_ad)}

Top 10 SBP-burden genes:
  {", ".join(top_sbp)}

Top 10 joint-z genes (z_AD + z_SBP, highest combined burden):
  {", ".join(top_both)}
"""
    OUT_SUM.write_text(summary)
    print(summary)
    print(f"summary saved: {OUT_SUM}")


if __name__ == "__main__":
    main()
