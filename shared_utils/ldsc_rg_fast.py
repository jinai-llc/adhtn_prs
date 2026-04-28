#!/usr/bin/env python3
"""
Standalone LDSC rg: AD × SBP   (v3)

Build-mismatch fix: AD harmonized is hg38 (chr_pos_ref_alt IDs), SBP harmonized
is hg38 (rsIDs, post-liftOver from hg19). LD scores are hg19-keyed.

Strategy:
  1. Inner-join AD (hg38 CHR+BP) with SBP (hg38 CHR+BP) to transfer rsID from SBP to AD.
  2. Join the merged (AD ⋒ SBP) with LD scores on rsID (rsIDs are build-agnostic).
  3. Filter to HM3 (single-column rsID whitelist), align alleles, regress.

This avoids needing hg38 LD scores or a hg38→hg19 chain file.

Output: results_htn/phase4_classification/ldsc_rg_AD_SBP.json
"""
from __future__ import annotations
import json, time
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import polars as pl

# ── paths ─────────────────────────────────────────────────────────────────────
HOME   = Path.home()
BASE   = HOME / "prs_otherdata"
PHASE1 = BASE / "pipeline" / "results_htn" / "phase1_gwas"
PHASE4 = BASE / "pipeline" / "results_htn" / "phase4_classification"
LD_DIR = BASE / "omics_resources" / "annotations" / "ldsc_annotations" / "eur_w_ld_chr"
HM3    = BASE / "omics_resources" / "ld_reference" / "w_hm3.snplist"

AD_FILE  = PHASE1 / "ad_gwas_harmonized.tsv.gz"            # hg38, SNP = chr_pos_ref_alt
SBP_FILE = PHASE1 / "sbp_gwas_harmonized_hg38.tsv.gz"       # hg38, SNP = rsID

N_AD  = 487_511
N_SBP = 1_028_980
N_BLOCKS = 200

PHASE4.mkdir(parents=True, exist_ok=True)


# ── loaders ───────────────────────────────────────────────────────────────────
def load_ldscores() -> pl.DataFrame:
    parts = []
    for chr_ in range(1, 23):
        d = pl.read_csv(str(LD_DIR / f"{chr_}.l2.ldscore.gz"), separator="\t")
        parts.append(d.select(["SNP", "L2"]))   # rsID + LD score; positions not needed
    ld = pl.concat(parts).unique("SNP").with_columns(
        pl.col("L2").cast(pl.Float64),
    )
    print(f"  LD scores        : {len(ld):,} rsIDs")
    return ld


def load_hm3_whitelist() -> set[str]:
    df = pl.read_csv(str(HM3), has_header=False, new_columns=["SNP"])
    s = set(df["SNP"].to_list())
    print(f"  HM3 whitelist    : {len(s):,} rsIDs")
    return s


def load_ad_hg38() -> pl.DataFrame:
    """Keep CHR, BP (hg38), A1, A2, Z — drop the chr_pos_ref_alt SNP column."""
    print("  reading AD (hg38, chr_pos_ref_alt)")
    df = pl.read_csv(str(AD_FILE), separator="\t", ignore_errors=True)
    df = df.with_columns(
        pl.col("CHR").cast(pl.Int64, strict=False),
        pl.col("BP").cast(pl.Int64, strict=False),
        pl.col("BETA").cast(pl.Float64, strict=False),
        pl.col("SE").cast(pl.Float64, strict=False),
    ).filter(
        pl.col("CHR").is_not_null() & pl.col("BP").is_not_null() &
        pl.col("BETA").is_not_null() & (pl.col("SE") > 0)
    ).with_columns((pl.col("BETA") / pl.col("SE")).alias("Z")) \
     .filter(pl.col("Z").is_finite())
    print(f"  AD raw           : {len(df):,} valid-Z variants")
    return df.select(["CHR", "BP", "A1", "A2", "Z"])


def load_sbp_hg38() -> pl.DataFrame:
    """Keep SNP (rsID), CHR, BP (hg38), A1, A2, Z."""
    print("  reading SBP (hg38, rsID)")
    df = pl.read_csv(str(SBP_FILE), separator="\t", ignore_errors=True)
    df = df.with_columns(
        pl.col("CHR").cast(pl.Int64, strict=False),
        pl.col("BP").cast(pl.Int64, strict=False),
        pl.col("BETA").cast(pl.Float64, strict=False),
        pl.col("SE").cast(pl.Float64, strict=False),
    ).filter(
        pl.col("BETA").is_not_null() & (pl.col("SE") > 0) &
        pl.col("SNP").str.starts_with("rs")
    ).with_columns((pl.col("BETA") / pl.col("SE")).alias("Z")) \
     .filter(pl.col("Z").is_finite())
    print(f"  SBP raw          : {len(df):,} rsID variants")
    return df.select(["SNP", "CHR", "BP", "A1", "A2", "Z"])


# ── LDSC regression ───────────────────────────────────────────────────────────
def ldsc_reg(y, L, slope_scale):
    A = np.column_stack([np.ones_like(L), L])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    intercept, slope = float(coef[0]), float(coef[1])
    return slope * slope_scale, intercept, slope


def rg_with_jackknife(z1, z2, L, N1, N2, n_blocks=200):
    M = len(L)
    chi2_1 = z1 * z1
    chi2_2 = z2 * z2
    cross  = z1 * z2

    h2_1, int1, _ = ldsc_reg(chi2_1, L, M / N1)
    h2_2, int2, _ = ldsc_reg(chi2_2, L, M / N2)
    rho,  intx, _ = ldsc_reg(cross,  L, M / np.sqrt(N1 * N2))
    rg = rho / np.sqrt(h2_1 * h2_2) if (h2_1 > 0 and h2_2 > 0) else np.nan

    rng = np.random.default_rng(42)
    order = rng.permutation(M)
    L_s, c1_s, c2_s, cx_s = L[order], chi2_1[order], chi2_2[order], cross[order]

    block_size = M // n_blocks
    jk_rg, jk_h1, jk_h2, jk_rho = [], [], [], []
    for b in range(n_blocks):
        lo = b * block_size
        hi = lo + block_size if b < n_blocks - 1 else M
        keep = np.ones(M, dtype=bool); keep[lo:hi] = False
        mk = int(keep.sum())
        h1, _, _ = ldsc_reg(c1_s[keep], L_s[keep], mk / N1)
        h2, _, _ = ldsc_reg(c2_s[keep], L_s[keep], mk / N2)
        rh, _, _ = ldsc_reg(cx_s[keep], L_s[keep], mk / np.sqrt(N1 * N2))
        jk_h1.append(h1); jk_h2.append(h2); jk_rho.append(rh)
        jk_rg.append(rh / np.sqrt(h1 * h2) if (h1 > 0 and h2 > 0) else np.nan)

    def jk_se(arr):
        arr = np.array(arr); arr = arr[np.isfinite(arr)]
        if len(arr) < 2: return float("nan")
        m = arr.mean()
        return float(np.sqrt((len(arr) - 1) / len(arr) * np.sum((arr - m) ** 2)))

    rg_se, h1_se, h2_se, rho_se = jk_se(jk_rg), jk_se(jk_h1), jk_se(jk_h2), jk_se(jk_rho)

    if np.isfinite(rg) and np.isfinite(rg_se) and rg_se > 0:
        z_rg = rg / rg_se
        p_rg = erfc(abs(z_rg) / sqrt(2))
    else:
        z_rg = p_rg = float("nan")

    return dict(
        M=int(M),
        rg=float(rg), rg_se=rg_se, rg_z=float(z_rg), rg_p=float(p_rg),
        rho_g=float(rho), rho_g_se=rho_se,
        h2_AD=float(h2_1), h2_AD_se=h1_se, h2_AD_intercept=float(int1),
        h2_SBP=float(h2_2), h2_SBP_se=h2_se, h2_SBP_intercept=float(int2),
        cross_intercept=float(intx),
    )


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 72)
    print("  Standalone LDSC rg : AD × SBP  (v3 — SBP bridges rsIDs to AD)")
    print("=" * 72)

    print("\n[1/5] Load LD scores (rsID-keyed)")
    ld = load_ldscores()

    print("\n[2/5] Load HM3 whitelist")
    hm3 = load_hm3_whitelist()

    print("\n[3/5] Load AD + SBP (both hg38)")
    ad  = load_ad_hg38()
    sbp = load_sbp_hg38()

    print("\n[4/5] Inner-join AD × SBP on (CHR, BP)_hg38 — rsID comes from SBP")
    m = ad.join(sbp, on=["CHR", "BP"], suffix="_sbp")
    print(f"  AD ∩ SBP on CHR+BP: {len(m):,}")
    # Rename the SBP-side columns clearly (after the join SBP columns got "_sbp" suffix,
    # but SNP, Z (from SBP) have no AD counterpart and so come through without suffix —
    # wait, AD has A1, A2, Z. Let me double check by selecting explicitly.)
    # After join: CHR, BP (join keys), A1, A2, Z (AD), SNP, A1_sbp, A2_sbp, Z_sbp (SBP).
    # SNP has no suffix because AD doesn't have an SNP column anymore.

    print("\n[5/5] HM3 filter → LD join → allele alignment")
    m = m.filter(pl.col("SNP").is_in(hm3))
    print(f"  ∩ HM3            : {len(m):,}")
    m = m.join(ld, on="SNP", how="inner").filter(pl.col("L2") > 0)
    print(f"  ∩ LD score       : {len(m):,}")

    m = m.with_columns([
        ((pl.col("A1") == pl.col("A1_sbp")) & (pl.col("A2") == pl.col("A2_sbp"))).alias("match"),
        ((pl.col("A1") == pl.col("A2_sbp")) & (pl.col("A2") == pl.col("A1_sbp"))).alias("flip"),
    ]).filter(pl.col("match") | pl.col("flip"))
    m = m.with_columns(
        pl.when(pl.col("flip")).then(-pl.col("Z_sbp")).otherwise(pl.col("Z_sbp")).alias("Z_sbp_a")
    )
    print(f"  after allele align: {len(m):,}")

    z1 = m["Z"].to_numpy()
    z2 = m["Z_sbp_a"].to_numpy()
    L  = m["L2"].to_numpy()

    print("\nRunning LDSC regression + block jackknife ...")
    res = rg_with_jackknife(z1, z2, L, N_AD, N_SBP, n_blocks=N_BLOCKS)
    res["N_AD"] = N_AD
    res["N_SBP"] = N_SBP
    res["runtime_s"] = round(time.time() - t0, 1)

    out = PHASE4 / "ldsc_rg_AD_SBP.json"
    out.write_text(json.dumps(res, indent=2))

    print("\n" + "=" * 72)
    print("  RESULT  (unweighted LDSC; estimates unbiased, SE slightly conservative)")
    print("=" * 72)
    print(f"  M                 : {res['M']:,} SNPs")
    print()
    print(f"  rg(AD, SBP)       : {res['rg']:+.4f}  (SE {res['rg_se']:.4f})")
    print(f"  rg z-score        : {res['rg_z']:+.3f}")
    print(f"  rg p-value        : {res['rg_p']:.3g}")
    print()
    print(f"  h²(AD)            : {res['h2_AD']:.4f}  (SE {res['h2_AD_se']:.4f})")
    print(f"  h²(SBP)           : {res['h2_SBP']:.4f}  (SE {res['h2_SBP_se']:.4f})")
    print()
    print(f"  intercept AD      : {res['h2_AD_intercept']:.4f}")
    print(f"  intercept SBP     : {res['h2_SBP_intercept']:.4f}")
    print(f"  cross intercept   : {res['cross_intercept']:.4f}")
    print()
    print(f"  runtime           : {res['runtime_s']} s")
    print(f"  json saved        : {out}")


if __name__ == "__main__":
    main()
