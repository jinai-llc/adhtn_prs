#!/usr/bin/env python3
"""
Bidirectional MR: AD <-> SBP (APOE-excluded reverse direction).

Output files (schema matches refined make_merged_fig_prs_mr_nob.py):

  phase6_mr/forward_harmonized_data.csv
  phase6_mr/forward_mr_sbp_to_ad.csv            (symlinked to .._t2d_to_ad.csv)
  phase6_mr/reverse_harmonized_data.csv
  phase6_mr/reverse_mr_ad_to_sbp.csv            (symlinked to .._t2d.csv)
  phase6_mr/reverse_harmonized_data_no_apoe.csv
  phase6_mr/reverse_mr_ad_to_sbp_no_apoe.csv    (symlinked to .._t2d_no_apoe.csv)

Results CSV schema : method, b, se, pval, nsnp
Harmonized CSV    : beta.exposure, se.exposure, beta.outcome, se.outcome, SNP, CHR, BP

Methods:
  - IVW (Inverse variance weighted)
  - MR Egger (with intercept test)
  - Weighted median
  - Weighted mode

Instrument selection protocol (matches AD×T2D paper):
  p < 5e-8, 1 Mb distance-based clumping, palindromic-SNP removal,
  allele harmonization to exposure effect allele (flips outcome beta as needed).

APOE exclusion: ±1 Mb around chr19:45,411,941 (hg38).
"""

from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

HOME = Path.home()
BASE = HOME / "prs_otherdata" / "pipeline"
PHASE1     = BASE / "results_htn" / "phase1_gwas"
PHASE1_AD  = BASE / "results"     / "phase1_gwas"
OUT_DIR    = BASE / "results_htn" / "phase6_mr"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AD_GWAS  = PHASE1_AD / "ad_gwas_harmonized.tsv.gz"     # hg38
SBP_GWAS = PHASE1    / "sbp_gwas_harmonized_hg38.tsv.gz"

# APOE exclusion window (hg38)
APOE_CHR, APOE_START, APOE_END = 19, 44_411_941, 46_411_941

P_THRESHOLD   = 5e-8
CLUMP_DIST_BP = 1_000_000


# ─── loaders ──────────────────────────────────────────────────────────────
def load_gwas(path: Path, label: str) -> pd.DataFrame:
    print(f"  loading {label}: {path.name}")
    df = pd.read_csv(str(path), sep="\t", low_memory=False,
                     dtype={"CHR": str, "BP": "Int64"})
    df = df.rename(columns={c: c.upper() for c in df.columns})
    need = {"CHR", "BP", "A1", "A2", "BETA", "SE", "P"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{label}: missing columns {miss}")
    df = df[df["CHR"].astype(str).str.replace("chr", "").str.isnumeric()].copy()
    df["CHR"]  = pd.to_numeric(df["CHR"].astype(str).str.replace("chr", ""),
                                errors="coerce").astype("Int64")
    df["BP"]   = pd.to_numeric(df["BP"],   errors="coerce").astype("Int64")
    df["BETA"] = pd.to_numeric(df["BETA"], errors="coerce")
    df["SE"]   = pd.to_numeric(df["SE"],   errors="coerce")
    df["P"]    = pd.to_numeric(df["P"],    errors="coerce")
    df = df.dropna(subset=["CHR","BP","A1","A2","BETA","SE","P"])
    df = df[(df["SE"] > 0) & (df["P"] > 0)]
    for c in ("A1","A2"):
        df[c] = df[c].astype(str).str.upper()
    print(f"    valid rows: {len(df):,}")
    return df


# ─── instrument selection ────────────────────────────────────────────────
def is_palindromic(a1: str, a2: str) -> bool:
    complements = {"A":"T", "T":"A", "C":"G", "G":"C"}
    return a1 in complements and complements[a1] == a2


def select_instruments(exposure: pd.DataFrame, label: str,
                        exclude_apoe: bool = False) -> pd.DataFrame:
    """p<5e-8, distance-clumped, palindromic removed."""
    sig = exposure[exposure["P"] < P_THRESHOLD].copy()
    print(f"  {label}  p<{P_THRESHOLD}: {len(sig):,}")
    if exclude_apoe:
        before = len(sig)
        sig = sig[~((sig["CHR"] == APOE_CHR) &
                    (sig["BP"].between(APOE_START, APOE_END)))]
        print(f"    APOE-excluded (chr{APOE_CHR}:{APOE_START:,}-{APOE_END:,}): "
              f"{before - len(sig)} variants dropped, {len(sig):,} remain")

    # Drop palindromic SNPs (strand-ambiguous)
    before = len(sig)
    sig = sig[~sig.apply(lambda r: is_palindromic(r["A1"], r["A2"]), axis=1)]
    print(f"    palindromic removed: {before - len(sig):,} variants dropped, "
          f"{len(sig):,} remain")

    # Sort by p-value ascending, then greedy distance clumping within chromosomes
    sig = sig.sort_values("P", ascending=True).reset_index(drop=True)
    keep = np.ones(len(sig), dtype=bool)
    for chr_val in sig["CHR"].unique():
        chr_idx = np.where((sig["CHR"] == chr_val).values)[0]
        for i_pos in chr_idx:
            if not keep[i_pos]: continue
            bp_i = sig.at[i_pos, "BP"]
            # Mark later variants (higher p) within 1 Mb for removal
            for j_pos in chr_idx:
                if j_pos <= i_pos or not keep[j_pos]: continue
                if abs(int(sig.at[j_pos, "BP"]) - int(bp_i)) < CLUMP_DIST_BP:
                    keep[j_pos] = False
    clumped = sig[keep].reset_index(drop=True)
    print(f"    after 1Mb clumping: {len(clumped):,} independent instruments")
    return clumped


# ─── allele harmonization ─────────────────────────────────────────────────
def harmonize(instruments: pd.DataFrame, outcome: pd.DataFrame,
              exposure_label: str, outcome_label: str) -> pd.DataFrame:
    """Match instruments to outcome by CHR+BP. Align alleles to exposure A1.
       Flip outcome BETA if outcome A1 == exposure A2 (allele swap).
       Drop rows where alleles don't form a matching pair.
    """
    m = instruments.merge(outcome, on=["CHR", "BP"],
                           suffixes=("_exp", "_out"))
    print(f"  instruments ({exposure_label}) matched in outcome "
          f"({outcome_label}) on CHR+BP: {len(m):,}")

    same = (m["A1_exp"] == m["A1_out"]) & (m["A2_exp"] == m["A2_out"])
    flip = (m["A1_exp"] == m["A2_out"]) & (m["A2_exp"] == m["A1_out"])
    keep = same | flip
    m = m[keep].copy()
    # Flip outcome BETA where alleles swapped
    m.loc[flip[keep], "BETA_out"] = -m.loc[flip[keep], "BETA_out"]
    print(f"    after allele harmonization: {len(m):,} instruments")

    harm = pd.DataFrame({
        "SNP": m.get("SNP_exp", [f"chr{c}_{b}" for c, b in zip(m["CHR"], m["BP"])]),
        "CHR":                 m["CHR"].astype(int),
        "BP":                  m["BP"].astype(int),
        "beta.exposure":       m["BETA_exp"].astype(float),
        "se.exposure":         m["SE_exp"].astype(float),
        "beta.outcome":        m["BETA_out"].astype(float),
        "se.outcome":          m["SE_out"].astype(float),
        "effect_allele.exposure": m["A1_exp"].astype(str),
        "other_allele.exposure":  m["A2_exp"].astype(str),
    })
    # Steiger-like sanity: drop rows with zero SE
    harm = harm[(harm["se.exposure"] > 0) & (harm["se.outcome"] > 0)].reset_index(drop=True)
    return harm


# ─── MR methods ───────────────────────────────────────────────────────────
def mr_ivw(bx, by, bxse, byse):
    w = 1.0 / byse**2
    b_num = np.sum(w * bx * by)
    b_den = np.sum(w * bx**2)
    if b_den == 0: return np.nan, np.nan
    b = b_num / b_den
    se = np.sqrt(1.0 / b_den)
    return b, se


def mr_egger(bx, by, bxse, byse):
    # Orient all instruments to positive bx
    sign = np.sign(bx)
    sign[sign == 0] = 1
    bx_o = bx * sign
    by_o = by * sign
    w = 1.0 / byse**2
    W = np.diag(w)
    X = np.column_stack([np.ones_like(bx_o), bx_o])
    # Weighted least squares (analytical)
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ by_o
    try:
        beta = np.linalg.solve(XtWX, XtWy)
        cov  = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, np.nan
    intercept, slope = beta
    se_int, se_slope = np.sqrt(np.diag(cov))
    return slope, se_slope, intercept, se_int


def mr_weighted_median(bx, by, bxse, byse):
    """Bowden et al weighted median."""
    b_iv = by / bx
    w    = (bx / byse) ** 2
    order = np.argsort(b_iv)
    b_sorted = b_iv[order]
    w_sorted = w[order]
    cum_w = np.cumsum(w_sorted) - 0.5 * w_sorted
    cum_w /= np.sum(w_sorted)
    # Interpolate to find 0.5
    k = np.searchsorted(cum_w, 0.5)
    if k == 0:
        b_med = b_sorted[0]
    elif k >= len(cum_w):
        b_med = b_sorted[-1]
    else:
        # Linear interpolation
        frac = (0.5 - cum_w[k-1]) / (cum_w[k] - cum_w[k-1])
        b_med = b_sorted[k-1] + frac * (b_sorted[k] - b_sorted[k-1])
    # Bootstrap SE (1000 reps)
    rng = np.random.default_rng(42)
    n = len(bx)
    boot = np.empty(1000)
    for i in range(1000):
        bx_b = bx + rng.normal(0, bxse, n)
        by_b = by + rng.normal(0, byse, n)
        biv_b = by_b / bx_b
        w_b   = (bx_b / byse) ** 2
        ob = np.argsort(biv_b)
        s_b = biv_b[ob]
        w_b_s = w_b[ob]
        cw = np.cumsum(w_b_s) - 0.5 * w_b_s
        cw /= np.sum(w_b_s)
        k = np.searchsorted(cw, 0.5)
        if k == 0: boot[i] = s_b[0]
        elif k >= len(cw): boot[i] = s_b[-1]
        else:
            frac = (0.5 - cw[k-1]) / (cw[k] - cw[k-1])
            boot[i] = s_b[k-1] + frac * (s_b[k] - s_b[k-1])
    return b_med, float(np.std(boot))


def mr_weighted_mode(bx, by, bxse, byse, phi=1.0):
    """Hartwig et al weighted mode estimator (continuous, normal kernel)."""
    b_iv = by / bx
    se_iv = np.sqrt((byse**2 / bx**2) + (by**2 * bxse**2 / bx**4))
    # Bandwidth: Silverman-like with phi tuning
    h = phi * 0.9 * min(np.std(b_iv),
                         (np.quantile(b_iv, .75) - np.quantile(b_iv, .25)) / 1.34) \
        * len(b_iv) ** (-0.2)
    if h <= 0 or not np.isfinite(h):
        h = 0.01
    grid = np.linspace(b_iv.min(), b_iv.max(), 512)
    # Weighted KDE evaluated on grid (weights = 1/se_iv^2)
    w = 1.0 / se_iv**2
    # density at each grid point
    density = np.zeros_like(grid)
    for i, g in enumerate(grid):
        density[i] = np.sum(w * np.exp(-0.5 * ((b_iv - g) / h) ** 2))
    b_mode = grid[np.argmax(density)]
    # Bootstrap SE
    rng = np.random.default_rng(42)
    n = len(bx)
    boot = np.empty(500)
    for k in range(500):
        bx_b = bx + rng.normal(0, bxse, n)
        by_b = by + rng.normal(0, byse, n)
        biv_b = by_b / bx_b
        se_b  = np.sqrt((byse**2 / bx_b**2) + (by_b**2 * bxse**2 / bx_b**4))
        w_b   = 1.0 / se_b**2
        h_b   = phi * 0.9 * min(np.std(biv_b),
                                 (np.quantile(biv_b, .75) - np.quantile(biv_b, .25)) / 1.34) \
                * n ** (-0.2)
        if h_b <= 0 or not np.isfinite(h_b):
            h_b = 0.01
        dens = np.zeros_like(grid)
        for i, g in enumerate(grid):
            dens[i] = np.sum(w_b * np.exp(-0.5 * ((biv_b - g) / h_b) ** 2))
        boot[k] = grid[np.argmax(dens)]
    return b_mode, float(np.std(boot))


def run_mr(harm: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Run 4 MR methods on harmonized data. Returns results DataFrame."""
    if len(harm) < 3:
        print(f"  [{tag}] fewer than 3 instruments ({len(harm)}), skipping")
        return pd.DataFrame(columns=["method","b","se","pval","nsnp"])
    bx   = harm["beta.exposure"].values
    by   = harm["beta.outcome"].values
    bxse = harm["se.exposure"].values
    byse = harm["se.outcome"].values

    rows = []
    # IVW
    b_ivw, se_ivw = mr_ivw(bx, by, bxse, byse)
    p_ivw = 2 * stats.norm.sf(abs(b_ivw / se_ivw)) if (se_ivw and np.isfinite(se_ivw)) else np.nan
    rows.append({"method":"Inverse variance weighted", "b":b_ivw, "se":se_ivw,
                  "pval":p_ivw, "nsnp":len(bx)})
    # MR Egger
    b_eg, se_eg, int_eg, se_int = mr_egger(bx, by, bxse, byse)
    p_eg = 2 * stats.norm.sf(abs(b_eg / se_eg)) if (se_eg and np.isfinite(se_eg)) else np.nan
    p_int = 2 * stats.norm.sf(abs(int_eg / se_int)) if (se_int and np.isfinite(se_int)) else np.nan
    rows.append({"method":"MR Egger", "b":b_eg, "se":se_eg, "pval":p_eg,
                  "nsnp":len(bx),
                  "egger_intercept": int_eg, "egger_intercept_se": se_int,
                  "egger_intercept_pval": p_int})
    # Weighted median
    b_wm, se_wm = mr_weighted_median(bx, by, bxse, byse)
    p_wm = 2 * stats.norm.sf(abs(b_wm / se_wm)) if (se_wm and np.isfinite(se_wm) and se_wm > 0) else np.nan
    rows.append({"method":"Weighted median", "b":b_wm, "se":se_wm,
                  "pval":p_wm, "nsnp":len(bx)})
    # Weighted mode
    b_wo, se_wo = mr_weighted_mode(bx, by, bxse, byse)
    p_wo = 2 * stats.norm.sf(abs(b_wo / se_wo)) if (se_wo and np.isfinite(se_wo) and se_wo > 0) else np.nan
    rows.append({"method":"Weighted mode", "b":b_wo, "se":se_wo,
                  "pval":p_wo, "nsnp":len(bx)})

    res = pd.DataFrame(rows)
    print(f"  [{tag}] results  (n={len(bx)}):")
    for _, r in res.iterrows():
        p = r["pval"]
        ps = f"{p:.2e}" if np.isfinite(p) else "NA"
        print(f"    {r['method']:<30} b = {r['b']:+.4f}  "
              f"se = {r['se']:.4f}  p = {ps}")
    return res


# ─── orchestrator ────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Bidirectional MR — AD <-> SBP  (APOE-excluded reverse)")
    print("=" * 70)

    ad  = load_gwas(AD_GWAS,  "AD")
    sbp = load_gwas(SBP_GWAS, "SBP")

    # ── FORWARD : SBP -> AD (SBP as exposure, AD as outcome) ──
    print("\n[1/3] Forward: SBP -> AD")
    sbp_iv = select_instruments(sbp, "SBP")
    fwd_harm = harmonize(sbp_iv, ad, "SBP", "AD")
    fwd_harm.to_csv(OUT_DIR / "forward_harmonized_data.csv", index=False)
    fwd_res = run_mr(fwd_harm, "forward SBP->AD")
    fwd_res.to_csv(OUT_DIR / "forward_mr_sbp_to_ad.csv", index=False)
    # Legacy alias so refined figure script finds it without edits
    fwd_res.to_csv(OUT_DIR / "forward_mr_t2d_to_ad.csv", index=False)

    # ── REVERSE : AD -> SBP (all SNPs) ──
    print("\n[2/3] Reverse: AD -> SBP  (all SNPs)")
    ad_iv = select_instruments(ad, "AD")
    rev_harm = harmonize(ad_iv, sbp, "AD", "SBP")
    rev_harm.to_csv(OUT_DIR / "reverse_harmonized_data.csv", index=False)
    rev_res = run_mr(rev_harm, "reverse AD->SBP (all)")
    rev_res.to_csv(OUT_DIR / "reverse_mr_ad_to_sbp.csv", index=False)
    rev_res.to_csv(OUT_DIR / "reverse_mr_ad_to_t2d.csv", index=False)   # legacy alias

    # ── REVERSE : AD -> SBP (APOE excluded) ──
    print("\n[3/3] Reverse: AD -> SBP  (APOE excluded)")
    ad_iv_no = select_instruments(ad, "AD", exclude_apoe=True)
    rev_harm_no = harmonize(ad_iv_no, sbp, "AD (no-APOE)", "SBP")
    rev_harm_no.to_csv(OUT_DIR / "reverse_harmonized_data_no_apoe.csv", index=False)
    rev_res_no = run_mr(rev_harm_no, "reverse AD->SBP (APOE-excl.)")
    rev_res_no.to_csv(OUT_DIR / "reverse_mr_ad_to_sbp_no_apoe.csv", index=False)
    rev_res_no.to_csv(OUT_DIR / "reverse_mr_ad_to_t2d_no_apoe.csv", index=False)

    # Summary json
    summary = {
        "forward_SBP_to_AD":  fwd_res.to_dict(orient="records"),
        "reverse_AD_to_SBP":  rev_res.to_dict(orient="records"),
        "reverse_AD_to_SBP_noAPOE": rev_res_no.to_dict(orient="records"),
        "n_instruments": {
            "forward_SBP_to_AD": int(len(fwd_harm)),
            "reverse_AD_to_SBP": int(len(rev_harm)),
            "reverse_AD_to_SBP_noAPOE": int(len(rev_harm_no)),
        },
    }
    (OUT_DIR / "mr_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 70)
    print("  Done.  Files in:", OUT_DIR)
    print("=" * 70)
    for f in sorted(OUT_DIR.glob("*")):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
