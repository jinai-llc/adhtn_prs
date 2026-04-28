#!/usr/bin/env python3
"""
Pathway enrichment for 4 AD×SBP TWAS quadrants.

Output schema matches refined figure `pathway_group_barplot()`:
  phase6_pathway/Q1_concordant_risk.tsv        (AD+ SBP+)
  phase6_pathway/Q2_discordant_AD_dn_SBP_up.tsv (AD- SBP+)
  phase6_pathway/Q3_concordant_protective.tsv  (AD- SBP-)
  phase6_pathway/Q4_discordant_AD_up_SBP_dn.tsv (AD+ SBP-)
  phase6_pathway/quadrant_gene_lists.json       (input gene lists per quadrant)
  phase6_pathway/pathway_enrichr_all.tsv        (all hits pooled with Quadrant col)

Columns (per TSV, standard gseapy output):
  Gene_set, Term, Overlap, P-value, Adjusted P-value, Odds Ratio,
  Combined Score, Genes, Quadrant

Quadrant assignment: per gene, we take its AVERAGE AD_z and AVERAGE SBP_z
across the 12–13 brain tissues it's predicted in. Sign of averages defines
quadrant. Ranking within quadrant is by sqrt(mean_AD_z^2 + mean_SBP_z^2).
Top --n-per-quadrant genes per quadrant (default 500) → enrichr against
Reactome_2022, GO_Biological_Process_2023, KEGG_2021_Human.

Pathways of interest (AD + T2D + vascular + neuro biology) are filtered
after enrichment to match the refined figure's whitelist approach.
"""

from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HOME  = Path.home()
BASE  = HOME / "prs_otherdata" / "pipeline"
PHASE3 = BASE / "results_htn" / "phase3_multiomics"
OUT   = BASE / "results_htn" / "phase6_pathway"
OUT.mkdir(parents=True, exist_ok=True)

TWAS_LONG = PHASE3 / "twas_cross_disease_all_tissues.tsv"
LIBRARIES = ["Reactome_2022", "GO_Biological_Process_2023", "KEGG_2021_Human"]

# ── Pathways of interest (adapted from refined figure for AD+SBP) ──────────
# Applied as case-insensitive substring match against pathway Term field.
PATHWAYS_OF_INTEREST = [
    # --- AD / amyloid / tau / neurodegeneration ---
    'amyloid', 'alzheimer', 'tau ', 'neurofibrillary',
    'microglia', 'neurodegeneration', 'astrocyte',
    # --- Cardiovascular / blood pressure / vascular ---  (NEW for AD×SBP)
    'blood pressure', 'hypertension', 'renin', 'angiotensin',
    'aldosterone', 'natriuretic', 'vascular', 'artery',
    'arterial', 'endothelial', 'vasoconstriction', 'vasodilation',
    'smooth muscle', 'cardiac', 'heart',
    # --- Sodium/potassium/calcium handling (BP biology) ---
    'sodium', 'potassium channel', 'calcium signaling', 'calcium ion',
    'ion transport', 'ion channel',
    # --- Insulin / glucose (retained from AD×T2D for bridge pathways) ---
    'insulin', 'glucose', 'incretin',
    # --- Lipid / lipoprotein ---
    'lipoprotein', 'cholesterol', 'lipid', 'apolipoprotein', 'fatty acid',
    # --- Autophagy / lysosome / protein quality ---
    'lysosome', 'autophagy', 'phagosome', 'proteasome',
    'unfolded protein', 'chaperone',
    # --- Synaptic / neuronal ---
    'synap', 'neuro', 'axon', 'dendrit', 'glutamat', 'gaba',
    'vesicle', 'endocyto', 'long-term potentiation',
    # --- Signaling bridges ---
    'mtor', 'ampk', 'pi3k', 'akt', 'irs ', 'erk', 'mapk', 'wnt',
    # --- Immune / inflammation ---
    'complement', 'cytokine', 'immune response', 'inflammation',
    'interferon', 'tnf', 'nf-kappa',
    # --- Mitochondria / energy ---
    'mitochondri', 'oxidative phosphorylation', 'electron transport',
]


def is_pathway_of_interest(term: str) -> bool:
    low = str(term).lower()
    return any(kw in low for kw in PATHWAYS_OF_INTEREST)


# ── load TWAS cross-disease table ─────────────────────────────────────────
def load_twas():
    if not TWAS_LONG.exists():
        raise FileNotFoundError(f"TWAS long table not found: {TWAS_LONG}")
    df = pd.read_csv(TWAS_LONG, sep="\t")
    for c in ("AD_z", "SBP_z", "AD_p", "SBP_p"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["AD_z", "SBP_z", "gene_name"])
    # Clean pseudogenes and LINC
    df = df[~df["gene_name"].str.startswith(
        ('RP11-','LINC','AC0','AC1','CTD-','RP4-','RP3-','CTC-'), na=False)].copy()
    print(f"  TWAS rows (after clean) : {len(df):,}")
    print(f"  unique genes            : {df['gene_name'].nunique():,}")
    print(f"  tissues                 : {df['tissue'].nunique()}")
    return df


# ── aggregate per-gene, assign quadrant, rank, take top-N ─────────────────
def quadrant_of(ad_mean: float, sbp_mean: float) -> str:
    if ad_mean > 0 and sbp_mean > 0: return "Q1"   # concordant risk
    if ad_mean < 0 and sbp_mean > 0: return "Q2"   # discordant AD↓ SBP↑
    if ad_mean < 0 and sbp_mean < 0: return "Q3"   # concordant protective
    if ad_mean > 0 and sbp_mean < 0: return "Q4"   # discordant AD↑ SBP↓
    return "ZERO"


def build_quadrant_genelists(df: pd.DataFrame, n_per_q: int = 500):
    """For each gene, mean AD_z and mean SBP_z across tissues;
       then assign quadrant + rank by sqrt(mean_AD^2 + mean_SBP^2)."""
    agg = df.groupby("gene_name").agg(
        AD_z_mean  = ("AD_z",  "mean"),
        SBP_z_mean = ("SBP_z", "mean"),
        AD_z_max   = ("AD_z",  lambda v: v.values[np.argmax(np.abs(v.values))] if len(v) else np.nan),
        SBP_z_max  = ("SBP_z", lambda v: v.values[np.argmax(np.abs(v.values))] if len(v) else np.nan),
        n_tissues  = ("tissue", "nunique"),
    ).reset_index()
    agg["quadrant"] = agg.apply(
        lambda r: quadrant_of(r["AD_z_mean"], r["SBP_z_mean"]), axis=1)
    agg["mag"] = np.sqrt(agg["AD_z_mean"]**2 + agg["SBP_z_mean"]**2)
    out = {}
    for q in ("Q1","Q2","Q3","Q4"):
        qdf = agg[agg["quadrant"] == q].sort_values("mag", ascending=False)
        top = qdf.head(n_per_q)
        out[q] = {
            "genes": top["gene_name"].tolist(),
            "n_available": int(len(qdf)),
            "n_taken": int(len(top)),
            "median_mag": float(top["mag"].median()) if len(top) else None,
            "top10": top.head(10)[["gene_name","AD_z_mean","SBP_z_mean","mag","n_tissues"]]
                         .to_dict(orient="records"),
        }
    return out, agg


# ── run enrichr for one gene list ─────────────────────────────────────────
def run_enrichr(genes: list[str], tag: str, retries: int = 3):
    try:
        import gseapy as gp
    except ImportError:
        print("  [error] gseapy not installed; run: pip install 'gseapy<1.1.0' --break-system-packages")
        sys.exit(2)

    if len(genes) < 5:
        print(f"    [{tag}] only {len(genes)} genes, skipping")
        return pd.DataFrame()

    results = []
    for lib in LIBRARIES:
        for attempt in range(retries):
            try:
                print(f"    [{tag}] enrichr against {lib} ({len(genes)} genes, try {attempt+1})")
                enr = gp.enrichr(gene_list=genes, gene_sets=lib,
                                  organism="human", outdir=None,
                                  no_plot=True, verbose=False)
                df = enr.results.copy()
                df["Gene_set"] = lib
                results.append(df)
                time.sleep(0.5)
                break
            except Exception as e:
                print(f"    [{tag}] {lib} attempt {attempt+1} failed: {type(e).__name__}: {e}")
                time.sleep(3)
    if not results:
        return pd.DataFrame()
    big = pd.concat(results, ignore_index=True)
    # Standard columns
    for col in ["Term","P-value","Adjusted P-value","Odds Ratio","Combined Score","Genes","Overlap"]:
        if col not in big.columns:
            big[col] = np.nan
    return big


# ── main ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-quadrant", type=int, default=500,
                    help="top N genes per quadrant (default 500)")
    ap.add_argument("--poi-only", action="store_true",
                    help="filter output to pathways-of-interest whitelist only")
    ap.add_argument("--top-per-lib", type=int, default=200,
                    help="keep top N terms per library before filtering")
    args = ap.parse_args()

    print("=" * 72)
    print("  Pathway enrichment — 4 quadrants (AD×SBP TWAS)")
    print(f"  Top {args.n_per_quadrant} genes per quadrant, libraries: {LIBRARIES}")
    print("=" * 72)

    df = load_twas()
    genelists, agg = build_quadrant_genelists(df, n_per_q=args.n_per_quadrant)

    print("\nQuadrant gene-list summary:")
    print(f"  {'Q':<4} {'label':<30} {'n_avail':>8} {'n_taken':>8} {'med_mag':>8}")
    labels = {"Q1":"concordant risk (AD+ SBP+)",
              "Q2":"discordant  (AD- SBP+)",
              "Q3":"concordant protective (AD- SBP-)",
              "Q4":"discordant  (AD+ SBP-)"}
    for q, meta in genelists.items():
        mm = f"{meta['median_mag']:.2f}" if meta['median_mag'] else "NA"
        print(f"  {q:<4} {labels[q]:<30} {meta['n_available']:>8} {meta['n_taken']:>8} {mm:>8}")

    # Save gene lists
    (OUT / "quadrant_gene_lists.json").write_text(json.dumps(genelists, indent=2))
    agg.sort_values("mag", ascending=False).to_csv(
        OUT / "gene_aggregated_cross_tissue.tsv", sep="\t", index=False)

    # Run enrichr per quadrant
    print("\nRunning enrichr per quadrant (3 libraries × 4 quadrants = 12 API calls):")
    all_hits = []
    for q in ("Q1","Q2","Q3","Q4"):
        genes = genelists[q]["genes"]
        print(f"\n{'─'*60}\n{q} : {labels[q]}  (n={len(genes)})")
        res = run_enrichr(genes, tag=q)
        if len(res) == 0:
            print(f"  [{q}] no enrichr results")
            continue
        res["Quadrant"] = q
        # Keep top per library by nominal P
        res = res.sort_values("P-value").groupby("Gene_set", group_keys=False) \
                  .head(args.top_per_lib)
        # Filter by pathways of interest
        res_poi = res[res["Term"].apply(is_pathway_of_interest)].copy()

        # Save full and POI-filtered
        fp_all = OUT / f"{q}_all_terms.tsv"
        fp_poi = OUT / f"{q}_POI_only.tsv"
        res.to_csv(fp_all, sep="\t", index=False)
        res_poi.to_csv(fp_poi, sep="\t", index=False)

        # Also save with the standardised refined-figure filename
        friendly = {"Q1":"Q1_concordant_risk",
                    "Q2":"Q2_discordant_AD_dn_SBP_up",
                    "Q3":"Q3_concordant_protective",
                    "Q4":"Q4_discordant_AD_up_SBP_dn"}
        (OUT / f"{friendly[q]}.tsv").write_text(res_poi.to_csv(sep="\t", index=False))

        print(f"  saved: {fp_all.name}  ({len(res)} terms)")
        print(f"  saved: {fp_poi.name}  ({len(res_poi)} POI terms)")
        # Print top 5 POI hits
        if len(res_poi):
            print(f"  top-5 POI hits in {q}:")
            for _, r in res_poi.head(5).iterrows():
                term = (r["Term"][:55] + "...") if len(str(r["Term"])) > 58 else r["Term"]
                print(f"    {r['Gene_set']:<28} {term:<58}  p = {r['P-value']:.2e}")
        all_hits.append(res_poi if args.poi_only else res)

    # Pooled table
    if all_hits:
        big = pd.concat(all_hits, ignore_index=True)
        big.to_csv(OUT / "pathway_enrichr_all.tsv", sep="\t", index=False)
        print(f"\npooled table saved: {OUT / 'pathway_enrichr_all.tsv'}  ({len(big)} rows)")

    print("\n" + "=" * 72)
    print(f"  Done.  Files in: {OUT}")
    print("=" * 72)


if __name__ == "__main__":
    main()
