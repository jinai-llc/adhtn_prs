#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  HYPERTENSION × AD MULTI-LEVEL PRS PIPELINE
  Adapted from ad_t2d_pipeline_complete.py (962 lines)

  GWAS sources:
    AD  : Bellenguez et al. 2022 Nat Genet 54:412   (GCST90565439) — GRCh38
    SBP : Keaton  et al. 2024 Nat Genet 56:778     (Keaton 2024) — GRCh37
    DBP : Keaton  et al. 2024 Nat Genet 56:778     (Keaton 2024) — GRCh37
    PP  : Keaton  et al. 2024 Nat Genet 56:778     (Keaton 2024) — GRCh37

  Design:
    - Same 7-phase architecture as AD/T2D pipeline
    - Trait-agnostic: config-driven; add a new trait by adding a TRAIT_CONFIG entry
    - Primary arm is SBP × AD (largest PRS effect); DBP, PP run as sensitivity

  Differences from T2D arm (things you do NOT need to change):
    - Same hg19→hg38 liftOver strategy (Keaton is GRCh37, matches T2D)
    - Same PRS-CS-auto LD reference (1000G EUR), same snpinfo liftOver
    - Same MKL_NUM_THREADS=5, 6 parallel chromosome jobs
    - Same custom Python 3 LDSC, gene-burden, S-PrediXcan calls
    - Same merged Fig 2-4 and Fig 6 plotting — just swap the secondary-trait color
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIG — the only section you should edit per trait
# ─────────────────────────────────────────────────────────────────────────────

BASE = Path.home() / "prs_otherdata"
GWAS_DIR = Path.home() / "gwas_summary"
PIPELINE = BASE / "pipeline"

# ── Where the Keaton files land after `gsutil cp` from your bucket ───────────
#    Edit this single path once you confirm the bucket filenames.
KEATON_LOCAL_DIR = GWAS_DIR / "HTN"

# Trait registry. Add/remove entries to extend the pipeline.
#   - gwas_file: path to the raw summary statistics on this machine
#   - n_gwas: sample size for PRS-CS --n_gwas (Neff for case/control, total N for continuous)
#   - genome_build: "hg19" or "hg38"  (drives whether liftOver is needed)
#   - rename_map: maps raw column names → canonical {CHR,BP,SNP,A1,A2,BETA,SE,P,FRQ,N}
#   - snp_from_chrpos: if True, synthesize SNP as "chr:pos" (no rsID in file)
TRAIT_CONFIG = {
    "AD": {
        "label": "Alzheimer's disease (Bellenguez 2022)",
        "gwas_file": str(GWAS_DIR / "AD" / "GCST90565439.tsv.gz"),
        "n_gwas": 487_511,   # Neff; kept from your existing pipeline
        "genome_build": "hg38",
        "rename_map": {
            "chromosome": "CHR", "base_pair_location": "BP", "variant_id": "SNP", "rs_id": "SNP",
            "effect_allele": "A1", "other_allele": "A2", "beta": "BETA",
            "standard_error": "SE", "p_value": "P",
            "effect_allele_frequency": "FRQ", "n": "N",
        },
        "snp_from_chrpos": False,
        "color": "#B71C1C",   # deep red (AD)
    },
    "SBP": {
        "label": "Systolic BP (Keaton 2024, GCST90310294)",
        "gwas_file": str(KEATON_LOCAL_DIR / "GCST90310294SBP.tsv.gz"),
        "n_gwas": 1_028_980,
        "genome_build": "hg19",
        "prs_method": "sbayesrc",        # use published weights instead of re-running PRS-CS
        "pgs_catalog_id": "PGS004603",   # SBayesRC, 7,356,519 variants, hg19 native / hg38 harmonized
        # GWAS Catalog harmonized format — update after we inspect the file if fields differ
        "rename_map": {
            "chromosome": "CHR", "base_pair_location": "BP", "variant_id": "SNP", "rs_id": "SNP",
            "effect_allele": "A1", "other_allele": "A2", "beta": "BETA",
            "standard_error": "SE", "p_value": "P",
            "effect_allele_frequency": "FRQ", "n": "N",
        },
        "snp_from_chrpos": False,  # flip to True if the file has no rsID column
        "color": "#1565C0",   # deep blue (primary SBP)
    },
    "DBP": {
        "label": "Diastolic BP (Keaton 2024, GCST90310295)",
        "gwas_file": str(KEATON_LOCAL_DIR / "GCST90310295DBP.tsv.gz"),
        "n_gwas": 1_028_980,
        "genome_build": "hg19",
        "prs_method": "sbayesrc",
        "pgs_catalog_id": "PGS004604",
        "rename_map": {
            "chromosome": "CHR", "base_pair_location": "BP", "variant_id": "SNP", "rs_id": "SNP",
            "effect_allele": "A1", "other_allele": "A2", "beta": "BETA",
            "standard_error": "SE", "p_value": "P",
            "effect_allele_frequency": "FRQ", "n": "N",
        },
        "snp_from_chrpos": False,
        "color": "#6A1B9A",   # purple (DBP)
    },
    "PP": {
        "label": "Pulse pressure (Keaton 2024, GCST90310296)",
        "gwas_file": str(KEATON_LOCAL_DIR / "GCST90310296PP.tsv.gz"),
        "n_gwas": 1_028_980,
        "genome_build": "hg19",
        "prs_method": "sbayesrc",
        "pgs_catalog_id": "PGS004605",
        "rename_map": {
            "chromosome": "CHR", "base_pair_location": "BP", "variant_id": "SNP", "rs_id": "SNP",
            "effect_allele": "A1", "other_allele": "A2", "beta": "BETA",
            "standard_error": "SE", "p_value": "P",
            "effect_allele_frequency": "FRQ", "n": "N",
        },
        "snp_from_chrpos": False,
        "color": "#2E7D32",   # green (PP)
    },
}

# Default run config — can be overridden on the CLI
PRIMARY_TRAIT = "AD"
SECONDARY_TRAIT = "SBP"      # swap to "DBP" or "PP" for sensitivity runs

RESULTS = PIPELINE / "results_htn"           # new output tree, keeps AD/T2D results untouched
PHASE1 = RESULTS / "phase1_gwas"
PHASE2 = RESULTS / "phase2_prs"
PHASE3 = RESULTS / "phase3_multiomics"
PHASE4 = RESULTS / "phase4_classification"

# Legacy results tree from the AD/T2D run — holds existing AD PRS-CS-auto weights
# that we reuse for the hypertension × AD comparison (no re-run needed).
RESULTS_LEGACY = PIPELINE / "results"
PHASE2_LEGACY = RESULTS_LEGACY / "phase2_prs"
PHASE1_LEGACY = RESULTS_LEGACY / "phase1_gwas"

# Computational resources (unchanged from AD/T2D run)
MKL_THREADS = 5
PARALLEL_CHRS = 6   # 6 × 5 = 30 vCPUs matches c2-standard-30

# Tool paths (unchanged)
PRSCS = BASE / "omics_resources" / "prs_tools" / "PRScs" / "PRScs.py"
LD_REF = BASE / "omics_resources" / "ld_reference" / "ldblk_1kg_eur"
HM3_BIM = PHASE2 / "hm3_snps"   # produced by your existing Phase 2.1 step
LIFTOVER = BASE / "pipeline" / "results" / "phase1_gwas" / "liftOver"
CHAIN_19_38 = BASE / "pipeline" / "results" / "phase1_gwas" / "hg19ToHg38.over.chain.gz"

# ─────────────────────────────────────────────────────────────────────────────
# 2. INSPECT — sanity-check the raw file before anything else
# ─────────────────────────────────────────────────────────────────────────────

def inspect_trait(trait: str) -> None:
    """Print header + 3 sample rows for a trait's GWAS file; verify rename_map covers all needed fields."""
    cfg = TRAIT_CONFIG[trait]
    path = cfg["gwas_file"]
    print(f"\n=== Inspecting {trait}: {cfg['label']} ===")
    print(f"  file : {path}")
    if not Path(path).exists():
        print(f"  !! MISSING — pull from bucket first:")
        print(f"     gsutil cp gs://<your-bucket>/keaton_2024_{trait.lower()}.tsv.gz {path}")
        return
    # Peek at header
    opener = "zcat" if path.endswith(".gz") else "cat"
    hdr = subprocess.check_output(f"{opener} {path} | head -1", shell=True, text=True).strip().split("\t")
    print(f"  header ({len(hdr)} cols): {hdr}")
    sample = subprocess.check_output(f"{opener} {path} | sed -n '2,4p'", shell=True, text=True)
    print(f"  first 3 rows:\n{sample}")

    # Check rename_map coverage
    needed = {"CHR", "BP", "A1", "A2", "BETA", "SE", "P"}
    mapped_to = set(cfg["rename_map"].values())
    missing_from_map = needed - mapped_to
    if missing_from_map:
        print(f"  WARNING: rename_map does not cover: {missing_from_map}")
    raw_missing = [k for k in cfg["rename_map"] if k not in hdr]
    if raw_missing:
        print(f"  WARNING: these keys in rename_map are NOT in the file header: {raw_missing}")
        print(f"           → edit TRAIT_CONFIG['{trait}']['rename_map'] to match the real column names")
    print(f"  n_gwas      : {cfg['n_gwas']:,}")
    print(f"  build       : {cfg['genome_build']}")
    print(f"  snp_from_chrpos: {cfg['snp_from_chrpos']}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PHASE 1 — harmonize + liftOver (per trait)
# ─────────────────────────────────────────────────────────────────────────────

def phase1_harmonize(trait: str) -> Path:
    """
    Harmonize one trait's GWAS to canonical columns using polars (10-50× faster than pandas).
    Returns path to harmonized file.
    """
    import polars as pl
    cfg = TRAIT_CONFIG[trait]
    PHASE1.mkdir(parents=True, exist_ok=True)
    out = PHASE1 / f"{trait.lower()}_gwas_harmonized.tsv"

    print(f"\n--- Phase 1.1 harmonize {trait} ---")
    t0 = time.time()
    df = pl.read_csv(cfg["gwas_file"], separator="\t", infer_schema_length=10_000)
    print(f"  raw rows    : {len(df):,}")
    print(f"  raw columns : {df.columns}")

    # Apply rename map only for keys that actually exist
    actual_map = {k: v for k, v in cfg["rename_map"].items() if k in df.columns}
    df = df.rename(actual_map)

    # Synthesize SNP ID if needed
    if cfg["snp_from_chrpos"] or "SNP" not in df.columns:
        df = df.with_columns(
            pl.concat_str([pl.col("CHR").cast(pl.Utf8), pl.lit(":"), pl.col("BP").cast(pl.Utf8)]).alias("SNP")
        )

    # If N column missing, add constant from config
    if "N" not in df.columns:
        df = df.with_columns(pl.lit(cfg["n_gwas"]).alias("N"))
    if "FRQ" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("FRQ"))

    cols = ["CHR", "BP", "SNP", "A1", "A2", "BETA", "SE", "P", "FRQ", "N"]
    df = df.select([c for c in cols if c in df.columns])

    # Strip "chr" prefix from CHR if present
    df = df.with_columns(
        pl.col("CHR").cast(pl.Utf8).str.replace(r"^chr", "").alias("CHR")
    )

    # Filter null essentials
    df = df.filter(
        pl.col("A1").is_not_null() & pl.col("BETA").is_not_null() & pl.col("P").is_not_null()
    )
    df.write_csv(out, separator="\t")
    subprocess.run(["gzip", "-f", str(out)], check=True)
    print(f"  harmonized  : {len(df):,} variants in {time.time()-t0:.1f}s")
    print(f"  saved       : {out}.gz")
    return Path(f"{out}.gz")


def phase1_liftover(trait: str) -> Path:
    """LiftOver hg19→hg38 if needed. Returns final harmonized hg38 path."""
    cfg = TRAIT_CONFIG[trait]
    harm = PHASE1 / f"{trait.lower()}_gwas_harmonized.tsv.gz"
    if cfg["genome_build"] == "hg38":
        print(f"  {trait}: already hg38, skip liftOver")
        return harm

    print(f"\n--- Phase 1.2 liftOver {trait} hg19 → hg38 ---")
    bed_in = PHASE1 / f"{trait.lower()}_hg19.bed"
    bed_out = PHASE1 / f"{trait.lower()}_hg38.bed"
    bed_unmapped = PHASE1 / f"{trait.lower()}_unmapped.bed"
    harm_hg38 = PHASE1 / f"{trait.lower()}_gwas_harmonized_hg38.tsv"

    # Build BED (chr\tstart\tend\tSNP)
    import polars as pl
    df = pl.read_csv(harm, separator="\t")
    bed = df.with_columns([
        (pl.lit("chr") + pl.col("CHR").cast(pl.Utf8)).alias("chr"),
        (pl.col("BP") - 1).alias("start"),
        pl.col("BP").alias("end"),
        pl.col("SNP").alias("name"),
    ]).select(["chr", "start", "end", "name"])
    bed.write_csv(bed_in, separator="\t", include_header=False)

    subprocess.run(
        [str(LIFTOVER), str(bed_in), str(CHAIN_19_38), str(bed_out), str(bed_unmapped)],
        check=True,
    )

    # Merge lifted coords back
    lifted = pl.read_csv(bed_out, separator="\t", has_header=False,
                         new_columns=["chr_new", "start_new", "end_new", "SNP"])
    lifted = lifted.with_columns([
        pl.col("chr_new").str.replace(r"^chr", "").alias("CHR_hg38"),
        pl.col("end_new").alias("BP_hg38"),
    ]).select(["SNP", "CHR_hg38", "BP_hg38"])

    merged = df.join(lifted, on="SNP", how="inner") \
               .drop(["CHR", "BP"]) \
               .rename({"CHR_hg38": "CHR", "BP_hg38": "BP"}) \
               .select(["CHR", "BP", "SNP", "A1", "A2", "BETA", "SE", "P", "FRQ", "N"])
    merged.write_csv(harm_hg38, separator="\t")
    subprocess.run(["gzip", "-f", str(harm_hg38)], check=True)

    n_in  = len(df)
    n_out = len(merged)
    print(f"  mapped  : {n_out:,} / {n_in:,}  ({100*n_out/n_in:.2f}%)")
    print(f"  saved   : {harm_hg38}.gz")
    return Path(f"{harm_hg38}.gz")


# ─────────────────────────────────────────────────────────────────────────────
# 4. PHASE 2 — PRS-CS-auto (per trait)
# ─────────────────────────────────────────────────────────────────────────────

def phase2_prscs(trait: str) -> None:
    """
    Run PRS-CS-auto for a single trait.
    Uses the SAME hg38-lifted snpinfo LD panel as the AD/T2D run
    (produced by your existing pipeline; do NOT re-run the liftover of snpinfo).
    """
    cfg = TRAIT_CONFIG[trait]
    PHASE2.mkdir(parents=True, exist_ok=True)
    prscs_input = PHASE2 / f"{trait.lower()}_prscs_input.txt"
    outdir = PHASE2 / f"{trait.lower()}_prscs"
    outdir.mkdir(exist_ok=True)

    # PRS-CS input format: SNP A1 A2 BETA P
    import polars as pl
    harm_path = PHASE1 / (f"{trait.lower()}_gwas_harmonized_hg38.tsv.gz"
                           if cfg["genome_build"] == "hg19"
                           else f"{trait.lower()}_gwas_harmonized.tsv.gz")
    df = pl.read_csv(harm_path, separator="\t")
    df.select(["SNP", "A1", "A2", "BETA", "P"]).write_csv(prscs_input, separator="\t")
    print(f"\n--- Phase 2.1 PRS-CS input for {trait}: {len(df):,} rows → {prscs_input}")

    # Emit the shell launcher (same pattern as AD/T2D)
    launcher = PHASE2 / f"run_prscs_{trait.lower()}.sh"
    with open(launcher, "w") as f:
        f.write(f"""#!/bin/bash
set -euo pipefail
export MKL_NUM_THREADS={MKL_THREADS}
export OMP_NUM_THREADS={MKL_THREADS}
export OPENBLAS_NUM_THREADS={MKL_THREADS}

echo "[PRS-CS-auto] trait={trait}  N={cfg['n_gwas']}  threads={MKL_THREADS}  parallel={PARALLEL_CHRS}"
for CHR in $(seq 1 22); do
    python3 {PRSCS} \\
        --ref_dir={LD_REF} \\
        --bim_prefix={HM3_BIM} \\
        --sst_file={prscs_input} \\
        --n_gwas={cfg['n_gwas']} \\
        --chrom=$CHR \\
        --out_dir={outdir}/{trait.lower()} \\
        --seed=42 &
    if (( CHR % {PARALLEL_CHRS} == 0 )); then wait; fi
done
wait
echo "[PRS-CS-auto] done. weights in {outdir}/"
""")
    os.chmod(launcher, 0o755)
    print(f"  launcher ready: {launcher}")
    print(f"  to run : nohup bash {launcher} > {launcher}.log 2>&1 &")


def phase2_sbayesrc(trait: str) -> None:
    """
    Alternative to phase2_prscs(): download published SBayesRC weights from
    PGS Catalog (Keaton 2024) and convert them to the same on-disk format
    as PRS-CS output so downstream gene_burden / concordance code works
    unchanged.

    Why SBayesRC over PRS-CS-auto for the hypertension arm:
      - 7,356,519 variants vs ~1.1M HM3 (6-7x more resolution)
      - SBayesRC uses per-SNP functional annotations (PRS-CS-auto does not)
      - Hyperparameters tuned by the Keaton authors on full n=1,028,980
      - Pre-validated: Lifelines EUR R^2=0.1137 (SBP), All of Us AA
        beta=10.59 mmHg high vs low tertile
      - ~5 min download vs ~4 h compute

    Note: AD remains on PRS-CS-auto. For the gene-level burden comparison
    the loader intersects on gene identity, not SNP set, so the SNP count
    asymmetry is absorbed. If you want a strict same-SNP comparison, an
    inner join on SNP rsID before burden computation does the job.
    """
    import polars as pl
    cfg = TRAIT_CONFIG[trait]
    pgs_id = cfg.get("pgs_catalog_id")
    if not pgs_id:
        raise ValueError(f"No pgs_catalog_id in TRAIT_CONFIG[{trait!r}]")

    outdir = PHASE2 / f"{trait.lower()}_sbayesrc"
    outdir.mkdir(parents=True, exist_ok=True)
    # Harmonized GRCh38 scoring file — lets us skip liftOver entirely
    url = (f"https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/"
           f"{pgs_id}/ScoringFiles/Harmonized/{pgs_id}_hmPOS_GRCh38.txt.gz")
    local = outdir / f"{pgs_id}_hmPOS_GRCh38.txt.gz"

    if not local.exists() or local.stat().st_size == 0:
        print(f"\n--- Phase 2.1b SBayesRC download {trait} ({pgs_id}) ---")
        print(f"  fetch: {url}")
        subprocess.run(["wget", "-q", "-O", str(local), url], check=True)
        print(f"  saved: {local}  ({local.stat().st_size/1e6:.1f} MB)")
    else:
        print(f"  [skip] {local} already present")

    # PGS Catalog harmonized scoring file: comment lines start with '#',
    # then a header row. Harmonized columns: hm_rsID, hm_chr, hm_pos, ...
    # Original columns: rsID, chr_name, chr_position, effect_allele,
    #                   other_allele, effect_weight
    df = pl.read_csv(local, separator="\t", comment_prefix="#",
                      infer_schema_length=10_000)
    print(f"  raw shape  : {df.shape}  cols: {df.columns[:12]}...")

    # Canonicalize column names. Prefer harmonized (hm_*) if present.
    col_aliases = {
        "CHR": ["hm_chr", "chr_name"],
        "BP":  ["hm_pos", "chr_position"],
        "SNP": ["hm_rsID", "rsID"],
        "A1":  ["effect_allele"],
        "A2":  ["other_allele", "hm_inferOtherAllele"],
        "BETA_PRS": ["effect_weight"],
    }
    rename_map = {}
    for canon, candidates in col_aliases.items():
        for c in candidates:
            if c in df.columns:
                rename_map[c] = canon
                break
    missing = [c for c in col_aliases if c not in rename_map.values()]
    if missing:
        raise RuntimeError(f"SBayesRC file missing columns {missing}; header was {df.columns}")
    df = df.rename(rename_map).select(list(col_aliases.keys()))

    # Clean CHR (strip 'chr', drop non-autosomal / nulls) and cast BP
    df = df.with_columns(
        pl.col("CHR").cast(pl.Utf8).str.replace(r"^chr", "").alias("CHR"),
        pl.col("BP").cast(pl.Int64),
        pl.col("BETA_PRS").cast(pl.Float64),
    ).filter(
        pl.col("CHR").is_in([str(i) for i in range(1, 23)]) &
        pl.col("BP").is_not_null() &
        pl.col("BETA_PRS").is_not_null()
    )

    n_total = len(df)
    print(f"  autosomal  : {n_total:,} variants with valid hg38 positions")

    # Split by chromosome into files matching PRS-CS output naming so that
    # load_weights() picks them up with zero code changes.
    for chr_ in range(1, 23):
        sub = df.filter(pl.col("CHR") == str(chr_)) \
                .select(["CHR", "SNP", "BP", "A1", "A2", "BETA_PRS"])
        out_f = outdir / f"{trait.lower()}_pst_eff_a1_b0.5_phiauto_chr{chr_}.txt"
        sub.write_csv(out_f, separator="\t", include_header=False)
    print(f"  wrote 22 per-chromosome weight files → {outdir}/")
    print(f"  format: CHR SNP BP A1 A2 BETA_PRS (matches PRS-CS output)")
    print(f"  downstream: phase2_gene_burden() auto-detects sbayesrc/ dir")


# ─────────────────────────────────────────────────────────────────────────────
# 5. PHASE 2.5 — gene-level PRS burden (pair of traits)
# ─────────────────────────────────────────────────────────────────────────────

def phase2_gene_burden(trait_a: str, trait_b: str) -> None:
    """
    Map PRS-CS weights to genes, compute |Σ β|·frac, and cross-tabulate
    concordant vs discordant gene burdens between two traits.
    Same logic as in your AD/T2D script — only file paths change.
    """
    import polars as pl, numpy as np
    print(f"\n--- Phase 2.5 gene burden {trait_a} vs {trait_b} ---")

    def load_weights(t: str) -> pl.DataFrame:
        # Search order: new tree first (current run), then legacy (AD/T2D run weights).
        # Within each tree, sbayesrc/ takes precedence over prscs/.
        candidates = [
            (PHASE2 / f"{t.lower()}_sbayesrc", "SBayesRC", "results_htn"),
            (PHASE2 / f"{t.lower()}_prscs",    "PRS-CS-auto", "results_htn"),
            (PHASE2_LEGACY / f"{t.lower()}_sbayesrc", "SBayesRC", "results"),
            (PHASE2_LEGACY / f"{t.lower()}_prscs",    "PRS-CS-auto", "results"),
        ]
        for weights_dir, method, tree in candidates:
            if not weights_dir.exists():
                continue
            parts = []
            for chr_ in range(1, 23):
                f = weights_dir / f"{t.lower()}_pst_eff_a1_b0.5_phiauto_chr{chr_}.txt"
                if f.exists() and f.stat().st_size > 0:
                    parts.append(pl.read_csv(f, separator="\t", has_header=False,
                                             new_columns=["CHR", "SNP", "BP", "A1", "A2", "BETA_PRS"]))
            if parts:
                w = pl.concat(parts)
                print(f"  {t}: {len(w):,} SNPs from {method} ({tree}/{weights_dir.name})")
                return w
        raise RuntimeError(
            f"No PRS weight files found for {t}. Searched:\n  " +
            "\n  ".join(str(p[0]) for p in candidates)
        )

    wa = load_weights(trait_a)
    wb = load_weights(trait_b)

    # Gene annotation: reuse ADsum_GCST_cortex_Tigar_TWAS.xlsx as fallback position table.
    gene_pos_candidates = [
        BASE / "ADsum_GCST_cortex_Tigar_TWAS.xlsx",
        PHASE3 / "ADsum_GCST_cortex_Tigar_TWAS.xlsx",
        RESULTS_LEGACY / "phase3_multiomics" / "ADsum_GCST_cortex_Tigar_TWAS.xlsx",
    ]
    gene_pos = next((p for p in gene_pos_candidates if p.exists()), None)
    if gene_pos is None:
        raise FileNotFoundError("Gene position file not found. Searched:\n  " +
            "\n  ".join(str(p) for p in gene_pos_candidates))
    print(f"  gene positions : {gene_pos}")
    raw = pl.read_excel(str(gene_pos))
    print(f"  raw columns    : {raw.columns}")
    aliases = {
        "gene_name": ["gene_name","GeneName","gene","Gene","TargetID","GeneID","gene_symbol","symbol","Symbol"],
        "CHR":       ["CHR","chr","chromosome","Chromosome","CHROM","chrom"],
        "start":     ["start","Start","GeneStart","gene_start","tx_start","txStart"],
        "end":       ["end","End","GeneEnd","gene_end","tx_end","txEnd","stop"],
    }
    rename_map = {}
    for canon, cands in aliases.items():
        for c in cands:
            if c in raw.columns:
                rename_map[c] = canon; break
    missing = [c for c in aliases if c not in rename_map.values()]
    if missing:
        raise RuntimeError(f"Missing cols {missing}. Available: {raw.columns}")
    genes = raw.rename(rename_map).select(["gene_name","CHR","start","end"]).unique("gene_name")
    genes = genes.with_columns(
        pl.col("CHR").cast(pl.Utf8).str.replace(r"^chr","").alias("CHR"),
        pl.col("start").cast(pl.Int64),
        pl.col("end").cast(pl.Int64),
    ).filter(pl.col("CHR").is_in([str(i) for i in range(1,23)]) &
             pl.col("start").is_not_null() & pl.col("end").is_not_null())
    print(f"  genes loaded   : {len(genes):,}")

    def burden(w: pl.DataFrame, label: str) -> pl.DataFrame:
        # Vectorized per-gene sum of |BETA_PRS| over SNPs in [start, end]
        # (Keep this structure identical to the AD/T2D run so comparisons are apples-to-apples.)
        w = w.with_columns(pl.col("CHR").cast(pl.Utf8).str.replace(r"^chr","").alias("CHR"), pl.col("BP").cast(pl.Int64), pl.col("BETA_PRS").cast(pl.Float64))
        out = []
        for gene, chr_, s, e in genes.iter_rows():
            hits = w.filter((pl.col("CHR") == str(chr_)) & (pl.col("BP") >= s) & (pl.col("BP") <= e))
            if len(hits):
                out.append({"gene_name": gene,
                            f"{label}_nsnp": len(hits),
                            f"{label}_burden": float(np.abs(hits["BETA_PRS"].to_numpy()).sum()),
                            f"{label}_signed": float(hits["BETA_PRS"].to_numpy().sum())})
        return pl.DataFrame(out)

    ba = burden(wa, trait_a)
    bb = burden(wb, trait_b)
    merged = ba.join(bb, on="gene_name", how="inner")
    # Concordance by sign of signed sum
    merged = merged.with_columns(
        (pl.col(f"{trait_a}_signed") * pl.col(f"{trait_b}_signed") > 0).alias("concordant")
    )
    out = PHASE2 / f"gene_burden_{trait_a}_{trait_b}.tsv"
    merged.write_csv(out, separator="\t")
    print(f"  genes scored : {len(merged):,}")
    print(f"  concordant   : {merged.filter(pl.col('concordant')).height:,}")
    print(f"  discordant   : {merged.filter(~pl.col('concordant')).height:,}")
    print(f"  saved        : {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. PHASE 3 — S-PrediXcan TWAS (secondary trait only; AD TWAS already exists)
# ─────────────────────────────────────────────────────────────────────────────

def phase3_twas(trait: str, tissues: list[str] | None = None) -> None:
    """
    Emit one S-PrediXcan launcher per GTEx v8 MASHR tissue.
    For hypertension, we use the 13-tissue brain panel (same as AD) so the
    cross-disease Fig 6 heatmap strip stays comparable.
    """
    if tissues is None:
        tissues = [
            "Brain_Amygdala", "Brain_Anterior_cingulate_cortex_BA24",
            "Brain_Caudate_basal_ganglia", "Brain_Cerebellar_Hemisphere",
            "Brain_Cerebellum", "Brain_Cortex", "Brain_Frontal_Cortex_BA9",
            "Brain_Hippocampus", "Brain_Hypothalamus",
            "Brain_Nucleus_accumbens_basal_ganglia", "Brain_Putamen_basal_ganglia",
            "Brain_Spinal_cord_cervical_c-1", "Brain_Substantia_nigra",
        ]
    PHASE3.mkdir(parents=True, exist_ok=True)
    cfg = TRAIT_CONFIG[trait]
    fname_hg38 = f"{trait.lower()}_gwas_harmonized_hg38.tsv.gz"
    fname      = f"{trait.lower()}_gwas_harmonized.tsv.gz"
    target = fname_hg38 if cfg["genome_build"] == "hg19" else fname
    harm_path = None
    for base in (PHASE1, PHASE1_LEGACY):
        p = base / target
        if p.exists():
            harm_path = p
            break
    if harm_path is None:
        raise FileNotFoundError(f"Cannot find {target} in {PHASE1} or {PHASE1_LEGACY}")

    MASHR = BASE / "omics_resources" / "spredixcan" / "eqtl" / "mashr"
    SPREDIXCAN = BASE / "omics_resources" / "MetaXcan" / "software" / "SPrediXcan.py"

    for tissue in tissues:
        script = PHASE3 / f"run_spredixcan_{trait.lower()}_{tissue}.sh"
        with open(script, "w") as f:
            f.write(f"""#!/bin/bash
set -euo pipefail
python3 {SPREDIXCAN} \\
  --model_db_path {MASHR}/mashr_{tissue}.db \\
  --covariance     {MASHR}/mashr_{tissue}.txt.gz \\
  --gwas_file      {harm_path} \\
  --snp_column     SNP --chromosome_column CHR --position_column BP \\
  --effect_allele_column A1 --non_effect_allele_column A2 \\
  --beta_column BETA --se_column SE --pvalue_column P \\
  --keep_non_rsid --additional_output --throw \\
  --output_file    {PHASE3}/spredixcan_{trait.lower()}_{tissue}.csv
""")
        os.chmod(script, 0o755)
    print(f"\n--- Phase 3 S-PrediXcan scripts for {trait}: {len(tissues)} tissues → {PHASE3}")
    print(f"  run in parallel (4 at a time):")
    print(f"  ls {PHASE3}/run_spredixcan_{trait.lower()}_*.sh | xargs -n1 -P4 bash")


# ─────────────────────────────────────────────────────────────────────────────
# 7. PHASE 4 — LDSC rg between two traits
# ─────────────────────────────────────────────────────────────────────────────

def phase4_ldsc_rg(trait_a: str, trait_b: str) -> None:
    """
    Custom Python 3 LDSC rg, identical to the one used for AD/T2D.
    Uses harmonized hg38 files for both traits (LD scores are hg38-indexed).
    """
    import importlib.util
    PHASE4.mkdir(parents=True, exist_ok=True)
    # Path to your existing custom LDSC implementation
    ldsc_py = PIPELINE / "ldsc_py3" / "ldsc_rg.py"
    if not ldsc_py.exists():
        print(f"  !! custom LDSC script not found at {ldsc_py}")
        print(f"  !! the AD/T2D pipeline already produced it — confirm the path")
        return

    def resolve_harm(t: str) -> Path:
        """Find the harmonized hg38 GWAS file for a trait across new + legacy trees."""
        cfg = TRAIT_CONFIG[t]
        fname_hg38 = f"{t.lower()}_gwas_harmonized_hg38.tsv.gz"
        fname      = f"{t.lower()}_gwas_harmonized.tsv.gz"
        target = fname_hg38 if cfg["genome_build"] == "hg19" else fname
        for base in (PHASE1, PHASE1_LEGACY):
            p = base / target
            if p.exists():
                return p
        raise FileNotFoundError(
            f"Cannot find {target} in {PHASE1} or {PHASE1_LEGACY} for {t}"
        )

    a_harm = resolve_harm(trait_a)
    b_harm = resolve_harm(trait_b)

    out = PHASE4 / f"ldsc_rg_{trait_a}_{trait_b}.json"
    cmd = ["python3", str(ldsc_py),
           "--trait1", str(a_harm), "--trait2", str(b_harm),
           "--n1", str(TRAIT_CONFIG[trait_a]["n_gwas"]),
           "--n2", str(TRAIT_CONFIG[trait_b]["n_gwas"]),
           "--out", str(out)]
    print(f"\n--- Phase 4 LDSC rg {trait_a} × {trait_b} ---")
    print(f"  trait1 : {a_harm}")
    print(f"  trait2 : {b_harm}")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"  rg saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. DRIVER
# ─────────────────────────────────────────────────────────────────────────────

def run_full(secondary: str) -> None:
    print("=" * 76)
    print(f"  Hypertension × AD pipeline — {PRIMARY_TRAIT} × {secondary}")
    print("=" * 76)

    # 0. inspect both traits
    inspect_trait(PRIMARY_TRAIT)
    inspect_trait(secondary)

    # Bail if files missing
    for t in (PRIMARY_TRAIT, secondary):
        if not Path(TRAIT_CONFIG[t]["gwas_file"]).exists():
            print(f"\n!! {t} GWAS file missing. Pull from bucket, then re-run.")
            return

    # 1. harmonize
    phase1_harmonize(PRIMARY_TRAIT)
    phase1_harmonize(secondary)

    # 1b. liftOver hg19 traits to hg38 (needed for rg, TWAS even if PRS uses hg38-harmonized weights)
    phase1_liftover(PRIMARY_TRAIT)
    phase1_liftover(secondary)

    # 2. PRS weights — branch on method
    #    AD: PRS-CS-auto (already computed; skip if weights exist)
    #    BP traits: SBayesRC via PGS Catalog (published, author-tuned, 7M variants)
    method_sec = TRAIT_CONFIG[secondary].get("prs_method", "prscs")
    if method_sec == "sbayesrc":
        phase2_sbayesrc(secondary)
    else:
        phase2_prscs(secondary)

    method_pri = TRAIT_CONFIG[PRIMARY_TRAIT].get("prs_method", "prscs")
    ad_weights_dir = PHASE2 / f"{PRIMARY_TRAIT.lower()}_prscs"
    if method_pri == "prscs" and ad_weights_dir.exists() and any(ad_weights_dir.glob("*pst_eff*.txt")):
        print(f"  [skip] AD PRS-CS-auto weights already exist in {ad_weights_dir}")
    else:
        if method_pri == "sbayesrc":
            phase2_sbayesrc(PRIMARY_TRAIT)
        else:
            phase2_prscs(PRIMARY_TRAIT)

    print("\n" + "=" * 76)
    print("  Phase 2 weights ready. Next steps:")
    print("=" * 76)
    if method_sec == "prscs":
        print(f"  Launch PRS-CS for {secondary} (~4 h):")
        print(f"     nohup bash {PHASE2}/run_prscs_{secondary.lower()}.sh \\")
        print(f"          > {PHASE2}/run_prscs_{secondary.lower()}.log 2>&1 &")
    else:
        print(f"  {secondary} SBayesRC weights already in place — proceed directly to:")
    print(f"  python3 {__file__} --step gene_burden --secondary {secondary}")
    print(f"  python3 {__file__} --step ldsc        --secondary {secondary}")
    print(f"  python3 {__file__} --step twas        --secondary {secondary}")


def main():
    ap = argparse.ArgumentParser(description="Hypertension × AD PRS pipeline")
    ap.add_argument("--step", default="full",
                    choices=["inspect", "harmonize", "liftover", "prscs", "sbayesrc",
                             "gene_burden", "twas", "ldsc", "full"])
    ap.add_argument("--secondary", default=SECONDARY_TRAIT,
                    choices=list(TRAIT_CONFIG.keys()))
    args = ap.parse_args()

    if args.step == "inspect":
        inspect_trait(PRIMARY_TRAIT); inspect_trait(args.secondary)
    elif args.step == "harmonize":
        phase1_harmonize(PRIMARY_TRAIT); phase1_harmonize(args.secondary)
    elif args.step == "liftover":
        phase1_liftover(PRIMARY_TRAIT); phase1_liftover(args.secondary)
    elif args.step == "prscs":
        phase2_prscs(PRIMARY_TRAIT); phase2_prscs(args.secondary)
    elif args.step == "sbayesrc":
        phase2_sbayesrc(args.secondary)
    elif args.step == "gene_burden":
        phase2_gene_burden(PRIMARY_TRAIT, args.secondary)
    elif args.step == "twas":
        phase3_twas(args.secondary)
    elif args.step == "ldsc":
        phase4_ldsc_rg(PRIMARY_TRAIT, args.secondary)
    elif args.step == "full":
        run_full(args.secondary)


if __name__ == "__main__":
    main()
