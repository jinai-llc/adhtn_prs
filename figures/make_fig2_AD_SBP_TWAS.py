#!/usr/bin/env python3
"""
Figure 2 — AD × SBP cross-disease TWAS (13 brain tissues).

Layout (15×15 inch, 14 pt):
  Top   : 4×4 grid of per-tissue scatters (AD z-score × SBP z-score),
          quadrant-coloured; canonical discordant/concordant genes labeled.
          One slot (bottom-right) used for a global summary scatter + stats box.
  Bottom: gene × tissue heatmap strip showing cross-tissue consistency of
          top discordant (AD-SBP+ / AD+SBP-) and concordant genes;
          colour: concordant (green) vs discordant (red);
          intensity: min(|z_AD|, |z_SBP|).

Usage (Mac):
  python3 make_fig2_AD_SBP_TWAS.py \
      --twas-long  ~/Downloads/prs_pipeline/results_htn/phase3_multiomics/twas_cross_disease_all_tissues.tsv \
      --out        ./fig2_AD_SBP_TWAS.pdf
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# ─── style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 14,
    'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ─── palette ──────────────────────────────────────────────────────────────
CONC = '#2E7D32'   # green (AD+SBP+ or AD-SBP-)
DISC = '#C62828'   # red   (AD+SBP- or AD-SBP+)
AD_SBP_POS = '#2E7D32'
AD_SBP_NEG = '#1B5E20'
AD_POS_SBP_NEG = '#C62828'
AD_NEG_SBP_POS = '#AD1457'

TISSUE_ORDER = [
    "Brain_Cortex",
    "Brain_Frontal_Cortex_BA9",
    "Brain_Anterior_cingulate_cortex_BA24",
    "Brain_Hippocampus",
    "Brain_Amygdala",
    "Brain_Hypothalamus",
    "Brain_Caudate_basal_ganglia",
    "Brain_Putamen_basal_ganglia",
    "Brain_Nucleus_accumbens_basal_ganglia",
    "Brain_Substantia_nigra",
    "Brain_Cerebellum",
    "Brain_Cerebellar_Hemisphere",
    "Brain_Spinal_cord_cervical_c-1",
]

TISSUE_SHORT = {
    "Brain_Cortex": "Cortex",
    "Brain_Frontal_Cortex_BA9": "Frontal Ctx",
    "Brain_Anterior_cingulate_cortex_BA24": "ACC",
    "Brain_Hippocampus": "Hippocamp.",
    "Brain_Amygdala": "Amygdala",
    "Brain_Hypothalamus": "Hypothal.",
    "Brain_Caudate_basal_ganglia": "Caudate",
    "Brain_Putamen_basal_ganglia": "Putamen",
    "Brain_Nucleus_accumbens_basal_ganglia": "Nuc. accumb.",
    "Brain_Substantia_nigra": "S. nigra",
    "Brain_Cerebellum": "Cerebellum",
    "Brain_Cerebellar_Hemisphere": "Cereb. hem.",
    "Brain_Spinal_cord_cervical_c-1": "Spinal cord",
}

# Genes to label consistently across panels. Picked from the top cross-disease
# hits in results_htn/phase3_multiomics/twas_top_cross_hits.tsv (|z|>=5 in both).
LABEL_DISC = ["SLC2A4", "HAUS3", "POLN", "SNX32", "CMTM3", "NOV", "CYP2U1", "CTDNEP1"]
LABEL_CONC = ["FOLH1", "CTSB", "MAPT", "NPPA", "SLC39A13", "FAM180B", "PLEKHJ1"]


# ─── data loaders ────────────────────────────────────────────────────────
def load_long(path):
    if path and Path(path).exists():
        df = pd.read_csv(path, sep="\t")
        for c in ("AD_z", "SBP_z", "AD_p", "SBP_p"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["AD_z", "SBP_z"])
        return df
    print("[warn] no TWAS long table — using synthetic preview data")
    rng = np.random.RandomState(7)
    rows = []
    for t in TISSUE_ORDER:
        n = rng.randint(1000, 1700)
        ad_z  = rng.normal(0, 1.5, n)
        sbp_z = rng.normal(0, 1.3, n)
        # plant some discordant outliers
        k = rng.randint(3, 10)
        idx = rng.choice(n, k, replace=False)
        ad_z[idx]  = rng.choice([-1, 1], k) * rng.uniform(5, 9, k)
        sbp_z[idx] = -np.sign(ad_z[idx]) * rng.uniform(5, 10, k)
        for i in range(n):
            rows.append({
                "tissue": t, "gene_name": f"g{i:05d}",
                "AD_z": ad_z[i], "SBP_z": sbp_z[i],
                "AD_p": 1.0, "SBP_p": 1.0,
                "quadrant": (("AD+" if ad_z[i] > 0 else "AD-") +
                             ("SBP+" if sbp_z[i] > 0 else "SBP-"))
            })
    return pd.DataFrame(rows)


def quad_colour(adz, sbpz):
    if adz > 0 and sbpz > 0: return AD_SBP_POS
    if adz < 0 and sbpz < 0: return AD_SBP_NEG
    if adz > 0 and sbpz < 0: return AD_POS_SBP_NEG
    return AD_NEG_SBP_POS


# ─── per-tissue scatter ───────────────────────────────────────────────────
def draw_tissue_scatter(ax, df_t, tissue, z_thresh=4):
    if len(df_t) == 0:
        ax.text(0.5, 0.5, f'{TISSUE_SHORT[tissue]}\n(no data)', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color="#888")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ("top","right","bottom","left"): ax.spines[s].set_visible(False)
        return

    # All points, light grey
    ax.scatter(df_t["AD_z"], df_t["SBP_z"], s=3, c="#CFCFCF", alpha=0.55, rasterized=True)
    # Significant in both: colour by quadrant
    sig = df_t[(df_t["AD_z"].abs() >= z_thresh) & (df_t["SBP_z"].abs() >= z_thresh)]
    if len(sig):
        cols = [quad_colour(a, s) for a, s in zip(sig["AD_z"], sig["SBP_z"])]
        ax.scatter(sig["AD_z"], sig["SBP_z"], s=22, c=cols,
                    edgecolors='white', linewidths=0.5, zorder=3)

    # Reference lines
    mx = max(8, np.ceil(df_t[["AD_z","SBP_z"]].abs().max().max()))
    ax.axhline(0, color="#AAA", lw=0.4, zorder=1)
    ax.axvline(0, color="#AAA", lw=0.4, zorder=1)
    ax.axhline( z_thresh, color="#D0D0D0", lw=0.4, ls='--', zorder=1)
    ax.axhline(-z_thresh, color="#D0D0D0", lw=0.4, ls='--', zorder=1)
    ax.axvline( z_thresh, color="#D0D0D0", lw=0.4, ls='--', zorder=1)
    ax.axvline(-z_thresh, color="#D0D0D0", lw=0.4, ls='--', zorder=1)
    ax.set_xlim(-mx, mx); ax.set_ylim(-mx, mx)

    # Gene labels (canonical set, if present in this tissue)
    labeled = set()
    for g in LABEL_DISC + LABEL_CONC:
        row = sig[sig["gene_name"] == g]
        if len(row):
            r = row.iloc[0]
            col = DISC if g in LABEL_DISC else CONC
            ax.annotate(g, xy=(r["AD_z"], r["SBP_z"]),
                         xytext=(4, 4), textcoords='offset points',
                         fontsize=9, fontweight='bold', color=col)
            labeled.add(g)

    ax.set_title(TISSUE_SHORT[tissue], fontsize=12, pad=3)
    ax.tick_params(axis='both', labelsize=9)


def draw_summary_panel(ax, df, z_thresh=4):
    """Single pooled AD×SBP scatter across tissues + stats box."""
    ax.scatter(df["AD_z"], df["SBP_z"], s=2, c="#D0D0D0", alpha=0.4, rasterized=True)
    sig = df[(df["AD_z"].abs() >= z_thresh) & (df["SBP_z"].abs() >= z_thresh)]
    if len(sig):
        cols = [quad_colour(a, s) for a, s in zip(sig["AD_z"], sig["SBP_z"])]
        ax.scatter(sig["AD_z"], sig["SBP_z"], s=20, c=cols,
                    edgecolors='white', linewidths=0.5, zorder=3)

    mx = max(10, np.ceil(df[["AD_z","SBP_z"]].abs().max().max()))
    ax.axhline(0, color="#AAA", lw=0.4); ax.axvline(0, color="#AAA", lw=0.4)
    for ln in (-z_thresh, z_thresh):
        ax.axhline(ln, color="#D0D0D0", lw=0.4, ls='--')
        ax.axvline(ln, color="#D0D0D0", lw=0.4, ls='--')
    ax.set_xlim(-mx, mx); ax.set_ylim(-mx, mx)
    ax.set_xlabel("AD z-score")
    ax.set_ylabel("SBP z-score")
    ax.set_title("All 13 tissues combined", fontsize=12, pad=3)

    # Stats box at each |z| threshold
    lines = []
    for zc in [2, 3, 4, 5]:
        s = df[(df["AD_z"].abs() >= zc) & (df["SBP_z"].abs() >= zc)]
        n = len(s)
        if n:
            c = (s["AD_z"] * s["SBP_z"] > 0).sum()
            lines.append(f"|z|≥{zc}  n={n:>5,}  conc {100*c/n:>4.1f}%")
        else:
            lines.append(f"|z|≥{zc}  n=0")
    box = "\n".join(lines)
    ax.text(0.02, 0.98, box, transform=ax.transAxes, va='top', ha='left',
            fontsize=9, family='monospace',
            bbox=dict(facecolor='white', edgecolor='#BBB', boxstyle='round,pad=0.4'))


# ─── heatmap strip ────────────────────────────────────────────────────────
def build_heatmap_matrix(df, genes, tissues, value="min_abs_z"):
    """Rows = genes, cols = tissues. Value = min(|z_AD|, |z_SBP|) or 0.
       Sign: +1 if concordant, -1 if discordant, 0 if below thresh/missing.
    """
    mat = np.zeros((len(genes), len(tissues)))
    for i, g in enumerate(genes):
        sub = df[df["gene_name"] == g]
        for j, t in enumerate(tissues):
            r = sub[sub["tissue"] == t]
            if len(r) == 0:
                mat[i, j] = 0.0; continue
            ad = float(r["AD_z"].iloc[0]); sbp = float(r["SBP_z"].iloc[0])
            mag = min(abs(ad), abs(sbp))
            sign = 1 if ad * sbp > 0 else -1
            mat[i, j] = sign * mag
    return mat


def draw_heatmap(ax, df):
    # Label list — use those actually present
    present = set(df["gene_name"].unique())
    disc_rows = [g for g in LABEL_DISC if g in present]
    conc_rows = [g for g in LABEL_CONC if g in present]
    genes = conc_rows + disc_rows
    if not genes:
        ax.text(0.5, 0.5, "No canonical top-hit genes present in TWAS",
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off'); return

    mat = build_heatmap_matrix(df, genes, TISSUE_ORDER)

    # Diverging colormap: negative = red (discordant), positive = green (concordant)
    vmax = max(np.abs(mat).max(), 5)
    cmap = LinearSegmentedColormap.from_list(
        'ccd', [(0.0, DISC), (0.5, '#FFFFFF'), (1.0, CONC)], N=256)
    im = ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')

    ax.set_xticks(np.arange(len(TISSUE_ORDER)))
    ax.set_xticklabels([TISSUE_SHORT[t] for t in TISSUE_ORDER], rotation=60,
                       ha='right', fontsize=10)
    ax.set_yticks(np.arange(len(genes)))
    ax.set_yticklabels(genes, fontsize=10,
                       color='#333')
    # Color gene labels by conc (green) vs disc (red) based on list origin
    for tick, g in zip(ax.get_yticklabels(), genes):
        tick.set_color(CONC if g in conc_rows else DISC)
        tick.set_fontweight('bold')
    ax.set_title("Cross-tissue consistency of top cross-disease genes "
                 "(colour: concordant/discordant, intensity: min(|z_AD|,|z_SBP|))",
                 fontsize=12, pad=6)

    # Grid lines between cells
    for i in range(mat.shape[0] + 1):
        ax.axhline(i - 0.5, color='white', lw=1)
    for j in range(mat.shape[1] + 1):
        ax.axvline(j - 0.5, color='white', lw=1)

    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.01)
    cbar.set_label("signed min |z|  (–: discordant, +: concordant)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)


# ─── main orchestrator ───────────────────────────────────────────────────
def build_figure(df, out_path):
    fig = plt.figure(figsize=(15, 15))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1.35, 1.0], hspace=0.35)

    # Top: 4×4 grid (16 slots; 13 tissues + 1 summary + 2 empty)
    gs_top = outer[0].subgridspec(4, 4, hspace=0.55, wspace=0.30)

    # Draw tissue scatters
    for i, t in enumerate(TISSUE_ORDER):
        r, c = i // 4, i % 4
        ax = fig.add_subplot(gs_top[r, c])
        df_t = df[df["tissue"] == t]
        draw_tissue_scatter(ax, df_t, t, z_thresh=4)
        if r == 3 and c == 0:
            ax.set_xlabel("AD z", fontsize=11)
            ax.set_ylabel("SBP z", fontsize=11)

    # Slot (3, 1) empty label / blank for spacing
    # Slot (3, 2) : summary pooled scatter
    ax_sum = fig.add_subplot(gs_top[3, 2])
    draw_summary_panel(ax_sum, df, z_thresh=4)

    # Slot (3, 3): legend
    ax_leg = fig.add_subplot(gs_top[3, 3])
    ax_leg.axis('off')
    handles = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=AD_SBP_POS,
               markersize=10, label='AD+ SBP+  concordant'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=AD_SBP_NEG,
               markersize=10, label='AD- SBP-  concordant'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=AD_POS_SBP_NEG,
               markersize=10, label='AD+ SBP-  discordant'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=AD_NEG_SBP_POS,
               markersize=10, label='AD- SBP+  discordant'),
    ]
    ax_leg.legend(handles=handles, loc='center left', frameon=False,
                  fontsize=11, title='Quadrant (|z|≥4 in both)',
                  title_fontsize=12)

    # Bottom: gene × tissue heatmap strip
    ax_hm = fig.add_subplot(outer[1])
    draw_heatmap(ax_hm, df)

    # Global panel labels
    fig.text(0.02, 0.97, 'a', fontsize=22, fontweight='bold')
    fig.text(0.02, 0.45, 'b', fontsize=22, fontweight='bold')

    plt.savefig(out_path)
    print(f"saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--twas-long", default="",
                    help="TSV from twas_cross_disease.py (all tissues × genes)")
    ap.add_argument("--out", default="./fig2_AD_SBP_TWAS.pdf")
    args = ap.parse_args()

    df = load_long(args.twas_long)
    print(f"TWAS rows     : {len(df):,}")
    print(f"tissues       : {df['tissue'].nunique()}")
    print(f"unique genes  : {df['gene_name'].nunique()}")

    # Quick stats for the |z|>=5 headline
    for zc in [3, 4, 5]:
        s = df[(df["AD_z"].abs() >= zc) & (df["SBP_z"].abs() >= zc)]
        n = len(s)
        if n:
            c = (s["AD_z"] * s["SBP_z"] > 0).sum()
            print(f"  |z|>={zc} in both: n={n:>5,}  concordant {100*c/n:>4.1f}%")

    build_figure(df, args.out)


if __name__ == "__main__":
    main()
