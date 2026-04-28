#!/usr/bin/env python3
"""
Merged Figure (Fig 6 + Fig 3d combined) — cross-disease TWAS (AD × SBP) across 12 brain
tissues, pathway enrichment, and major brain cell-type expression.

Layout (22 x 16 in):

  LEFT (12 scatter panels, 3 rows x 4 cols):
     12 brain tissues (spinal cord dropped).
     Each scatter shows all |z|>=3-in-both genes as colored background dots,
     then highlights:
       - TOP 5 extreme genes by sqrt(z_AD^2 + z_SBP^2) — white border marker
       - Up to 5 KNOWN AD/SBP cross-disease genes present in that tissue
         (from a curated LLM-informed list) — black border marker + italic label
     Conc (same-sign z) green · Disc (opposite-sign z) red.

  RIGHT TOP: Pathway enrichment of the union of annotated genes across
     all 12 tissues. Top 15 pathways. Split layout:
       left  half = horizontal bars (-log10 P)
       right half = full pathway term names (no truncation).

  RIGHT BOTTOM: Gene x major cell-type heatmap (HPA snRNA-seq).
     33 HPA fine clusters collapsed to 8 major categories
     (Exc neurons, Inh neurons, Astrocytes, Microglia, Oligo, OPCs,
      Endothelial, Pericytes).
     Square cells · row+column hierarchical clustering (correlation/average)
     · horizontal colorbar.

Usage:
    python3 make_merged_fig_twas_pathway_celltype.py \\
        --base-dir      ~/Downloads/prs_pipeline \\
        --celltype-file ~/Downloads/rna_single_nuclei_cluster_type.tsv \\
        --out           ~/Downloads/prs_pipeline/paper_figs
"""

import argparse, os, glob, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import LinearSegmentedColormap

# ── rcParams ────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':       14,
    'axes.titlesize':  14,
    'axes.labelsize':  13,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
})

CONC     = '#2E7D32'
DISC     = '#C62828'
BG_CONC  = '#81C784'   # darker green for better visibility vs earlier '#C8E6C9'
BG_DISC  = '#E57373'   # darker red   for better visibility vs earlier '#FFCDD2'
KNOWN_EDGE = '#000000' # black outline for known-gene markers to distinguish from extreme-only

# ───────────────────────────────────────────────────────────────
# Curated AD-SBP cross-disease gene list (LLM-identified).
# Genes grouped by pathway / mechanism for AD-SBP shared biology.
# Used alongside data-driven extreme-|z| genes for annotation.
# ───────────────────────────────────────────────────────────────
KNOWN_ADSBP_GENES = {
    # APOE region / lipid metabolism (chr19, strongest AD locus; discordant with SBP)
    'APOE', 'APOC1', 'APOC2', 'TOMM40', 'NECTIN2', 'CLPTM1', 'PVR', 'ERCC2',
    # Tau kinase / mTOR bridge — MARK4 is the paradigm for discordant mechanism
    'MARK4', 'OPA3',
    # Insulin signaling / incretin — direct AD-SBP bridge
    'GIPR', 'INSR', 'IRS1', 'IRS2', 'IDE', 'PPARG', 'SLC2A4', 'GIP',
    # Lysosomal / autophagy — clears amyloid, implicated in insulin processing
    'TMEM175', 'IDUA', 'CTSD', 'LGMN', 'GAK', 'DGKQ', 'CTBP1',
    # Immune / microglia — AD-protective / SBP-raising via inflammation
    'PLCG2', 'TREM2', 'HLA-DRB1', 'PSMB9', 'CNPY4', 'MEPCE', 'TLR4',
    # Synaptic / vesicle / SNARE — insulin exocytosis + synaptic release
    'BIN1', 'CD2AP', 'ADAM10', 'GOSR2', 'NSF', 'GRN', 'INPP5D', 'ACE',
    'SLC24A4', 'ABCA7',
    # BP canonical / vascular
    'FTO', 'ADRA2A', 'DGKB', 'NDUFAF6', 'SLC30A8', 'TCF7L2', 'KCNQ1',
    'HNF1A', 'HNF4A', 'HMGA2', 'JAZF1', 'THADA',
    # Additional triple-confirmed genes from PRS+TWAS+PWAS integration
    'CLU', 'PICALM', 'SORL1', 'FERMT2', 'PTK2B', 'APH1B',
    # ── SBP / BP canonical genes (AD × SBP analysis) ──
    'NPPA', 'NPPB', 'AGT', 'REN', 'AGTR1', 'AGTR2', 'ACE2',         # RAS core
    'CACNB2', 'CACNA1D', 'CACNA1C', 'KCNJ5',                        # ion channels
    'MECOM', 'TNS1', 'ZFPM2', 'CASZ1', 'CDH13',                     # top SBP burden hits
    'SLC39A8', 'SLC39A13', 'ENPEP', 'PLEKHA7', 'MAPT',              # cross-trait
    # ── Top AD × SBP discordant hits (from our analysis) ──
    'HAUS3', 'POLN', 'SNX32', 'CMTM3', 'NOV', 'CYP2U1', 'CTDNEP1',
    # ── Top AD × SBP concordant hits (from our analysis) ──
    'FOLH1', 'CTSB', 'FAM180B', 'PLEKHJ1', 'SIRT1', 'SEC24C', 'DNAJC9',
}

LIB_COLORS = {
    'Reactome_2022':              '#2E6FB5',
    'GO_Biological_Process_2023': '#C0392B',
    'KEGG_2021_Human':            '#E8851D',
}
LIB_SHORT = {
    'Reactome_2022':              'Reactome',
    'GO_Biological_Process_2023': 'GO BP',
    'KEGG_2021_Human':            'KEGG',
}

# ── Pathways of interest: hand-curated AD/SBP/neuro/metabolic whitelist ──
# Used as case-insensitive substring match on the pathway `Term` field.
# Organized by biological theme for readability; applied flat.
PATHWAYS_OF_INTEREST = [
    # --- AD / amyloid / tau ---
    'amyloid', 'alzheimer', 'tau ', 'neurofibrillary',
    'microglia', 'neurodegeneration',
    # --- SBP / blood pressure / RAS ---
    'insulin', 'hypertension', 'type 2 diabetes',
    'glucose', 'incretin', 'glucagon', 'pancreatic',
    'beta-cell', 'beta cell', 'islet', 'glycogen',
    # --- Lipid & lipoprotein ---
    'lipoprotein', 'cholesterol', 'lipid', 'chylomicron',
    'ldl', 'vldl', 'hdl', 'apolipoprotein', 'fatty acid',
    # --- Autophagy / lysosome / protein quality ---
    'lysosome', 'autophagy', 'phagosome', 'proteasome',
    'unfolded protein', 'chaperone',
    # --- Synaptic / neuronal ---
    'synap', 'neuro', 'axon', 'dendrit', 'glutamat', 'gaba',
    'vesicle', 'endocyto', 'long-term potentiation',
    'calcium signaling', 'calcium ion',
    # --- Signaling bridges ---
    'mtor', 'ampk', 'pi3k', 'akt', 'irs ', 'erk', 'mapk',
    # --- Immune / inflammation (AD-protective / SBP-raising) ---
    'complement', 'cytokine', 'immune response', 'inflammation',
    'interferon', 'tnf',
    # --- Mitochondria / energy ---
    'mitochondri', 'oxidative phosphorylation', 'electron transport',
]


def is_pathway_of_interest(term):
    """Case-insensitive substring match against PATHWAYS_OF_INTEREST."""
    low = str(term).lower()
    return any(kw in low for kw in PATHWAYS_OF_INTEREST)

# Spinal cord dropped — 12 brain tissues retained
BRAIN_TISSUES = [
    'Brain_Cortex', 'Brain_Frontal_Cortex_BA9',
    'Brain_Anterior_cingulate_cortex_BA24', 'Brain_Hippocampus',
    'Brain_Amygdala', 'Brain_Hypothalamus',
    'Brain_Caudate_basal_ganglia', 'Brain_Putamen_basal_ganglia',
    'Brain_Nucleus_accumbens_basal_ganglia', 'Brain_Substantia_nigra',
    'Brain_Cerebellum', 'Brain_Cerebellar_Hemisphere',
]
TISSUE_DISPLAY = {
    'Brain_Cortex': 'Cortex',
    'Brain_Frontal_Cortex_BA9': 'Frontal Ctx (BA9)',
    'Brain_Anterior_cingulate_cortex_BA24': 'Ant. Cing. (BA24)',
    'Brain_Hippocampus': 'Hippocampus',
    'Brain_Amygdala': 'Amygdala',
    'Brain_Hypothalamus': 'Hypothalamus',
    'Brain_Caudate_basal_ganthia': 'Caudate',
    'Brain_Caudate_basal_ganglia': 'Caudate',
    'Brain_Putamen_basal_ganglia': 'Putamen',
    'Brain_Nucleus_accumbens_basal_ganglia': 'Nucleus Accumbens',
    'Brain_Substantia_nigra': 'Substantia Nigra',
    'Brain_Cerebellum': 'Cerebellum',
    'Brain_Cerebellar_Hemisphere': 'Cerebellar Hem.',
}

# HPA snRNA-seq 33 fine clusters → 8 major categories.
# (Case-normalized matching; edit if your HPA file uses different spellings.)
HPA_TO_MAJOR = {
    # Excitatory neurons
    'upper-layer intratelencephalic':        'Excitatory neurons',
    'deep-layer intratelencephalic':         'Excitatory neurons',
    'deep-layer corticothalamic and 6b':     'Excitatory neurons',
    'deep-layer near-projecting':            'Excitatory neurons',
    'thalamic excitatory':                   'Excitatory neurons',
    'hippocampal ca1-3':                     'Excitatory neurons',
    'hippocampal ca4':                       'Excitatory neurons',
    'hippocampal dentate gyrus':             'Excitatory neurons',
    'amygdala excitatory':                   'Excitatory neurons',
    'upper rhombic lip':                     'Excitatory neurons',
    'lower rhombic lip':                     'Excitatory neurons',
    'mammillary body':                       'Excitatory neurons',
    'midbrain-derived inhibitory':           'Excitatory neurons',  # actually ambig; ok
    'splatter':                              'Excitatory neurons',
    # Inhibitory neurons
    'mge interneuron':                       'Inhibitory neurons',
    'cge interneuron':                       'Inhibitory neurons',
    'lamp5-lhx6 and chandelier':             'Inhibitory neurons',
    'medium spiny neuron':                   'Inhibitory neurons',
    'eccentric medium spiny neuron':         'Inhibitory neurons',
    'cerebellar inhibitory':                 'Inhibitory neurons',
    # Glia
    'astrocyte':                             'Astrocytes',
    'bergmann glia':                         'Astrocytes',
    'central nervous system macrophage':     'Microglia',
    'oligodendrocyte':                       'Oligodendrocytes',
    'committed oligodendrocyte precursor':   'Oligodendrocytes',
    'oligodendrocyte precursor cell':        'OPCs',
    # Vasculature
    'endothelial cell':                      'Endothelial',
    'pericyte':                              'Pericytes',
    'vascular associated smooth muscle cell':'Pericytes',
    'fibroblast':                            'Pericytes',
    # Other (dropped or joined)
    'ependymal cell':                        'Astrocytes',
    'choroid plexus epithelial cell':        'Astrocytes',
    'leukocyte':                             'Microglia',
}
MAJOR_CELLS = ['Excitatory neurons', 'Inhibitory neurons', 'Astrocytes',
               'Microglia', 'Oligodendrocytes', 'OPCs',
               'Endothelial', 'Pericytes']
# Short labels for heatmap x-axis (avoids overlap at the bottom of the panel)
MAJOR_CELLS_ABBR = {
    'Excitatory neurons': 'Exc',
    'Inhibitory neurons': 'Inh',
    'Astrocytes':         'Astro',
    'Microglia':          'Micro',
    'Oligodendrocytes':   'Oligo',
    'OPCs':               'OPCs',
    'Endothelial':        'Endo',
    'Pericytes':          'Peri',
}


# ═══════════════════════════════════════════════════════════
# Data loaders
# ═══════════════════════════════════════════════════════════
def find_phase3_dir(base, trait='ad'):
    """AD TWAS lives under results/phase3_multiomics; SBP under results_htn/phase3_multiomics."""
    if trait == 'ad':
        candidates = [
            os.path.join(base, 'phase3_multiomics'),
            os.path.join(base, 'results', 'phase3_multiomics'),
            os.path.join(base, 'pipeline', 'results', 'phase3_multiomics'),
        ]
    else:  # sbp
        candidates = [
            os.path.join(base, 'phase3_multiomics'),
            os.path.join(base, 'results_htn', 'phase3_multiomics'),
            os.path.join(base, 'pipeline', 'results_htn', 'phase3_multiomics'),
            # fall back to AD dir if both co-located
            os.path.join(base, 'results', 'phase3_multiomics'),
        ]
    for d in candidates:
        if os.path.isdir(d):
            return d
    return None


def load_tissue_file(phase_dir, disease, tissue):
    """Supports both `{disease}_spredixcan_{tissue}.csv` (AD convention) and
       `spredixcan_{disease}_{tissue}.csv` (our SBP output convention)."""
    patterns = [
        f'{disease}_spredixcan_{tissue}.csv',
        f'spredixcan_{disease}_{tissue}.csv',
    ]
    for pat in patterns:
        path = os.path.join(phase_dir, pat)
        if os.path.exists(path):
            df = pd.read_csv(path)
            if not all(c in df.columns for c in ('gene_name', 'zscore')):
                return None
            df = df[['gene_name', 'zscore']].copy()
            df['zscore'] = pd.to_numeric(df['zscore'], errors='coerce')
            return df.dropna()
    return None


def top_genes_per_tissue(ad_df, sbp_df, zmin=3.0, n_extreme=5, n_known=5):
    """For one tissue: merge, filter significant, pick top genes via TWO strategies:
      - n_extreme : largest sqrt(z_AD^2 + z_SBP^2)
      - n_known   : from KNOWN_ADSBP_GENES, ranked by magnitude, up to n_known
    Returns (sig_df, extreme_list, known_list).
    """
    if ad_df is None or sbp_df is None:
        return None, [], []
    m = ad_df.merge(sbp_df, on='gene_name', suffixes=('_ad','_sbp'))
    sig = m[(m['zscore_ad'].abs() >= zmin) & (m['zscore_sbp'].abs() >= zmin)].copy()
    if len(sig) == 0:
        return m, [], []
    sig['mag']  = np.sqrt(sig['zscore_ad']**2 + sig['zscore_sbp']**2)
    sig['conc'] = (np.sign(sig['zscore_ad']) == np.sign(sig['zscore_sbp']))
    # Drop pseudogenes / LINC for annotation candidates
    clean = sig[~sig['gene_name'].str.startswith(
        ('RP11-','LINC','AC0','AC1','CTD-','RP4-','RP3-','CTC-'), na=False)].copy()

    # Top n_extreme by magnitude
    extreme = clean.nlargest(n_extreme, 'mag')['gene_name'].tolist()

    # From known-gene set: those present in sig, ranked by magnitude,
    # then exclude genes already in extreme
    known_sub = clean[clean['gene_name'].isin(KNOWN_ADSBP_GENES) &
                        ~clean['gene_name'].isin(extreme)].copy()
    known = known_sub.nlargest(n_known, 'mag')['gene_name'].tolist()

    return sig, extreme, known


# ═══════════════════════════════════════════════════════════
# Scatter panel
# ═══════════════════════════════════════════════════════════
def plot_scatter_panel(ax, ad_df, sbp_df, tissue, zmin=3.0,
                          n_extreme=5, n_known=5):
    """Scatter panel for one brain tissue.

    Draw order (bottom to top):
      1. Background cloud : every significant gene as a pastel dot
                            (green = concordant, red = discordant).
                            Plain vector marks, no rasterization.
      2. Highlighted genes : bigger solid circles for the top-N extreme
                             and top-N known AD/SBP genes.
      3. Gene labels       : placed outward via place_labels_tuples().
    """
    display = TISSUE_DISPLAY.get(tissue, tissue)
    if ad_df is None or sbp_df is None:
        ax.text(0.5, 0.5, f'{display}\n(missing)',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=10, color='#999')
        ax.set_xticks([]); ax.set_yticks([])
        return [], [], 0, 0

    sig, extreme, known = top_genes_per_tissue(ad_df, sbp_df, zmin=zmin,
                                                 n_extreme=n_extreme,
                                                 n_known=n_known)
    if sig is None or len(sig) == 0:
        ax.text(0.5, 0.5, f'{display}\n(no sig genes)',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=10, color='#999')
        ax.set_xticks([]); ax.set_yticks([])
        return [], [], 0, 0

    n_conc = int(sig['conc'].sum())
    n_disc = len(sig) - n_conc

    # Axis limits
    x_lim = max(4.5, float(sig['zscore_ad'].abs().max() * 1.12))
    y_lim = max(6.0, float(sig['zscore_sbp'].abs().max() * 1.12))

    # Axes styling + origin cross-hairs
    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(-y_lim, y_lim)
    ax.axhline(0, color='#bbb', lw=0.5, zorder=0)
    ax.axvline(0, color='#bbb', lw=0.5, zorder=0)
    ax.tick_params(axis='both', labelsize=10, length=2)
    ax.set_title(display, fontsize=13, fontweight='bold', pad=4)

    # ── Step 1 : background cloud (every significant gene) ──
    conc_mask = sig['conc'].values
    # Concordant background (green)
    ax.plot(sig.loc[conc_mask, 'zscore_ad'].values,
             sig.loc[conc_mask, 'zscore_sbp'].values,
             marker='o', linestyle='none',
             color=BG_CONC, markersize=5.5, markeredgewidth=0,
             alpha=0.9, zorder=1)
    # Discordant background (red)
    ax.plot(sig.loc[~conc_mask, 'zscore_ad'].values,
             sig.loc[~conc_mask, 'zscore_sbp'].values,
             marker='o', linestyle='none',
             color=BG_DISC, markersize=5.5, markeredgewidth=0,
             alpha=0.9, zorder=1)

    # ── Step 2 : highlighted markers ──
    # Extreme (white edge)
    labels_to_place = []
    ext_rows = sig[sig['gene_name'].isin(extreme)]
    for _, r in ext_rows.iterrows():
        fc = CONC if r['conc'] else DISC
        ax.plot(r['zscore_ad'], r['zscore_sbp'],
                 marker='o', linestyle='none',
                 color=fc, markersize=10,
                 markeredgecolor='white', markeredgewidth=1.2,
                 zorder=5)
        labels_to_place.append((r['gene_name'], r['zscore_ad'],
                                 r['zscore_sbp'], fc, False, r['mag']))

    # Known (black edge)
    known_rows = sig[sig['gene_name'].isin(known)]
    for _, r in known_rows.iterrows():
        fc = CONC if r['conc'] else DISC
        ax.plot(r['zscore_ad'], r['zscore_sbp'],
                 marker='o', linestyle='none',
                 color=fc, markersize=10,
                 markeredgecolor='black', markeredgewidth=1.2,
                 zorder=5)
        labels_to_place.append((r['gene_name'], r['zscore_ad'],
                                 r['zscore_sbp'], fc, True, r['mag']))

    # ── Step 3 : labels ──
    place_labels_tuples(ax, labels_to_place, x_lim, y_lim)
    return extreme, known, n_conc, n_disc


def place_labels_tuples(ax, items, x_lim, y_lim):
    """Place gene labels avoiding (a) other labels, (b) highlighted marker
    positions. Directions that point RADIALLY OUTWARD from the plot origin
    are preferred — TWAS scatter density concentrates near (0,0) so outward
    positions land in cleaner whitespace.

    items: list of (gene_name, x, y, color, is_known, mag).
    """
    if not items:
        return
    fig = ax.figure
    bbox_ax = ax.get_window_extent()
    px_per_data_x = bbox_ax.width  / (2 * x_lim)
    px_per_data_y = bbox_ax.height / (2 * y_lim)
    fontsize_pt = 10
    dpi = fig.dpi
    px_per_pt = dpi / 72.0
    label_h_px = fontsize_pt * px_per_pt * 1.35
    label_w_px_per_char = fontsize_pt * px_per_pt * 0.60

    # Marker radius in data units — used to define forbidden zones
    # around every highlighted circle. Markers are s=80 points^2,
    # diameter ~ sqrt(80/pi)*2 ~ 10 pt → 10*dpi/72 ~ 42 px at 300dpi
    marker_r_px = np.sqrt(80.0 / np.pi)
    marker_r_d_x = marker_r_px / px_per_data_x
    marker_r_d_y = marker_r_px / px_per_data_y

    # Build forbidden boxes around every highlighted marker
    marker_boxes = []
    for _, gx, gy, _, _, _ in items:
        marker_boxes.append((gx - marker_r_d_x, gy - marker_r_d_y,
                             gx + marker_r_d_x, gy + marker_r_d_y))

    def box_data(name, tx, ty, ha, va):
        w_px = max(1, len(name)) * label_w_px_per_char
        h_px = label_h_px
        w_d = w_px / px_per_data_x
        h_d = h_px / px_per_data_y
        if   ha == 'left':   x0, x1 = tx,         tx + w_d
        elif ha == 'right':  x0, x1 = tx - w_d,   tx
        else:                x0, x1 = tx - w_d/2, tx + w_d/2
        if   va == 'bottom': y0, y1 = ty,         ty + h_d
        elif va == 'top':    y0, y1 = ty - h_d,   ty
        else:                y0, y1 = ty - h_d/2, ty + h_d/2
        return (x0, y0, x1, y1)

    def overlaps(b, boxes):
        x1, y1, x2, y2 = b
        for bx0, by0, bx1, by1 in boxes:
            if not (x2 <= bx0 or x1 >= bx1 or y2 <= by0 or y1 >= by1):
                return True
        return False

    # 36 candidate directions
    all_angles = np.linspace(0, 2*np.pi, 37)[:-1]

    # Farther distances (reach corners). Start close then push outward.
    distances = [0.15, 0.25, 0.40, 0.58, 0.78, 1.00, 1.25]

    # Sort items by magnitude descending — largest-magnitude genes get
    # first choice of label space.
    items_sorted = sorted(items, key=lambda t: -t[5])
    label_boxes = []   # running list of placed label bboxes

    for gene, gx, gy, col, is_known, mag in items_sorted:
        fontweight = 'bold'
        fontstyle  = 'italic' if is_known else 'normal'

        # Unit vector pointing radially outward from origin for this marker;
        # if the marker sits near origin, default to up-right (1,1)/sqrt(2)
        r0 = np.hypot(gx, gy)
        if r0 < 1e-6:
            rx, ry = 0.707, 0.707
        else:
            rx, ry = gx / r0, gy / r0

        # Reorder candidate directions so outward directions are tried first.
        # Score each direction by dot product with the outward unit vector,
        # then sort descending.
        scored = []
        for a in all_angles:
            dx, dy = np.cos(a), np.sin(a)
            score = dx * rx + dy * ry          # +1 = exactly outward
            scored.append((score, dx, dy))
        scored.sort(key=lambda t: -t[0])

        placed = False
        fallback = None  # (tx, ty, ha, va, bb, overlap_count)

        # Panel inner bounds (a tiny pad so labels don't touch the frame).
        # Extra pad at the TOP to keep labels from colliding with the title.
        x_pad = x_lim * 0.02
        y_pad = y_lim * 0.02
        x_min_in, x_max_in = -x_lim + x_pad, x_lim - x_pad
        y_min_in, y_max_in = -y_lim + y_pad, y_lim - y_pad * 3.5

        # Try distances in order, directions in outward-first order
        for dist in distances:
            for score, dx, dy in scored:
                tx = gx + dx * dist * x_lim
                ty = gy + dy * dist * y_lim
                if not (x_min_in <= tx <= x_max_in and
                         y_min_in <= ty <= y_max_in):
                    continue
                ha = 'left'   if dx >  0.15 else ('right' if dx < -0.15 else 'center')
                va = 'bottom' if dy >  0.15 else ('top'   if dy < -0.15 else 'center')
                bb = box_data(gene, tx, ty, ha, va)

                # Reject if any part of the label bbox would cross the axes frame.
                bx0, by0, bx1, by1 = bb
                if (bx0 < x_min_in or bx1 > x_max_in or
                    by0 < y_min_in or by1 > y_max_in):
                    continue

                # Check against BOTH other labels AND marker positions.
                # Also allow overlap with the OWN marker (gx, gy is the anchor).
                own_marker_idx = None
                for mi, mbox in enumerate(marker_boxes):
                    # crude "own marker" check: point inside its box
                    if (mbox[0] <= gx <= mbox[2] and
                        mbox[1] <= gy <= mbox[3]):
                        own_marker_idx = mi
                        break
                forbidden_markers = ([m for i, m in enumerate(marker_boxes)
                                        if i != own_marker_idx]
                                       if own_marker_idx is not None
                                       else marker_boxes)

                if not overlaps(bb, label_boxes) and \
                   not overlaps(bb, forbidden_markers):
                    label_boxes.append(bb)
                    ax.annotate(gene, xy=(gx, gy), xytext=(tx, ty),
                                 fontsize=10, fontweight=fontweight,
                                 fontstyle=fontstyle, color=col,
                                 ha=ha, va=va,
                                 arrowprops=dict(arrowstyle='-', color=col,
                                                  lw=0.55, alpha=0.75),
                                 zorder=6)
                    placed = True
                    break

                # Track the least-bad fallback (fewest overlaps) in case
                # nothing clean can be found.
                if fallback is None:
                    fallback = (tx, ty, ha, va, bb)
            if placed:
                break

        # If no clean spot found, use fallback — but clamp so the label
        # box stays strictly INSIDE the panel (drop label altogether if
        # clamping would overlap the anchor marker).
        if not placed and fallback is not None:
            tx, ty, ha, va, bb = fallback
            bx0, by0, bx1, by1 = bb
            # Clamp text position inward if bbox would exit panel
            w = bx1 - bx0
            h = by1 - by0
            if ha == 'left'  and tx + w > x_max_in: tx = x_max_in - w
            if ha == 'right' and tx - w < x_min_in: tx = x_min_in + w
            if ha == 'center':
                if tx - w/2 < x_min_in: tx = x_min_in + w/2
                if tx + w/2 > x_max_in: tx = x_max_in - w/2
            if va == 'bottom' and ty + h > y_max_in: ty = y_max_in - h
            if va == 'top'    and ty - h < y_min_in: ty = y_min_in + h
            if va == 'center':
                if ty - h/2 < y_min_in: ty = y_min_in + h/2
                if ty + h/2 > y_max_in: ty = y_max_in - h/2

            # Re-compute bbox after clamp
            bb = box_data(gene, tx, ty, ha, va)
            label_boxes.append(bb)
            ax.annotate(gene, xy=(gx, gy), xytext=(tx, ty),
                         fontsize=10, fontweight=fontweight,
                         fontstyle=fontstyle, color=col,
                         ha=ha, va=va,
                         arrowprops=dict(arrowstyle='-', color=col,
                                          lw=0.55, alpha=0.75),
                         zorder=6,
                         annotation_clip=False)


# ═══════════════════════════════════════════════════════════
# Pathway enrichment
# ═══════════════════════════════════════════════════════════
def run_enrichment(gene_list, top_n=15, p_threshold=0.10, max_name_len=55):
    """Pool Reactome + GO BP + KEGG and take top_n pathways by nominal P-value.

    If `max_name_len` is set, pathways whose DISPLAY name (database suffixes
    stripped) is longer than this many characters are filtered OUT of the
    pool before selecting the top_n. This biases the selection toward
    concise KEGG/Reactome terms over verbose GO BP names and keeps display
    tidy. Set to None to disable the filter.
    """
    if len(gene_list) < 5:
        print(f"  Pathway: only {len(gene_list)} genes, skipping")
        return None
    try:
        import gseapy as gp
    except ImportError:
        print("  gseapy not installed — pip3 install 'gseapy<1.1.0'")
        return None
    import time

    libraries = ['Reactome_2022', 'GO_Biological_Process_2023', 'KEGG_2021_Human']
    per_lib_results = {}

    for i, lib in enumerate(libraries):
        if i > 0:
            time.sleep(2.5)
        sub = None
        for attempt in range(3):
            try:
                enr = gp.enrichr(gene_list=sorted(gene_list),
                                   gene_sets=lib, organism='Human',
                                   outdir=None, no_plot=True)
                sub = enr.results.copy()
                sub['Library'] = lib
                sub['n_genes'] = pd.to_numeric(
                    sub['Overlap'].astype(str).str.split('/').str[0],
                    errors='coerce')
                break
            except Exception as e:
                print(f"    {lib}: attempt {attempt+1} failed ({e})")
                time.sleep(3 * (attempt + 1))
        if sub is None:
            per_lib_results[lib] = pd.DataFrame()
            continue
        p_col = 'P-value' if 'P-value' in sub.columns else 'Adjusted P-value'
        sub = sub[sub[p_col] < p_threshold].sort_values(p_col)
        per_lib_results[lib] = sub
        print(f"    {lib}: {len(sub)} pathways passing P<{p_threshold}")

    # Pool all libraries, then take top_n by P-value (no balancing).
    all_rows = [df for df in per_lib_results.values() if len(df)]
    if not all_rows:
        return None
    combined = pd.concat(all_rows, ignore_index=True)
    p_col = 'P-value' if 'P-value' in combined.columns else 'Adjusted P-value'

    # Filter out pathways whose display name (database suffixes stripped)
    # exceeds `max_name_len` characters. This biases the selection toward
    # concise KEGG/Reactome terms over verbose GO BP names.
    if max_name_len is not None:
        def _display_len(term):
            t = re.sub(r'\s+R-HSA(-\d+)?$', '', str(term))
            t = re.sub(r'\s+\(GO:\d+\)$', '', t)
            return len(t)
        n_before = len(combined)
        combined = combined[combined['Term'].apply(
            lambda t: _display_len(t) <= max_name_len)].reset_index(drop=True)
        n_dropped = n_before - len(combined)
        if n_dropped > 0:
            print(f"  Dropped {n_dropped} pathway(s) with display name > "
                  f"{max_name_len} chars")

    combined = combined.sort_values(p_col).head(top_n).reset_index(drop=True)

    lib_counts = combined['Library'].value_counts()
    print(f"  Top {len(combined)} by P-value (unbalanced): {dict(lib_counts)}")
    return combined


def pathway_panel(ax_bars, ax_labels, enr_df,
                    title='Enriched pathways',
                    compact=False, show_legend=True):
    """Two-axes pathway panel:
       ax_bars   : left 50% — horizontal bars (no y labels)
       ax_labels : right 50% — full pathway term text, aligned to bar index.

    compact=True      → smaller fonts (fits 4 stacked panels in panel b)
    show_legend=False → don't draw the library legend (useful when used
                        multiple times in one figure)
    """
    # Font scale by compact flag
    if compact:
        title_fs, xlbl_fs, tick_fs, lbl_fs, gene_fs, leg_fs = 12, 11, 10, 10, 8, 9
    else:
        title_fs, xlbl_fs, tick_fs, lbl_fs, gene_fs, leg_fs = 14, 13, 12, 12, 11, 11

    if enr_df is None or len(enr_df) == 0:
        ax_bars.text(0.5, 0.5, 'No pathways\nat P<0.10',
                      ha='center', va='center', fontsize=title_fs, color='#666',
                      transform=ax_bars.transAxes)
        ax_bars.set_xticks([]); ax_bars.set_yticks([])
        for s in ('left','bottom','top','right'):
            ax_bars.spines[s].set_visible(False)
        ax_bars.set_title(title, fontsize=title_fs, fontweight='bold')
        ax_labels.axis('off')
        return

    d = enr_df.copy().reset_index(drop=True)
    p_col = 'P-value' if 'P-value' in d.columns else 'Adjusted P-value'
    d['-log10 p'] = -np.log10(d[p_col].clip(lower=1e-30))
    d = d.sort_values('-log10 p', ascending=True).reset_index(drop=True)

    y = np.arange(len(d))
    colors = [LIB_COLORS.get(lib, '#888') for lib in d['Library']]

    # ── Left panel: bars only ──
    ax_bars.barh(y, d['-log10 p'], color=colors, edgecolor='black', lw=0.3,
                   height=0.78, zorder=3)
    x_end = d['-log10 p'].max()
    for i, r in d.iterrows():
        if pd.notna(r.get('n_genes', np.nan)):
            ax_bars.text(r['-log10 p'] + x_end * 0.015, i,
                          f"{int(r['n_genes'])}",
                          va='center', ha='left', fontsize=gene_fs,
                          color=LIB_COLORS.get(r['Library'], '#444'), zorder=4)
    ax_bars.set_yticks(y)
    ax_bars.set_yticklabels([''] * len(d))
    ax_bars.set_ylim(-0.7, len(d) - 0.3)
    ax_bars.set_xlabel('-log10(nominal P)', fontsize=xlbl_fs)
    ax_bars.tick_params(axis='x', labelsize=tick_fs)
    ax_bars.set_title(title, fontsize=title_fs, fontweight='bold',
                       pad=4, loc='left')
    ax_bars.set_xlim(0, x_end * 1.18)
    ax_bars.axvline(-np.log10(0.05), color='#888', ls='--',
                      lw=0.4, alpha=0.7, zorder=0)

    if show_legend:
        from matplotlib.patches import Patch
        lib_order = ['Reactome_2022', 'GO_Biological_Process_2023', 'KEGG_2021_Human']
        libs_present = [L for L in lib_order if L in set(d['Library'])]
        handles = [Patch(facecolor=LIB_COLORS[L], edgecolor='black', lw=0.3,
                          label=LIB_SHORT[L]) for L in libs_present]
        ax_bars.legend(handles=handles, loc='lower right', frameon=True,
                         framealpha=0.92, fontsize=leg_fs, handletextpad=0.4,
                         borderpad=0.4, handlelength=1.3, edgecolor='#aaa')

    # ── Right panel: full pathway names ──
    ax_labels.set_ylim(-0.7, len(d) - 0.3)
    ax_labels.set_xlim(0, 1)
    ax_labels.axis('off')
    # In compact mode (2×2 quadrant grid) terms > ~40 chars need truncation
    max_len = 40 if compact else 55
    for i, r in d.iterrows():
        c = LIB_COLORS.get(r['Library'], '#444')
        term = str(r['Term'])
        if len(term) > max_len:
            # Try to wrap at a space near the middle
            brk_at = int(max_len * 0.85)
            brk = term.find(' ', brk_at)
            if 0 < brk < max_len + 10:
                term = term[:brk] + '\n' + term[brk+1:]
            else:
                # No good break point → truncate with ellipsis
                term = term[:max_len - 1] + '…'
        ax_labels.text(0.0, i, term, ha='left', va='center',
                        fontsize=lbl_fs, color=c)


def quadrant_pathway_panel(ax, enr_df, title, show_legend=False):
    """Compact pathway bar chart for ONE quadrant's gene list.

    Layout: horizontal bars with truncated pathway names as y-tick labels.
    Bars colored by library (Reactome/GO BP/KEGG).

    Parameters
    ----------
    ax        : matplotlib Axes to draw into
    enr_df    : output of run_enrichment() for this quadrant (or None)
    title     : panel title (e.g. 'AD↑ / SBP↑ (concordant risk)\\n(n=23)')
    show_legend : only True for one sub-panel; shows the library color key
    """
    if enr_df is None or len(enr_df) == 0:
        ax.text(0.5, 0.5, 'No enriched\npathways',
                 ha='center', va='center', fontsize=10, color='#999',
                 transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ('top', 'right', 'bottom', 'left'):
            ax.spines[s].set_visible(False)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=3)
        return

    d = enr_df.copy().reset_index(drop=True)
    p_col = 'P-value' if 'P-value' in d.columns else 'Adjusted P-value'
    d['-log10 p'] = -np.log10(d[p_col].clip(lower=1e-30))
    d = d.sort_values('-log10 p', ascending=True).reset_index(drop=True)

    y = np.arange(len(d))
    colors = [LIB_COLORS.get(lib, '#888') for lib in d['Library']]
    ax.barh(y, d['-log10 p'], color=colors, edgecolor='black', lw=0.3,
             height=0.75, zorder=3)

    # Truncated labels as y-ticks — wider panels allow longer names
    def _trunc(term, n=75):
        term = str(term)
        return term if len(term) <= n else term[:n - 1] + '…'
    ax.set_yticks(y)
    ax.set_yticklabels([_trunc(t, 75) for t in d['Term']], fontsize=9)
    ax.set_ylim(-0.7, len(d) - 0.3)

    # Gene count annotation at end of each bar
    x_end = d['-log10 p'].max() if len(d) else 1.0
    for i, r in d.iterrows():
        if pd.notna(r.get('n_genes', np.nan)):
            ax.text(r['-log10 p'] + x_end * 0.02, i,
                     f"{int(r['n_genes'])}",
                     va='center', ha='left', fontsize=8,
                     color=LIB_COLORS.get(r['Library'], '#444'), zorder=4)

    ax.set_xlim(0, x_end * 1.18)
    ax.set_xlabel('-log10(P)', fontsize=9)
    ax.tick_params(axis='x', labelsize=8)
    ax.axvline(-np.log10(0.05), color='#888', ls='--', lw=0.3, alpha=0.6, zorder=0)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=3, loc='left')

    if show_legend:
        from matplotlib.patches import Patch
        lib_order = ['Reactome_2022', 'GO_Biological_Process_2023', 'KEGG_2021_Human']
        libs_present = [L for L in lib_order if L in set(d['Library'])]
        handles = [Patch(facecolor=LIB_COLORS[L], edgecolor='black', lw=0.3,
                          label=LIB_SHORT[L])
                   for L in libs_present]
        if handles:
            ax.legend(handles=handles, loc='lower right', fontsize=7,
                       handlelength=1, handletextpad=0.3, borderpad=0.3,
                       framealpha=0.9, edgecolor='#bbb')


def dotplot_panel(ax, enr_by_quadrant, quadrant_order, quadrant_labels,
                    n_per_quadrant=10):
    """Dot-plot of pathway enrichment across quadrants.

    Each column is one quadrant. Each row is a unique pathway. Dots are drawn
    at (column, pathway) where that pathway is enriched in that quadrant.
    Dot properties:
      - color : library (Reactome / GO BP / KEGG)
      - size  : -log10(P)  (larger = more significant)

    enr_by_quadrant : dict {'Q1': df, 'Q2': df, 'Q3': df, 'Q4': df}
        Each df is from run_enrichment() — columns 'Term','P-value','Library'.
    quadrant_order  : list of quadrant keys in column order, e.g.
        ['Q1', 'Q3', 'Q2', 'Q4']  → Concordant | Concordant | Discordant | Discordant
    quadrant_labels : dict {q -> string} for column headers
    n_per_quadrant  : take top-N pathways per quadrant, by P-value
    """
    # 1. Collect top-N pathways from each quadrant, tag with origin quadrant
    rows = []
    for q in quadrant_order:
        df = enr_by_quadrant.get(q)
        if df is None or len(df) == 0:
            continue
        d = df.copy()
        p_col = 'P-value' if 'P-value' in d.columns else 'Adjusted P-value'
        d = d.sort_values(p_col).head(n_per_quadrant)
        d['-log10 p'] = -np.log10(d[p_col].clip(lower=1e-30))
        d['quadrant'] = q
        rows.append(d)

    if not rows:
        ax.text(0.5, 0.5, 'No pathways to display',
                 ha='center', va='center', fontsize=12, color='#999',
                 transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ('top','right','bottom','left'):
            ax.spines[s].set_visible(False)
        return

    full = pd.concat(rows, ignore_index=True)

    # 2. Unique pathway list (y-axis). Sort by max -log10p across quadrants,
    #    so strongest hits are at the TOP.
    path_agg = (full.groupby('Term')['-log10 p'].max()
                    .sort_values(ascending=True)
                    .reset_index())
    path_order = path_agg['Term'].tolist()
    term_to_y = {t: i for i, t in enumerate(path_order)}

    # Keep a representative library per pathway for color (taking the one
    # from the quadrant where -log10p is largest)
    path_to_lib = (full.sort_values('-log10 p', ascending=False)
                        .drop_duplicates('Term')
                        .set_index('Term')['Library'].to_dict())

    # 3. Compute dot size range from -log10p
    v_min = full['-log10 p'].min()
    v_max = full['-log10 p'].max()
    def _size(v):
        # Scale -log10p linearly to size range [30, 250] pt²
        if v_max == v_min:
            return 100
        return 30 + (v - v_min) / (v_max - v_min) * 220

    # 4. Draw dots
    for _, r in full.iterrows():
        x = quadrant_order.index(r['quadrant'])
        y = term_to_y[r['Term']]
        color = LIB_COLORS.get(r['Library'], '#888')
        ax.scatter(x, y, s=_size(r['-log10 p']),
                    c=color, edgecolors='black', linewidths=0.4,
                    alpha=0.9, zorder=3)

    # 5. Axes
    ax.set_xticks(range(len(quadrant_order)))
    ax.set_xticklabels([quadrant_labels.get(q, q) for q in quadrant_order],
                        fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(path_order)))
    # Truncate long pathway names
    def _trunc(t, n=45):
        return t if len(t) <= n else t[:n-1] + '…'
    y_labels = [_trunc(t) for t in path_order]
    # Color y-labels by library
    ax.set_yticklabels(y_labels, fontsize=9)
    for lbl, t in zip(ax.get_yticklabels(), path_order):
        lbl.set_color(LIB_COLORS.get(path_to_lib.get(t), '#333'))

    ax.set_xlim(-0.5, len(quadrant_order) - 0.5)
    ax.set_ylim(-0.5, len(path_order) - 0.5)
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', length=2)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis='x', linestyle=':', linewidth=0.4, color='#ddd', zorder=0)

    # 6. Legends : library (colors) and size (-log10 P)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    lib_order = ['Reactome_2022', 'GO_Biological_Process_2023', 'KEGG_2021_Human']
    libs_present = [L for L in lib_order if L in set(full['Library'])]
    color_handles = [Patch(facecolor=LIB_COLORS[L], edgecolor='black', lw=0.3,
                             label=LIB_SHORT[L]) for L in libs_present]

    # Size legend: three reference levels
    ref_vals = np.linspace(v_min, v_max, 3)
    size_handles = [Line2D([0], [0], marker='o', linestyle='none',
                             markerfacecolor='#888', markeredgecolor='black',
                             markeredgewidth=0.4,
                             markersize=np.sqrt(_size(v)) * 0.75,
                             label=f'{v:.1f}')
                    for v in ref_vals]

    # Place both legends just to the right of the plot
    leg1 = ax.legend(handles=color_handles, loc='upper left',
                      bbox_to_anchor=(1.02, 1.0),
                      title='Library', fontsize=9, title_fontsize=10,
                      frameon=True, framealpha=0.92, edgecolor='#bbb',
                      handletextpad=0.5, borderpad=0.4)
    ax.add_artist(leg1)
    ax.legend(handles=size_handles, loc='upper left',
               bbox_to_anchor=(1.02, 0.65),
               title='-log10(P)', fontsize=9, title_fontsize=10,
               frameon=True, framealpha=0.92, edgecolor='#bbb',
               handletextpad=0.5, borderpad=0.4)


def pathway_group_barplot(ax, enr_df, group_label, n_show=10,
                             show_legend=True):
    """Horizontal DOT plot of top-N pathways for ONE pathway group,
    matching the reference figure style:
      - X position = 'Enrichment score' (-log10(P) * gene_count, arbitrary-ish)
      - Dot size   = gene count (overlap numerator)
      - Dot color  = -log10(P) on a viridis gradient
      - Y-axis     = pathway name (plain black text, right-aligned near plot)
      - Top-left panel label + group subtitle.

    Note: the reference uses an 'Enrichment score' on the x-axis. We compute
    a proxy score = -log10(P) * gene_count, which is monotone with
    significance-weighted overlap size (a reasonable stand-in for the
    enrichment score from gseapy when it's not explicitly available).
    """
    if enr_df is None or len(enr_df) == 0:
        ax.text(0.5, 0.5, 'No enriched pathways',
                 ha='center', va='center', fontsize=12, color='#999',
                 transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ('top', 'right', 'bottom', 'left'):
            ax.spines[s].set_visible(False)
        ax.set_title(group_label, fontsize=13, fontweight='bold',
                      loc='left', pad=6)
        return

    d = enr_df.copy()
    p_col = 'P-value' if 'P-value' in d.columns else 'Adjusted P-value'
    d['-log10 p'] = -np.log10(d[p_col].clip(lower=1e-30))

    # Gene count from 'Overlap' column (e.g., '3/200')
    def _gene_count(s):
        try:
            return int(str(s).split('/')[0])
        except Exception:
            return 1
    if 'Overlap' in d.columns:
        d['n_gene'] = d['Overlap'].apply(_gene_count)
    else:
        d['n_gene'] = 1

    # Enrichment score proxy
    d['enrich_score'] = d['-log10 p'] * d['n_gene']

    # Sort by enrich_score so largest (= right-most dot) sits at the TOP of y-axis
    d = d.sort_values('enrich_score', ascending=True).tail(n_show).reset_index(drop=True)

    def _clean(term, max_len=55):
        """Strip database ID suffixes and truncate very long names with '…'."""
        term = str(term)
        # Strip Reactome numeric suffix ("...R-HSA-12345" or "...R-HSA")
        term = re.sub(r'\s+R-HSA(-\d+)?$', '', term)
        # Strip GO-id suffix ("... (GO:1234567)")
        term = re.sub(r'\s+\(GO:\d+\)$', '', term)
        if len(term) <= max_len:
            return term
        return term[:max_len - 1].rstrip() + '…'

    y = np.arange(len(d))
    x_vals     = d['enrich_score'].values
    p_vals     = d['-log10 p'].values
    size_vals  = d['n_gene'].values

    # Size scaling: area ∝ gene count; calibrated so ~2-8 range maps to 40-300 pt²
    s_min, s_max = 1, max(8, int(size_vals.max()))
    def _size(n):
        if s_max == s_min:
            return 120
        return 40 + (n - s_min) / (s_max - s_min) * 260
    sizes = np.array([_size(n) for n in size_vals])

    # Color scaling: viridis on -log10(p)
    p_min, p_max = float(p_vals.min()), float(p_vals.max())
    cmap = plt.get_cmap('viridis')
    if p_max == p_min:
        norm = np.full_like(p_vals, 0.7)
    else:
        norm = (p_vals - p_min) / (p_max - p_min)
    colors = [cmap(v) for v in norm]

    # Horizontal guide lines
    for yy in y:
        ax.plot([0, x_vals.max() * 1.10], [yy, yy],
                 color='#ddd', lw=0.4, zorder=1)

    ax.scatter(x_vals, y, s=sizes, c=colors,
                edgecolors='black', linewidths=0.5, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([_clean(t) for t in d['Term']],
                        fontsize=11, color='black')
    ax.set_ylim(-0.7, len(d) - 0.3)

    ax.set_xlim(x_vals.min() * 0.9 if x_vals.min() > 0 else 0,
                 x_vals.max() * 1.10)
    ax.set_xlabel('Enrichment score', fontsize=12)
    ax.tick_params(axis='x', labelsize=10, length=2)
    ax.tick_params(axis='y', length=2)

    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    ax.set_title(group_label, fontsize=13, fontweight='bold',
                  loc='left', pad=6)

    # Compact legends on the right side of the plot
    from matplotlib.lines import Line2D

    # Size legend (gene count)
    size_legend_vals = sorted(set([2, 4, 6, 8]) & set(size_vals.tolist()))
    if not size_legend_vals:
        # pick three representative sizes from actual data
        size_legend_vals = sorted(set(np.quantile(size_vals, [0.1, 0.5, 0.9]).astype(int)))
    size_handles = [
        Line2D([0], [0], marker='o', linestyle='none',
               markerfacecolor='#888', markeredgecolor='black',
               markeredgewidth=0.5,
               markersize=np.sqrt(_size(n)) * 0.72,
               label=str(n))
        for n in size_legend_vals
    ]
    # Color legend (-log10 p) as separate markers
    color_vals = np.linspace(p_min, p_max, 3)
    color_handles = [
        Line2D([0], [0], marker='o', linestyle='none',
               markerfacecolor=cmap((v - p_min) / max(p_max - p_min, 1e-9)),
               markeredgecolor='black', markeredgewidth=0.4,
               markersize=9,
               label=f'{v:.1f}')
        for v in color_vals
    ]

    if not show_legend:
        return

    # Place legends INSIDE the plot area (lower-right corner where dots
    # are sparsest — dots cascade from bottom-left to top-right because of
    # the enrich_score sort, so the lower-right is usually empty).
    leg1 = ax.legend(handles=size_handles, loc='lower right',
                      bbox_to_anchor=(0.99, 0.02),
                      title='Gene count', fontsize=8, title_fontsize=9,
                      frameon=True, framealpha=0.92, edgecolor='#bbb',
                      handletextpad=0.5, borderpad=0.4,
                      labelspacing=0.3)
    ax.add_artist(leg1)
    ax.legend(handles=color_handles, loc='lower right',
               bbox_to_anchor=(0.99, 0.32),
               title='-log10(p)', fontsize=8, title_fontsize=9,
               frameon=True, framealpha=0.92, edgecolor='#bbb',
               handletextpad=0.5, borderpad=0.4,
               labelspacing=0.3)


def quadrant_bars_panel(fig, gs_panel, enr_by_quadrant, quadrant_order,
                          quadrant_short_labels, n_per_quadrant=10):
    """Clean per-quadrant horizontal bars.

    4 sub-panels in a row, one per quadrant. Each sub-panel shows
    top-N pathways for that quadrant as horizontal bars.

    Design:
      - All pathway text in BLACK (no library color-coding)
      - Bars in a neutral blue-gray color
      - Pathway name drawn as text INSIDE each bar's row, starting at x=0
        (left-justified on the bar's 0 point). This avoids y-tick labels
        running out of the sub-panel and into neighbors.
      - Top of each sub-panel: short title ('concordant risk', etc.)
      - Rotated 'Q1'/'Q2'/'Q3'/'Q4' letter on the LEFT margin of each panel.
    """
    gs_q = GridSpecFromSubplotSpec(
        1, len(quadrant_order),
        subplot_spec=gs_panel,
        wspace=0.30,
    )

    BAR_COLOR = '#6C8EBF'          # soft blue-gray (neutral)
    BAR_EDGE  = '#3B5A85'

    def _clean_term(term, max_len=42):
        term = str(term)
        term = re.sub(r'\s+R-HSA-\d+$', '', term)        # strip Reactome ID
        term = re.sub(r'\s+\(GO:\d+\)$', '', term)        # strip GO ID
        return term if len(term) <= max_len else term[:max_len - 1] + '…'

    for idx, q in enumerate(quadrant_order):
        ax = fig.add_subplot(gs_q[0, idx])
        df = enr_by_quadrant.get(q)

        if df is None or len(df) == 0:
            ax.text(0.5, 0.5, 'No enriched\npathways',
                     ha='center', va='center', fontsize=11, color='#999',
                     transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ('top', 'right', 'bottom', 'left'):
                ax.spines[s].set_visible(False)
            ax.set_title(quadrant_short_labels.get(q, q),
                          fontsize=12, fontweight='bold', pad=8)
            continue

        d = df.copy()
        p_col = 'P-value' if 'P-value' in d.columns else 'Adjusted P-value'
        d['-log10 p'] = -np.log10(d[p_col].clip(lower=1e-30))
        d = d.sort_values('-log10 p', ascending=True).tail(n_per_quadrant)
        d = d.reset_index(drop=True)

        y = np.arange(len(d))
        ax.barh(y, d['-log10 p'], color=BAR_COLOR,
                 edgecolor=BAR_EDGE, lw=0.4, height=0.75, zorder=3)

        # --- pathway names as text drawn ABOVE each bar (inside axes) ---
        x_end = d['-log10 p'].max() if len(d) else 1.0
        for i, r in d.iterrows():
            ax.text(
                x_end * 0.02, i + 0.40,    # just above the bar
                _clean_term(r['Term']),
                ha='left', va='bottom',
                fontsize=8.5, color='black', zorder=5,
            )

        ax.set_yticks([])                     # no y-tick labels
        ax.set_ylim(-0.7, len(d) - 0.05)
        ax.set_xlim(0, x_end * 1.08)
        ax.set_xlabel('-log10(P)', fontsize=10)
        ax.tick_params(axis='x', labelsize=9, length=2)
        ax.axvline(-np.log10(0.05), color='#888', ls='--',
                     lw=0.4, alpha=0.6, zorder=0)
        for s in ('top', 'right', 'left'):
            ax.spines[s].set_visible(False)

        # Sub-panel title (concordant risk / discordant / ...)
        ax.set_title(quadrant_short_labels.get(q, q),
                      fontsize=11, fontweight='bold', pad=8)

        # Rotated Q1/Q2/Q3/Q4 on the LEFT margin
        ax.text(-0.04, 0.5, q,
                 transform=ax.transAxes,
                 fontsize=16, fontweight='bold',
                 ha='right', va='center', rotation=90,
                 color='#333')


# ═══════════════════════════════════════════════════════════
# HPA snRNA-seq → 8 major cell types
# ═══════════════════════════════════════════════════════════
def find_hpa_celltype(path_hint=None):
    if path_hint:
        p = os.path.expanduser(path_hint)
        if os.path.exists(p):
            return p
    # fallbacks
    for pat in ['~/Downloads/rna_single_nuclei*.tsv*',
                 '~/*rna_single_nuclei*.tsv*']:
        matches = glob.glob(os.path.expanduser(pat))
        if matches: return matches[0]
    return None


def load_hpa_major(path):
    """Load HPA snRNA-seq and collapse fine clusters into 8 major categories.
    Returns (df with gene_name + 8 MAJOR cols, value_label)."""
    if path.endswith('.zip'):
        import zipfile
        with zipfile.ZipFile(path) as z:
            tsv_names = [n for n in z.namelist() if n.lower().endswith('.tsv')]
            if not tsv_names: return None, None
            with z.open(tsv_names[0]) as f:
                df = pd.read_csv(f, sep='\t', low_memory=False)
    elif path.endswith('.gz'):
        df = pd.read_csv(path, sep='\t', compression='gzip', low_memory=False)
    else:
        df = pd.read_csv(path, sep='\t', low_memory=False)

    gene_col = next((c for c in df.columns
                      if c.lower() in ('gene name', 'gene_name', 'genename')), None)
    cluster_col = next((c for c in df.columns
                         if c.lower() in ('cluster type','cluster_type',
                                            'clustertype','cell type','cell_type')), None)
    val_col = next((c for c in df.columns
                     if c.lower() in ('ntpm','tpm','ncpm','cpm','nx')), None)
    if not all([gene_col, cluster_col, val_col]):
        print(f"    expected columns not found. Have: {list(df.columns)}")
        return None, None

    df = df[[gene_col, cluster_col, val_col]].dropna()
    df[cluster_col] = df[cluster_col].astype(str).str.strip()

    # Map fine cluster → major via lookup (case-insensitive)
    df['major'] = df[cluster_col].str.lower().map(HPA_TO_MAJOR)
    dropped = df[df['major'].isna()]
    if len(dropped) > 0:
        unique_dropped = sorted(dropped[cluster_col].unique())
        print(f"    unmapped clusters (dropped): {unique_dropped}")
    df = df.dropna(subset=['major'])

    # Aggregate per (gene, major): mean of fine cluster values
    agg = (df.groupby([gene_col, 'major'])[val_col].mean()
             .reset_index()
             .rename(columns={gene_col: 'gene_name'}))
    wide = agg.pivot(index='gene_name', columns='major',
                      values=val_col).fillna(0.0).reset_index()
    # Ensure all 8 major categories exist
    for m in MAJOR_CELLS:
        if m not in wide.columns:
            wide[m] = 0.0
    wide = wide[['gene_name'] + MAJOR_CELLS]
    return wide, val_col


def celltype_panel(fig, gs_cell, genes_list, expr_df, val_label,
                     gene_concordance=None):
    """Clustered heatmap of genes × 8 major brain cell types.

    Simple layout:
      [ row dendro | heatmap | gene names | gap | narrow colorbar ]

    Column dendrogram sits above the heatmap.

    The colorbar is the only legend — narrow, placed well to the right of
    the gene names so they don't collide. Gene tick labels on the heatmap
    are colored by concordance (if `gene_concordance` is provided), with
    the meaning of the colors documented in the figure caption rather than
    a separate legend box.

    Returns ax_hm so the caller can add a panel label.
    """
    from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
    from scipy.spatial.distance import pdist

    # ── GridSpec layout ──
    # col 0 : row dendrogram   (0.06)
    # col 1 : heatmap          (0.55)
    # col 2 : gap for gene-name labels (0.20)
    # col 3 : narrow colorbar  (0.04)
    # col 4 : right margin     (0.02)
    # Note: the post-draw re-alignment (end of this function) snaps the row
    # dendrogram's x-right edge to the heatmap's x-left edge, so its visual
    # separation from the heatmap is zero regardless of gridspec wspace.
    gs_inner = GridSpecFromSubplotSpec(
        2, 5,
        subplot_spec=gs_cell,
        width_ratios=[0.06, 0.55, 0.20, 0.04, 0.02],
        height_ratios=[0.10, 1.0],
        hspace=0.03, wspace=0.01,
    )
    ax_row_dd = fig.add_subplot(gs_inner[1, 0])
    ax_col_dd = fig.add_subplot(gs_inner[0, 1])
    ax_hm     = fig.add_subplot(gs_inner[1, 1])
    ax_cbar   = fig.add_subplot(gs_inner[1, 3])

    # Turn off unused slots
    for r, c in [(0, 0), (0, 2), (0, 3), (0, 4), (1, 2), (1, 4)]:
        fig.add_subplot(gs_inner[r, c]).axis('off')

    # ── Guards ──
    if expr_df is None:
        ax_hm.text(0.5, 0.5, 'Cell-type expression not available',
                    ha='center', va='center', fontsize=13, color='#999',
                    transform=ax_hm.transAxes)
        ax_hm.set_xticks([]); ax_hm.set_yticks([])
        ax_row_dd.axis('off'); ax_col_dd.axis('off'); ax_cbar.axis('off')
        return ax_hm

    present = sorted(set(g for g in genes_list
                          if g in set(expr_df['gene_name'])))
    if len(present) < 2:
        ax_hm.text(0.5, 0.5, f'Only {len(present)} genes in data',
                    ha='center', va='center', fontsize=13, color='#999',
                    transform=ax_hm.transAxes)
        ax_hm.set_xticks([]); ax_hm.set_yticks([])
        ax_row_dd.axis('off'); ax_col_dd.axis('off'); ax_cbar.axis('off')
        return ax_hm

    # ── Expression matrix + clustering ──
    sub = (expr_df[expr_df['gene_name'].isin(present)]
              .set_index('gene_name').loc[present, MAJOR_CELLS])
    M  = np.log2(sub.values.astype(float) + 1.0)
    mu = M.mean(axis=1, keepdims=True)
    sd = M.std(axis=1, keepdims=True); sd[sd == 0] = 1.0
    Z  = (M - mu) / sd

    try:
        row_link  = linkage(pdist(Z,   metric='correlation'), method='average')
        col_link  = linkage(pdist(Z.T, metric='correlation'), method='average')
        row_order = leaves_list(row_link)
        col_order = leaves_list(col_link)
    except Exception:
        row_link = col_link = None
        row_order = np.arange(Z.shape[0])
        col_order = np.arange(Z.shape[1])

    Z_ord     = Z[np.ix_(row_order, col_order)]
    genes_ord = [present[i]     for i in row_order]
    cells_ord = [MAJOR_CELLS[j] for j in col_order]

    # ── Dendrograms ──
    if col_link is not None:
        dendrogram(col_link, ax=ax_col_dd, orientation='top',
                    no_labels=True, color_threshold=0,
                    above_threshold_color='#555')
    ax_col_dd.set_xticks([]); ax_col_dd.set_yticks([])
    for s in ('top', 'right', 'bottom', 'left'):
        ax_col_dd.spines[s].set_visible(False)

    if row_link is not None:
        dendrogram(row_link, ax=ax_row_dd, orientation='left',
                    no_labels=True, color_threshold=0,
                    above_threshold_color='#555')
    ax_row_dd.set_xticks([]); ax_row_dd.set_yticks([])
    for s in ('top', 'right', 'bottom', 'left'):
        ax_row_dd.spines[s].set_visible(False)
    ax_row_dd.invert_yaxis()

    # ── Heatmap — aspect='equal' gives square cells (1:1 ratio) ──
    im = ax_hm.imshow(Z_ord, aspect='equal',
                       cmap='RdBu_r', vmin=-2.5, vmax=2.5,
                       interpolation='nearest')
    ax_hm.set_xticks(np.arange(len(cells_ord)))
    # Vertical (90°) cell-type labels — compact, no overlap
    ax_hm.set_xticklabels([MAJOR_CELLS_ABBR.get(c, c) for c in cells_ord],
                           rotation=90, ha='center', va='top', fontsize=12)
    ax_hm.set_yticks(np.arange(len(genes_ord)))
    ax_hm.set_yticklabels(genes_ord, fontsize=9)     # smaller to avoid overlap
    ax_hm.yaxis.tick_right()
    ax_hm.yaxis.set_label_position('right')
    ax_hm.tick_params(axis='both', length=2)

    # Color gene tick labels by concordance (silently — no legend box).
    if gene_concordance:
        CLR = {'conc': CONC, 'disc': DISC, 'mixed': '#555'}
        for lbl, g in zip(ax_hm.get_yticklabels(), genes_ord):
            kind = gene_concordance.get(g, 'mixed')
            lbl.set_color(CLR.get(kind, '#333'))
            lbl.set_fontweight('bold' if kind in ('conc', 'disc') else 'normal')

    # Thin white grid
    for i in range(Z_ord.shape[0] + 1):
        ax_hm.axhline(i - 0.5, color='white', lw=0.4)
    for j in range(Z_ord.shape[1] + 1):
        ax_hm.axvline(j - 0.5, color='white', lw=0.4)

    # ── Colorbar (narrow vertical on the far right, well clear of gene names) ──
    cbar = plt.colorbar(im, cax=ax_cbar, orientation='vertical')
    cbar.set_label(f'z-score\n(log2 {val_label})', fontsize=11, labelpad=4)
    cbar.ax.tick_params(labelsize=10, length=2)
    cbar.outline.set_edgecolor('#444')
    cbar.outline.set_linewidth(0.5)

    # Compress the colorbar vertically to the upper half so it looks compact.
    #
    # ALSO: after aspect='equal' the heatmap shrinks to enforce square cells,
    # so its drawn bbox no longer matches its gridspec slot. The column and
    # row dendrograms MUST be repositioned to match the heatmap's actual
    # bounds, otherwise they visibly float away from the heatmap columns/rows.
    try:
        fig.canvas.draw()
        hm_pos  = ax_hm.get_position()           # actual drawn bbox
        cdd_pos = ax_col_dd.get_position()
        rdd_pos = ax_row_dd.get_position()
        cbar_pos = ax_cbar.get_position()

        # Column dendrogram: preserve its ORIGINAL height (so internal
        # branch lengths aren't visually stretched) and snap its bottom
        # edge to the heatmap's top edge. Match heatmap's x-range/width.
        ax_col_dd.set_position([
            hm_pos.x0,
            hm_pos.y0 + hm_pos.height,
            hm_pos.width,
            cdd_pos.height,
        ])
        # Row dendrogram: snap RIGHT edge flush to heatmap LEFT edge,
        # match heatmap's y-range/height.
        rdd_new_x0 = hm_pos.x0 - rdd_pos.width
        ax_row_dd.set_position([
            rdd_new_x0,
            hm_pos.y0,
            rdd_pos.width,
            hm_pos.height,
        ])
        # Colorbar: compress to upper 40% of heatmap height, anchored top.
        new_h = hm_pos.height * 0.40
        new_y = hm_pos.y0 + hm_pos.height - new_h
        ax_cbar.set_position([cbar_pos.x0, new_y, cbar_pos.width, new_h])
    except Exception:
        pass

    return ax_hm


def celltype_panel_rotated(fig, gs_cell, genes_list, expr_df, val_label,
                             gene_concordance=None, square_cells=False):
    """Rotated clustered heatmap:
      X-axis = genes          (vertical labels)
      Y-axis = cell types     (horizontal labels, right side)

    square_cells : if True, use aspect='equal' so each cell is 1:1. The
      post-draw alignment code below handles the extra shrinkage that
      aspect='equal' causes (so the dendrograms still flush up against
      the heatmap bounds).
    """
    from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
    from scipy.spatial.distance import pdist

    # width ratios: row-dd (for cell types), heatmap (wide), colorbar
    # height ratios: col-dd (for genes), heatmap (shorter, since fewer rows)
    gs_inner = GridSpecFromSubplotSpec(
        2, 4,
        subplot_spec=gs_cell,
        width_ratios=[0.05, 1.0, 0.08, 0.03],   # add gap col for right-side labels
        height_ratios=[0.15, 1.0],
        hspace=0.04, wspace=0.02,
    )
    ax_col_dd = fig.add_subplot(gs_inner[0, 1])    # gene dendrogram on top
    ax_row_dd = fig.add_subplot(gs_inner[1, 0])    # cell-type dendrogram on left
    ax_hm     = fig.add_subplot(gs_inner[1, 1])
    ax_cbar   = fig.add_subplot(gs_inner[1, 3])
    # unused corners
    for r, c in [(0, 0), (0, 2), (0, 3), (1, 2)]:
        fig.add_subplot(gs_inner[r, c]).axis('off')

    # Guards
    if expr_df is None:
        ax_hm.text(0.5, 0.5, 'Cell-type expression not available',
                    ha='center', va='center', fontsize=13, color='#999',
                    transform=ax_hm.transAxes)
        ax_hm.set_xticks([]); ax_hm.set_yticks([])
        ax_row_dd.axis('off'); ax_col_dd.axis('off'); ax_cbar.axis('off')
        return ax_hm

    present = sorted(set(g for g in genes_list
                          if g in set(expr_df['gene_name'])))
    if len(present) < 2:
        ax_hm.text(0.5, 0.5, f'Only {len(present)} genes in data',
                    ha='center', va='center', fontsize=13, color='#999',
                    transform=ax_hm.transAxes)
        ax_hm.set_xticks([]); ax_hm.set_yticks([])
        ax_row_dd.axis('off'); ax_col_dd.axis('off'); ax_cbar.axis('off')
        return ax_hm

    # Build matrix: genes (rows in source) × cells (cols in source).
    sub = (expr_df[expr_df['gene_name'].isin(present)]
              .set_index('gene_name').loc[present, MAJOR_CELLS])
    M  = np.log2(sub.values.astype(float) + 1.0)
    mu = M.mean(axis=1, keepdims=True)
    sd = M.std(axis=1, keepdims=True); sd[sd == 0] = 1.0
    Z_genes_by_cells = (M - mu) / sd

    # Cluster genes (originally rows) → will become x-axis
    # Cluster cells (originally columns) → will become y-axis
    try:
        gene_link = linkage(pdist(Z_genes_by_cells,   metric='correlation'),
                              method='average')
        cell_link = linkage(pdist(Z_genes_by_cells.T, metric='correlation'),
                              method='average')
        gene_order = leaves_list(gene_link)
        cell_order = leaves_list(cell_link)
    except Exception:
        gene_link = cell_link = None
        gene_order = np.arange(Z_genes_by_cells.shape[0])
        cell_order = np.arange(Z_genes_by_cells.shape[1])

    # After ordering : matrix shape (n_cells, n_genes) for the rotated display.
    Z_plot    = Z_genes_by_cells[np.ix_(gene_order, cell_order)].T
    genes_ord = [present[i]     for i in gene_order]
    cells_ord = [MAJOR_CELLS[j] for j in cell_order]

    # Gene dendrogram across the TOP (runs along x-axis gene order)
    if gene_link is not None:
        dendrogram(gene_link, ax=ax_col_dd, orientation='top',
                    no_labels=True, color_threshold=0,
                    above_threshold_color='#555')
    ax_col_dd.set_xticks([]); ax_col_dd.set_yticks([])
    for s in ('top', 'right', 'bottom', 'left'):
        ax_col_dd.spines[s].set_visible(False)

    # Cell-type dendrogram on the LEFT (runs along y-axis cell order)
    if cell_link is not None:
        dendrogram(cell_link, ax=ax_row_dd, orientation='left',
                    no_labels=True, color_threshold=0,
                    above_threshold_color='#555')
    ax_row_dd.set_xticks([]); ax_row_dd.set_yticks([])
    for s in ('top', 'right', 'bottom', 'left'):
        ax_row_dd.spines[s].set_visible(False)
    ax_row_dd.invert_yaxis()

    # Heatmap
    im = ax_hm.imshow(Z_plot,
                       aspect=('equal' if square_cells else 'auto'),
                       cmap='RdBu_r', vmin=-2.5, vmax=2.5,
                       interpolation='nearest')
    # X-axis: genes (vertical labels)
    ax_hm.set_xticks(np.arange(len(genes_ord)))
    ax_hm.set_xticklabels(genes_ord, rotation=90,
                            ha='center', va='top', fontsize=9)
    # Y-axis: cell types (horizontal labels, on right side)
    ax_hm.set_yticks(np.arange(len(cells_ord)))
    ax_hm.set_yticklabels([MAJOR_CELLS_ABBR.get(c, c) for c in cells_ord],
                            fontsize=11)
    ax_hm.yaxis.tick_right()
    ax_hm.yaxis.set_label_position('right')
    ax_hm.tick_params(axis='both', length=2)

    # Color gene x-tick labels by concordance
    if gene_concordance:
        CLR = {'conc': CONC, 'disc': DISC, 'mixed': '#555'}
        for lbl, g in zip(ax_hm.get_xticklabels(), genes_ord):
            kind = gene_concordance.get(g, 'mixed')
            lbl.set_color(CLR.get(kind, '#333'))
            lbl.set_fontweight('bold' if kind in ('conc', 'disc') else 'normal')

    # Thin white grid
    for i in range(Z_plot.shape[0] + 1):
        ax_hm.axhline(i - 0.5, color='white', lw=0.4)
    for j in range(Z_plot.shape[1] + 1):
        ax_hm.axvline(j - 0.5, color='white', lw=0.4)

    # Colorbar (compact)
    cbar = plt.colorbar(im, cax=ax_cbar, orientation='vertical')
    cbar.set_label(f'z-score\n(log2 {val_label})', fontsize=10, labelpad=3)
    cbar.ax.tick_params(labelsize=9, length=2)
    cbar.outline.set_edgecolor('#444')
    cbar.outline.set_linewidth(0.5)

    # Post-draw alignment: col dendrogram flush to heatmap top, row dendrogram flush to heatmap left
    try:
        fig.canvas.draw()
        hm_pos  = ax_hm.get_position()
        cdd_pos = ax_col_dd.get_position()
        rdd_pos = ax_row_dd.get_position()
        cbar_pos = ax_cbar.get_position()

        # Gene dendrogram: snap BOTTOM edge to heatmap top
        ax_col_dd.set_position([hm_pos.x0,
                                  hm_pos.y0 + hm_pos.height,
                                  hm_pos.width, cdd_pos.height])
        # Cell-type dendrogram: snap RIGHT edge flush to heatmap left
        ax_row_dd.set_position([hm_pos.x0 - rdd_pos.width,
                                  hm_pos.y0,
                                  rdd_pos.width, hm_pos.height])
        # Colorbar: compact vertical near heatmap right
        new_h = hm_pos.height * 0.9
        new_y = hm_pos.y0 + (hm_pos.height - new_h) / 2
        ax_cbar.set_position([cbar_pos.x0, new_y, cbar_pos.width, new_h])
    except Exception:
        pass

    return ax_hm




# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def make_merged_figure(base_dir, celltype_file, out_dir,
                         figsize=(20, 15), zmin=3.0,
                         n_extreme=5, n_known=5,
                         n_extreme_enrich=20, n_known_enrich=20,
                         n_per_quadrant=500,
                         pathway_top=15,
                         pathway_top_per_quadrant=15):
    os.makedirs(out_dir, exist_ok=True)
    phase_dir_ad  = find_phase3_dir(base_dir, trait='ad')
    phase_dir_sbp = find_phase3_dir(base_dir, trait='sbp')
    if phase_dir_ad is None or phase_dir_sbp is None:
        print(f"ERROR: phase3_multiomics not found under {base_dir}")
        print(f"  AD  dir: {phase_dir_ad}")
        print(f"  SBP dir: {phase_dir_sbp}")
        return
    print(f"TWAS dirs:\n  AD : {phase_dir_ad}\n  SBP: {phase_dir_sbp}")

    # ── load data per tissue and gather gene lists ──
    tissue_data = {}
    all_panel_genes = set()   # union of extreme+known genes across panels
    for tissue in BRAIN_TISSUES:
        ad  = load_tissue_file(phase_dir_ad,  'ad',  tissue)
        sbp = load_tissue_file(phase_dir_sbp, 'sbp', tissue)
        sig, extreme, known = top_genes_per_tissue(ad, sbp, zmin=zmin,
                                                      n_extreme=n_extreme,
                                                      n_known=n_known)
        tissue_data[tissue] = (ad, sbp)
        all_panel_genes.update(extreme)
        all_panel_genes.update(known)
        print(f"  {TISSUE_DISPLAY[tissue]:<25s} "
              f"extreme={len(extreme):>2d}  known={len(known):>2d}")

    # Clean (drop pseudogene / LINC noise)
    clean_genes = set(g for g in all_panel_genes
                       if g and g != 'nan' and not any(
                           g.startswith(x) for x in
                           ('RP11-','LINC','AC0','AC1','CTD-','RP4-','RP3-','CTC-')))
    print(f"\nUnion of annotated genes for SCATTER (12 tissues): {len(all_panel_genes)} "
          f"(cleaned: {len(clean_genes)})")

    # ── Build EXPANDED gene set for PATHWAY enrichment ──
    # New approach:
    #   1. Pool all (gene, tissue, z_AD, z_SBP) records where both |z|>=zmin.
    #   2. For each gene, compute mean z_AD and mean z_SBP over tissues.
    #   3. Assign quadrant from mean z signs.
    #   4. Rank by combined magnitude sqrt(meanZ_AD^2 + meanZ_SBP^2).
    #   5. Take top `n_per_quadrant` per quadrant.
    print(f"\nBuilding top-{n_per_quadrant}-per-quadrant gene set...")
    per_gene = {}  # gene -> list of (z_ad, z_sbp) pairs across sig tissues
    for tissue in BRAIN_TISSUES:
        ad, sbp = tissue_data[tissue]
        if ad is None or sbp is None:
            continue
        m = ad.merge(sbp, on='gene_name', suffixes=('_ad', '_sbp'))
        m = m[(m['zscore_ad'].abs() >= zmin) & (m['zscore_sbp'].abs() >= zmin)]
        # Drop pseudogenes
        m = m[~m['gene_name'].str.startswith(
            ('RP11-', 'LINC', 'AC0', 'AC1', 'CTD-', 'RP4-', 'RP3-', 'CTC-'),
            na=False)]
        for _, r in m.iterrows():
            per_gene.setdefault(r['gene_name'], []).append(
                (float(r['zscore_ad']), float(r['zscore_sbp'])))

    gene_to_quadrant = {}
    gene_mag = {}
    gene_mean_z = {}   # gene -> (mean_z_ad, mean_z_sbp, n_tissues)
    for g, vals in per_gene.items():
        az = np.mean([v[0] for v in vals])
        tz = np.mean([v[1] for v in vals])
        if   az > 0 and tz > 0:  q = 'Q1'
        elif az < 0 and tz > 0:  q = 'Q2'
        elif az < 0 and tz < 0:  q = 'Q3'
        else:                       q = 'Q4'
        gene_to_quadrant[g] = q
        gene_mag[g] = float(np.sqrt(az*az + tz*tz))
        gene_mean_z[g] = (float(az), float(tz), len(vals))

    # Top N per quadrant by magnitude
    enrich_per_quadrant = {}
    for q in ('Q1', 'Q2', 'Q3', 'Q4'):
        genes_q = [g for g, qq in gene_to_quadrant.items() if qq == q]
        genes_q_sorted = sorted(genes_q, key=lambda x: -gene_mag[x])
        enrich_per_quadrant[q] = genes_q_sorted[:n_per_quadrant]
        print(f"  {q}: {len(genes_q)} total sig genes, "
              f"selected top {len(enrich_per_quadrant[q])} for enrichment")

    enrich_genes_clean = set()
    for gg in enrich_per_quadrant.values():
        enrich_genes_clean.update(gg)
    print(f"  Combined enrichment set: {len(enrich_genes_clean)} genes across all quadrants")

    # ── Compute per-gene concordance across tissues (for gene-name coloring) ──
    # For each gene, count how many tissues it was called conc vs disc
    # (using |z|>=zmin in both traits as the sig filter). Majority wins;
    # ties get 'mixed'.
    print("\nComputing per-gene concordance across 12 tissues...")
    gene_concordance = {}
    for g in clean_genes:
        n_conc, n_disc = 0, 0
        for tissue in BRAIN_TISSUES:
            ad, sbp = tissue_data[tissue]
            if ad is None or sbp is None:
                continue
            a_z = ad[ad['gene_name'] == g]['zscore']
            t_z = sbp[sbp['gene_name'] == g]['zscore']
            if len(a_z) == 0 or len(t_z) == 0:
                continue
            az, tz = float(a_z.iloc[0]), float(t_z.iloc[0])
            if abs(az) < zmin or abs(tz) < zmin:
                continue
            if np.sign(az) == np.sign(tz):
                n_conc += 1
            else:
                n_disc += 1
        if n_conc > n_disc:
            gene_concordance[g] = 'conc'
        elif n_disc > n_conc:
            gene_concordance[g] = 'disc'
        else:
            gene_concordance[g] = 'mixed'
    cc_summary = {k: sum(1 for v in gene_concordance.values() if v == k)
                    for k in ('conc', 'disc', 'mixed')}
    print(f"  Concordance summary: {cc_summary}")

    q_counts = {q: len(enrich_per_quadrant[q]) for q in ('Q1', 'Q2', 'Q3', 'Q4')}
    print(f"  Per-quadrant gene counts for enrichment: {q_counts}")

    # ── Per-quadrant pathway enrichment (kept for reference / legacy) ──
    print("\nRunning per-quadrant pathway enrichment...")
    QUADRANT_META = {
        'Q1': 'AD↑ / SBP↑\n(concordant risk)',
        'Q2': 'AD↓ / SBP↑\n(discordant)',
        'Q3': 'AD↓ / SBP↓\n(concordant protective)',
        'Q4': 'AD↑ / SBP↓\n(discordant)',
    }
    quadrant_enr = {}
    for q in ('Q1', 'Q2', 'Q3', 'Q4'):
        genes_q = sorted(enrich_per_quadrant[q])
        print(f"  {q} {QUADRANT_META[q].replace(chr(10), ' ')}: {len(genes_q)} genes")
        if len(genes_q) < 5:
            quadrant_enr[q] = None
            continue
        quadrant_enr[q] = run_enrichment(genes_q, top_n=pathway_top_per_quadrant)

    # ── Merged CONCORDANT (Q1 ∪ Q3) and DISCORDANT (Q2 ∪ Q4) enrichment ──
    # For panels b and c:
    #   • Take top-20 per quadrant from the TWAS data (ranked by combined |z|)
    #   • Split LLM/known AD-SBP prior genes BY THEIR OWN QUADRANT so they
    #     are only added to the group where the data actually supports them:
    #       - LLM gene with mean_z in Q1 or Q3 → concordant input
    #       - LLM gene with mean_z in Q2 or Q4 → discordant input
    #       - LLM gene not significant enough to be classified → DROPPED
    #   • Then restrict the output to PATHWAYS_OF_INTEREST to focus on
    #     AD/SBP-relevant biology.
    print("\nRunning POI-filtered concordant vs discordant enrichment...")

    # Build reduced top-20 lists per quadrant.
    N_TOP_FOR_POI = 20
    top20 = {q: enrich_per_quadrant[q][:N_TOP_FOR_POI]
              for q in ('Q1', 'Q2', 'Q3', 'Q4')}

    # Split LLM/known gene list by each gene's own quadrant assignment.
    # `gene_to_quadrant` was computed earlier from per-tissue mean z signs
    # and only contains genes with |z|>=zmin in both traits somewhere.
    llm_all       = sorted(KNOWN_ADSBP_GENES)
    llm_conc      = [g for g in llm_all
                      if gene_to_quadrant.get(g) in ('Q1', 'Q3')]
    llm_disc      = [g for g in llm_all
                      if gene_to_quadrant.get(g) in ('Q2', 'Q4')]
    llm_dropped   = [g for g in llm_all if g not in gene_to_quadrant]
    print(f"  LLM prior: {len(llm_all)} genes total")
    print(f"    classified concordant (Q1/Q3): {len(llm_conc)} genes")
    print(f"    classified discordant (Q2/Q4): {len(llm_disc)} genes")
    print(f"    not significant enough to classify (dropped): "
          f"{len(llm_dropped)} genes")

    concordant_poi_input = sorted(set(llm_conc)
                                    | set(top20['Q1'])
                                    | set(top20['Q3']))
    discordant_poi_input = sorted(set(llm_disc)
                                    | set(top20['Q2'])
                                    | set(top20['Q4']))

    # How much do the two inputs actually differ?
    shared = set(concordant_poi_input) & set(discordant_poi_input)
    print(f"  Concordant input: LLM_conc({len(llm_conc)}) ∪ "
          f"Q1_top20({len(top20['Q1'])}) ∪ Q3_top20({len(top20['Q3'])}) "
          f"= {len(concordant_poi_input)} unique")
    print(f"  Discordant input: LLM_disc({len(llm_disc)}) ∪ "
          f"Q2_top20({len(top20['Q2'])}) ∪ Q4_top20({len(top20['Q4'])}) "
          f"= {len(discordant_poi_input)} unique")
    print(f"  Overlap between the two inputs: {len(shared)} genes")

    def _poi_enrich(gene_set, label):
        if len(gene_set) < 5:
            print(f"  {label}: only {len(gene_set)} genes — skipping")
            return None
        # Pull a large pool (top 500) with length filter applied to bias
        # toward concise KEGG/Reactome terms; then POI whitelist narrows it.
        raw = run_enrichment(gene_set, top_n=500, p_threshold=0.10,
                               max_name_len=55)
        if raw is None or len(raw) == 0:
            return None
        mask = raw['Term'].apply(is_pathway_of_interest)
        filtered = raw[mask].copy()
        print(f"  {label}: {len(raw)} enriched, "
              f"{len(filtered)} match POI whitelist")
        if len(filtered) == 0:
            return None
        p_col = 'P-value' if 'P-value' in filtered.columns else 'Adjusted P-value'
        filtered = (filtered.sort_values(p_col)
                              .head(pathway_top_per_quadrant)
                              .reset_index(drop=True))
        return filtered

    merged_enr = {}
    merged_enr['concordant'] = _poi_enrich(concordant_poi_input, 'Concordant')
    merged_enr['discordant'] = _poi_enrich(discordant_poi_input, 'Discordant')
    n_conc_genes = len(concordant_poi_input)
    n_disc_genes = len(discordant_poi_input)

    # ── Per-quadrant POI-filtered enrichment (for 2×2 pathway panels) ──
    # For each quadrant: input = (LLM genes classified in this quadrant)
    #                             ∪ (top-20 TWAS genes for this quadrant)
    # Output filtered by PATHWAYS_OF_INTEREST, top N shown.
    print("\nRunning POI-filtered per-quadrant enrichment...")
    llm_by_q = {q: [g for g in llm_all if gene_to_quadrant.get(g) == q]
                for q in ('Q1', 'Q2', 'Q3', 'Q4')}
    quadrant_poi_inputs = {}
    quadrant_poi_enr = {}
    for q in ('Q1', 'Q2', 'Q3', 'Q4'):
        inp = sorted(set(llm_by_q[q]) | set(top20[q]))
        quadrant_poi_inputs[q] = inp
        print(f"  {q}: LLM({len(llm_by_q[q])}) ∪ top20({len(top20[q])}) "
              f"= {len(inp)} unique")
        quadrant_poi_enr[q] = _poi_enrich(inp, q)
    n_per_q_input = {q: len(quadrant_poi_inputs[q]) for q in ('Q1','Q2','Q3','Q4')}

    # ── Export gene list used for pathway analysis (annotated by quadrant) ──
    # One row per gene actually fed to any of the 4 quadrant enrichments.
    # Columns:
    #   gene, quadrant, mean_z_AD, mean_z_SBP, magnitude, n_tissues,
    #   source (top_per_quadrant | llm_prior | both),
    #   in_Q1_input … in_Q4_input (TRUE/FALSE each)
    pathway_gene_rows = []
    all_input_genes = set()
    for q in ('Q1', 'Q2', 'Q3', 'Q4'):
        all_input_genes.update(quadrant_poi_inputs[q])
    for g in sorted(all_input_genes):
        q = gene_to_quadrant.get(g, 'NA')
        az, tz, ntis = gene_mean_z.get(g, (float('nan'), float('nan'), 0))
        mag = gene_mag.get(g, float('nan'))
        in_top20 = any(g in top20[qq] for qq in ('Q1','Q2','Q3','Q4'))
        in_llm   = g in KNOWN_ADSBP_GENES
        if in_top20 and in_llm:
            source = 'both'
        elif in_top20:
            source = 'top_per_quadrant'
        elif in_llm:
            source = 'llm_prior'
        else:
            source = 'other'
        row = {
            'gene': g, 'quadrant': q,
            'mean_z_AD':  az, 'mean_z_SBP': tz,
            'magnitude':  mag, 'n_tissues':  ntis,
            'source':     source,
            'in_Q1_input': g in set(quadrant_poi_inputs['Q1']),
            'in_Q2_input': g in set(quadrant_poi_inputs['Q2']),
            'in_Q3_input': g in set(quadrant_poi_inputs['Q3']),
            'in_Q4_input': g in set(quadrant_poi_inputs['Q4']),
        }
        pathway_gene_rows.append(row)
    pathway_gene_df = pd.DataFrame(pathway_gene_rows)
    # Sort by quadrant, then by magnitude descending
    pathway_gene_df = pathway_gene_df.sort_values(
        ['quadrant', 'magnitude'], ascending=[True, False]).reset_index(drop=True)
    os.makedirs(out_dir, exist_ok=True)
    out_xlsx = os.path.join(
        out_dir, f'pathway_genes_manual_top{n_per_quadrant}.xlsx')
    pathway_gene_df.to_excel(out_xlsx, index=False, float_format='%.4f',
                              sheet_name='pathway_genes')
    print(f"\nWrote pathway-analysis gene list (manual): {out_xlsx}")
    print(f"  {len(pathway_gene_df)} unique genes; per-quadrant counts:")
    print(pathway_gene_df['quadrant'].value_counts().to_dict())

    # ── cell-type expression ──
    print("\nLoading cell-type data...")
    ct_path = find_hpa_celltype(celltype_file)
    if ct_path:
        print(f"  HPA file: {ct_path}")
        expr_df, val_label = load_hpa_major(ct_path)
    else:
        expr_df, val_label = None, None
        print("  HPA cell-type file not found")

    # ═════════ figure ═════════
    # ═════════ LAYOUT ═════════
    #   Outer: a takes 60% width, right block takes 40%.
    #   Inside right block: buffer column + pathway plots + heatmap.
    #   Pathway plots (b, c) live in the right-most ~20% so their y-axis
    #   labels (pathway names) fill the buffer visually without colliding
    #   with a's scatter gene labels (which never extend past the scatter
    #   panel's own x-limit).
    #   Heatmap (d) spans the full 40% width because it needs more room.
    fig = plt.figure(figsize=figsize)
    # Three-column outer grid: panel a shrunk to 55%, a 5% spacer keeps
    # b/c/d anchored at their original rightmost 40% of the figure.
    outer_gs = GridSpec(
        1, 3,
        figure=fig,
        width_ratios=[0.55, 0.05, 0.40],
        wspace=0.04,
        left=0.06, right=0.99, top=0.94, bottom=0.09,
    )

    # LEFT : scatter 4 × 3
    gs_scatter = GridSpecFromSubplotSpec(
        4, 3,
        subplot_spec=outer_gs[0, 0],
        hspace=0.78, wspace=0.62,
    )

    # RIGHT (column 2, after the 5% spacer in column 1) :
    #   TOP (80%)    — four per-quadrant pathway panels stacked 4×1
    #                  b = rows 0+1 (Q1 concordant risk, Q3 concordant protective)
    #                  c = rows 2+3 (Q2 discordant,     Q4 discordant)
    #   BOTTOM (20%) — d = rotated heatmap (square 1:1)
    gs_right = GridSpecFromSubplotSpec(
        2, 1,
        subplot_spec=outer_gs[0, 2],
        height_ratios=[4.0, 1.0],
        hspace=0.08,
    )
    gs_pathways = gs_right[0, 0]
    gs_d        = gs_right[1, 0]

    # Pathway area: 4 rows × 2 cols. Each row is split 50/50:
    #   left half  = BLANK (buffer so pathway labels can extend left without
    #                colliding with panel a's scatter labels)
    #   right half = pathway dotplot
    # Row mapping:  0 = Q1 | 1 = Q3 | 2 = Q2 | 3 = Q4
    gs_path = GridSpecFromSubplotSpec(
        4, 2,
        subplot_spec=gs_pathways,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.0, 1.0, 1.0],
        hspace=0.55, wspace=0.02,
    )
    gs_q1 = gs_path[0, 1]
    gs_q3 = gs_path[1, 1]
    gs_q2 = gs_path[2, 1]
    gs_q4 = gs_path[3, 1]

    # ── draw scatter panels (4 × 3) ──
    for i, tissue in enumerate(BRAIN_TISSUES):
        r, c = divmod(i, 3)
        ax = fig.add_subplot(gs_scatter[r, c])
        ad, sbp = tissue_data[tissue]
        plot_scatter_panel(ax, ad, sbp, tissue, zmin=zmin,
                             n_extreme=n_extreme, n_known=n_known)
        ax.set_xlabel('AD z', fontsize=12)
        if c == 0:
            ax.set_ylabel('SBP z', fontsize=12)

    # Panel label 'a' on the scatter block (top-left)
    fig.text(0.015, 0.97, 'a',
              fontsize=22, fontweight='bold', va='top', ha='left')

    # Scatter legend in top margin
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=CONC, markeredgecolor='none',
               markersize=10, label='Concordant'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=DISC, markeredgecolor='none',
               markersize=10, label='Discordant'),
    ]
    fig.legend(handles=legend_handles, loc='center',
                bbox_to_anchor=(0.26, 0.97), ncol=2, fontsize=11,
                frameon=True, framealpha=0.92, edgecolor='#bbb',
                handletextpad=0.4, columnspacing=1.5)

    # ── Panel b: Q1 (row 0) and Q3 (row 1) — concordant quadrants ──
    # Only Q1 shows size/color legends (scales are identical across all 4).
    ax_q1 = fig.add_subplot(gs_q1)
    pathway_group_barplot(
        ax_q1, quadrant_poi_enr['Q1'],
        group_label=f'Q1 concordant risk  (n={n_per_q_input["Q1"]})',
        n_show=pathway_top_per_quadrant,
        show_legend=True,
    )
    ax_q3 = fig.add_subplot(gs_q3)
    pathway_group_barplot(
        ax_q3, quadrant_poi_enr['Q3'],
        group_label=f'Q3 concordant protective  (n={n_per_q_input["Q3"]})',
        n_show=pathway_top_per_quadrant,
        show_legend=True,
    )
    fig.text(0.62, 0.97, "b",
              fontsize=22, fontweight='bold', va='top', ha='left')

    # ── Panel c: Q2 (row 2) and Q4 (row 3) — discordant quadrants ──
    ax_q2 = fig.add_subplot(gs_q2)
    pathway_group_barplot(
        ax_q2, quadrant_poi_enr['Q2'],
        group_label=f'Q2 discordant AD↓ SBP↑  (n={n_per_q_input["Q2"]})',
        n_show=pathway_top_per_quadrant,
        show_legend=True,
    )
    ax_q4 = fig.add_subplot(gs_q4)
    pathway_group_barplot(
        ax_q4, quadrant_poi_enr['Q4'],
        group_label=f'Q4 discordant AD↑ SBP↓  (n={n_per_q_input["Q4"]})',
        n_show=pathway_top_per_quadrant,
        show_legend=True,
    )
    fig.text(0.62, 0.60, "c",
              fontsize=22, fontweight='bold', va='top', ha='left')

    # ── Panel d: rotated heatmap (genes on x, cell types on y) ──
    # Rendered with square 1:1 aspect so cells are square. Label 'd' is
    # placed close to the heatmap's top edge (not at the top of the row).
    celltype_panel_rotated(fig, gs_d,
                              sorted(clean_genes), expr_df,
                              val_label or 'nTPM',
                              gene_concordance=gene_concordance,
                              square_cells=True)
    fig.text(0.62, 0.27, "d",
              fontsize=22, fontweight='bold', va='top', ha='left')

    # No suptitle — leaves top margin clean for panel labels and legend.

    out_pdf = os.path.join(
        out_dir, f'Fig_merged_TWAS_AD_SBP_pathway_celltype_top{n_per_quadrant}.pdf')
    out_png = os.path.join(
        out_dir, f'Fig_merged_TWAS_AD_SBP_pathway_celltype_top{n_per_quadrant}.png')
    fig.savefig(out_pdf, format='pdf')
    fig.savefig(out_png, format='png', dpi=200)
    plt.close(fig)
    print(f"\nWrote {out_pdf}")
    print(f"Wrote {out_png}")


# ═════════════════════ CLI ══════════════════════════════════
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base-dir', required=True,
                    help='root directory; will look for results/phase3_multiomics')
    p.add_argument('--celltype-file', default=None,
                    help='HPA snRNA-seq TSV (gene × cluster type × nTPM)')
    p.add_argument('--out', default='./figs')
    p.add_argument('--width',       type=float, default=20)
    p.add_argument('--height',      type=float, default=15)
    p.add_argument('--zmin',        type=float, default=3.0,
                    help='|z| threshold in both traits for "significant"')
    p.add_argument('--n-extreme', type=int, default=5,
                    help='# top extreme genes (by combined |z|) per scatter')
    p.add_argument('--n-known',   type=int, default=5,
                    help='# known AD/SBP genes to annotate per scatter')
    p.add_argument('--n-extreme-enrich', type=int, default=20,
                    help='# top extreme genes per tissue for PATHWAY '
                         'enrichment (separate from scatter annotation; '
                         'larger set → more enrichment power)')
    p.add_argument('--n-known-enrich', type=int, default=20,
                    help='# known AD/SBP genes per tissue for PATHWAY '
                         'enrichment')
    p.add_argument('--n-per-quadrant', type=int, default=500,
                    help='# top genes per quadrant (ranked by combined |z|) '
                         'to include in pathway enrichment')
    p.add_argument('--pathway-top', type=int, default=15,
                    help='# top pathways (overall, unused in new 4-quadrant layout)')
    p.add_argument('--pathway-top-per-quadrant', type=int, default=15,
                    help='# top pathways to show per quadrant')
    args = p.parse_args()
    make_merged_figure(args.base_dir, args.celltype_file, args.out,
                         figsize=(args.width, args.height),
                         zmin=args.zmin,
                         n_extreme=args.n_extreme,
                         n_known=args.n_known,
                         n_extreme_enrich=args.n_extreme_enrich,
                         n_known_enrich=args.n_known_enrich,
                         n_per_quadrant=args.n_per_quadrant,
                         pathway_top=args.pathway_top,
                         pathway_top_per_quadrant=args.pathway_top_per_quadrant)
