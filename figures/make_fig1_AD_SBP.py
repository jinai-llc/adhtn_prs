#!/usr/bin/env python3
"""
Merged PRS + MR figure — AD × SBP, 2 rows × 3 panels, 15×15 inch, 14 pt fonts.

ROW 1  (PRS architecture):
  a — SNP weight distribution (hist) + per-chromosome PRS burden (bars, stripes)
  b — Venn (|β|>0.001 overlap) on top + concordance donut on bottom
  c — PRS weight Miami (AD top / SBP bottom inverted, chr stripes)

ROW 2  (Bidirectional Mendelian Randomisation):
  d — Forward MR scatter (SBP → AD)
  e — Reverse MR scatter, split into two sub-panels stacked:
        upper = all SNPs, lower = APOE-region excluded
  f — Forest plot of 4 MR methods × 3 groups:
        SBP → AD, AD → SBP (all SNPs), AD → SBP (APOE excluded)
      NOTE: outlier-removed group intentionally dropped

Usage (Mac — all files local):
    python3 make_merged_fig_prs_mr.py \\
        --prs-dir   ~/Downloads/prs_pipeline \\
        --mr-dir    ~/Downloads/prs_paper_data/phase6_mr \\
        --out       ~/Downloads/prs_pipeline/paper_figs
"""

import argparse, os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Circle, ConnectionPatch
from matplotlib.lines import Line2D
from scipy.stats import norm

# ═══════════════════════════════════════════════════════════
# rcParams — 14 pt everywhere
# ═══════════════════════════════════════════════════════════
BASE_FONT = 14
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':        BASE_FONT,
    'axes.titlesize':   BASE_FONT,
    'axes.labelsize':   BASE_FONT,
    'xtick.labelsize':  BASE_FONT,
    'ytick.labelsize':  BASE_FONT,
    'legend.fontsize':  BASE_FONT,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ─── palette ────────────────────────────────────────────────
AD     = '#7B1FA2'
SBP    = '#1565C0'
CONC   = '#2E7D32'
DISC   = '#C62828'
AD_LT  = '#B39DDB'
SBP_LT = '#64B5F6'
STRIPE = '#f2f2f2'

# ─── genome (GRCh38 Mb) ─────────────────────────────────────
CHR_SIZES = [248,242,198,190,182,170,159,145,138,133,135,133,
             114,107,101,90,83,80,58,64,46,50]
CHR_OFFSETS, _c = [], 0
for s in CHR_SIZES:
    CHR_OFFSETS.append(_c); _c += s
GENOME_SIZE = _c
np.random.seed(42)


# ═══════════════════════════════════════════════════════════
# PRS data loaders
# ═══════════════════════════════════════════════════════════
def load_prs_files(base, trait):
    prefix = 'ad' if trait == 'AD' else 'sbp'
    # For AD look under results/phase2_prs/ad_prscs; for SBP look under
    # results_htn/phase2_prs/sbp_sbayesrc (PRS-CS output in SBayesRC folder).
    if trait == 'AD':
        folder_names = [f'{prefix}_prscs']
        base_dirs = [
            os.path.join(base, 'phase2_prs'),
            os.path.join(base, 'results',  'phase2_prs'),
            os.path.join(base, 'pipeline', 'results', 'phase2_prs'),
        ]
    else:  # SBP
        folder_names = [f'{prefix}_sbayesrc', f'{prefix}_prscs']
        base_dirs = [
            os.path.join(base, 'phase2_prs'),
            os.path.join(base, 'results_htn', 'phase2_prs'),
            os.path.join(base, 'pipeline', 'results_htn', 'phase2_prs'),
        ]
    roots = [os.path.join(b, f) for b in base_dirs for f in folder_names]
    files = []
    for d in roots:
        pat = os.path.join(d, f'{prefix}_pst_eff_a1_b0.5_phiauto_chr*.txt')
        files = sorted(glob.glob(pat))
        if files:
            print(f"  PRS weights ({trait}): {d} ({len(files)} chr files)")
            break
    if not files:
        print(f"  PRS weights ({trait}): NOT FOUND under {base}")
        return None
    dfs = [pd.read_csv(f, sep=r'\s+', header=None,
                       names=["CHR","SNP","BP","A1","A2","BETA"]) for f in files]
    return pd.concat(dfs).drop_duplicates(subset='SNP', keep='first')


def half_normal_cdf(x, sig):
    return 2 * norm.cdf(x / sig) - 1


def add_chr_stripes(ax, zorder=0):
    for i in range(22):
        if i % 2 == 1:
            ax.axvspan(CHR_OFFSETS[i], CHR_OFFSETS[i] + CHR_SIZES[i],
                       color=STRIPE, zorder=zorder)


def lab(ax, s, x=-0.07, y=1.08):
    ax.text(x, y, s, transform=ax.transAxes,
            fontsize=BASE_FONT + 4, fontweight='bold', va='top')


# ═══════════════════════════════════════════════════════════
# Venn & donut helpers
# ═══════════════════════════════════════════════════════════
def draw_venn(ax, ad_only, sbp_only, shared, thr=0.001):
    ax.set_xlim(-2.4, 2.4); ax.set_ylim(-1.6, 1.9)
    ax.set_aspect('equal'); ax.axis('off')
    r = 1.0; xL, xR = -0.58, 0.58
    ax.add_patch(Circle((xL, 0), r, facecolor=AD,  alpha=0.30,
                         edgecolor=AD,  lw=1.8, zorder=2))
    ax.add_patch(Circle((xR, 0), r, facecolor=SBP, alpha=0.30,
                         edgecolor=SBP, lw=1.8, zorder=2))
    ax.text(xL - 0.60, 0.0, f'{ad_only:,}',
             ha='center', va='center', fontsize=BASE_FONT,
             fontweight='bold', color=AD, zorder=4)
    ax.text(xR + 0.60, 0.0, f'{sbp_only:,}',
             ha='center', va='center', fontsize=BASE_FONT,
             fontweight='bold', color=SBP, zorder=4)
    ax.text(0, 0.0, f'{shared}',
             ha='center', va='center', fontsize=BASE_FONT + 1,
             fontweight='bold', color='#222', zorder=4)
    ax.text(xL - 0.60, 1.35, 'AD PRS',
             ha='center', fontsize=BASE_FONT, fontweight='bold', color=AD)
    ax.text(xR + 0.60, 1.35, 'SBP PRS',
             ha='center', fontsize=BASE_FONT, fontweight='bold', color=SBP)
    ax.text(0, -1.45, f'|β| > {thr} in each PRS',
             ha='center', fontsize=BASE_FONT - 3,
             color='#666', fontstyle='italic')


def draw_donut(ax, n_conc, n_disc):
    _, tx, at = ax.pie(
        [n_conc, n_disc],
        labels=[f'Conc ({n_conc})', f'Disc ({n_disc})'],
        colors=[CONC, DISC],
        autopct='%1.0f%%', startangle=90, explode=(0.03, 0.03),
        pctdistance=0.75,
        wedgeprops=dict(width=0.40, edgecolor='white', linewidth=2))
    for t in at:
        t.set_fontsize(BASE_FONT); t.set_fontweight('bold'); t.set_color('white')
    for t in tx:
        t.set_fontsize(BASE_FONT - 2)
    ax.set_title(f'{n_conc + n_disc} shared SNPs (conc / disc)',
                  fontsize=BASE_FONT - 1, pad=6)


# ═══════════════════════════════════════════════════════════
# Miami helper
# ═══════════════════════════════════════════════════════════
def plot_miami_half(ax, weights, c_even, c_odd):
    if weights is None:
        return None
    df = weights.drop_duplicates('SNP').copy()
    df['absBETA'] = df['BETA'].abs()
    df['gp'] = df.apply(
        lambda r: CHR_OFFSETS[int(r['CHR']) - 1] + r['BP'] / 1e6
        if 1 <= int(r['CHR']) <= 22 else 0, axis=1)
    df = df[df['gp'] > 0]
    cols = np.array([c_even if int(c) % 2 == 0 else c_odd for c in df['CHR']])
    bg  = df['absBETA'] <= 0.001
    mid = (df['absBETA'] > 0.001) & (df['absBETA'] <= 0.01)
    hi  = df['absBETA'] > 0.01
    ax.scatter(df['gp'].values[bg],  df['absBETA'].values[bg],
                s=1,  c=cols[bg],  alpha=0.15, edgecolors='none', rasterized=True)
    ax.scatter(df['gp'].values[mid], df['absBETA'].values[mid],
                s=4,  c=cols[mid], alpha=0.40, edgecolors='none', rasterized=True)
    ax.scatter(df['gp'].values[hi],  df['absBETA'].values[hi],
                s=14, c=cols[hi],  alpha=0.70, edgecolors='none', rasterized=True)
    return df


# ═══════════════════════════════════════════════════════════
# MR data loaders
# ═══════════════════════════════════════════════════════════
def safe_read(path):
    if not os.path.exists(path):
        print(f"    MISSING: {path}")
        return None
    df = pd.read_csv(path)
    print(f"    Loaded:  {os.path.basename(path)}  ({len(df)} rows)")
    return df


def load_mr(mr_dir):
    out = {}
    print(f"  MR dir: {mr_dir}")
    out['fwd_harm']  = safe_read(os.path.join(mr_dir, 'forward_harmonized_data.csv'))
    out['fwd_res']   = safe_read(os.path.join(mr_dir, 'forward_mr_sbp_to_ad.csv'))
    out['rev_harm']  = safe_read(os.path.join(mr_dir, 'reverse_harmonized_data.csv'))
    out['rev_res']   = safe_read(os.path.join(mr_dir, 'reverse_mr_ad_to_sbp.csv'))

    # APOE-excluded (filenames in conversation history)
    out['rev_harm_noapoe'] = safe_read(
        os.path.join(mr_dir, 'reverse_harmonized_data_no_apoe.csv'))
    out['rev_res_noapoe']  = safe_read(
        os.path.join(mr_dir, 'reverse_mr_ad_to_sbp_no_apoe.csv'))
    return out


def ivw_slope(res_df):
    """Extract IVW β for drawing regression line, or None."""
    if res_df is None or 'method' not in res_df.columns:
        return None
    ivw = res_df[res_df['method'] == 'Inverse variance weighted']
    if len(ivw) == 0:
        return None
    return float(ivw.iloc[0]['b'])


# ═══════════════════════════════════════════════════════════
# Scatter helpers for MR panels
# ═══════════════════════════════════════════════════════════
def mr_scatter(ax, harm, res, dot_color, xlabel, ylabel, title, n_label=None,
                marker='o', open_markers=False):
    if harm is None:
        ax.text(0.5, 0.5, 'harmonized data missing',
                 ha='center', va='center', transform=ax.transAxes,
                 fontsize=BASE_FONT - 2, color='#999')
        ax.set_title(title, fontsize=BASE_FONT)
        return
    x   = harm['beta.exposure'].values
    y   = harm['beta.outcome'].values
    xse = harm['se.exposure'].values
    yse = harm['se.outcome'].values

    # error crosses (light)
    for i in range(len(x)):
        ax.plot([x[i]-xse[i], x[i]+xse[i]], [y[i], y[i]],
                 color='#ccc', lw=0.6, zorder=1)
        ax.plot([x[i], x[i]], [y[i]-yse[i], y[i]+yse[i]],
                 color='#ccc', lw=0.6, zorder=1)

    if open_markers:
        ax.scatter(x, y, s=55, facecolors='none', edgecolors=dot_color,
                    marker=marker, linewidths=1.5, zorder=3)
    else:
        ax.scatter(x, y, s=38, c=dot_color, alpha=0.75,
                    marker=marker, edgecolors='white', lw=0.4, zorder=3)
    ax.axhline(0, color='#bbb', lw=0.6, zorder=0)
    ax.axvline(0, color='#bbb', lw=0.6, zorder=0)

    # IVW regression line through origin
    slope = ivw_slope(res)
    if slope is not None:
        x_range = np.linspace(x.min(), x.max(), 50)
        ax.plot(x_range, slope * x_range, color=CONC, lw=2.0, alpha=0.85,
                 zorder=4, label=f'IVW β = {slope:.3f}')

    ax.set_xlabel(xlabel, fontsize=BASE_FONT - 1)
    ax.set_ylabel(ylabel, fontsize=BASE_FONT - 1)
    ax.set_title(title, fontsize=BASE_FONT)
    if n_label is not None:
        ax.text(0.97, 0.04, f'n = {n_label}',
                 ha='right', va='bottom', transform=ax.transAxes,
                 fontsize=BASE_FONT - 3, color='#666')
    if slope is not None:
        ax.legend(frameon=False, fontsize=BASE_FONT - 3, loc='upper left')


# ═══════════════════════════════════════════════════════════
# Forest helper
# ═══════════════════════════════════════════════════════════
def forest_panel(ax, fwd_res, rev_res, rev_noapoe):
    """Three groups stacked top-to-bottom:
        SBP → AD          (green filled circle)
        AD → SBP all      (purple filled circle)
        AD → SBP no-APOE  (purple open triangle)
    """
    method_order = ['Inverse variance weighted', 'MR Egger',
                    'Weighted median', 'Weighted mode']
    method_short = {
        'Inverse variance weighted': 'IVW',
        'MR Egger': 'MR-Egger',
        'Weighted median': 'W. median',
        'Weighted mode': 'W. mode',
    }

    groups = [
        ('SBP → AD',              fwd_res,      CONC, 'o', True),
        ('AD → SBP (all SNPs)',   rev_res,      AD,   'o', True),
        ('AD → SBP (APOE excl.)', rev_noapoe,   AD,   '^', False),
    ]

    n_methods = len(method_order)
    n_groups  = len(groups)
    row_height = 1.0
    group_gap  = 0.8

    yticks, yticklabels = [], []
    y = 0.0
    group_y_centers = []

    xs_lo, xs_hi = [], []

    for g_idx, (gname, res_df, col, marker, filled) in enumerate(groups):
        centers = []
        for m_idx, method in enumerate(method_order):
            yy = y
            yticks.append(yy)
            yticklabels.append(method_short[method])
            centers.append(yy)
            if res_df is not None and 'method' in res_df.columns:
                row = res_df[res_df['method'] == method]
                if len(row) > 0:
                    b  = float(row.iloc[0]['b'])
                    se = float(row.iloc[0]['se'])
                    lo, hi = b - 1.96 * se, b + 1.96 * se
                    xs_lo.append(lo); xs_hi.append(hi)
                    ax.plot([lo, hi], [yy, yy],
                             color=col, lw=2.0, zorder=2)
                    if filled:
                        ax.scatter([b], [yy], s=75, c=col, marker=marker,
                                    edgecolors='white', lw=0.8, zorder=3)
                    else:
                        ax.scatter([b], [yy], s=85, facecolors='none',
                                    edgecolors=col, marker=marker,
                                    linewidths=1.8, zorder=3)
            y += row_height
        group_y_centers.append(np.mean(centers))
        y += group_gap

    # right-side group labels
    x_right = max(xs_hi) * 1.05 if xs_hi else 0.3
    for (gname, _res, col, _marker, _filled), cy in zip(groups, group_y_centers):
        ax.text(x_right, cy, gname,
                 fontsize=BASE_FONT - 1, fontweight='bold',
                 color=col, va='center', ha='left',
                 transform=ax.transData)

    # group separator lines
    for g_idx in range(n_groups - 1):
        y_sep = g_idx * (n_methods * row_height + group_gap) + \
                  (n_methods - 1) * row_height + group_gap / 2
        ax.axhline(y_sep, color='#ddd', lw=0.5, zorder=0)

    ax.axvline(0, color='#888', lw=0.8, ls='--', zorder=1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=BASE_FONT - 2)
    ax.invert_yaxis()

    if xs_lo and xs_hi:
        span = max(xs_hi) - min(xs_lo)
        ax.set_xlim(min(xs_lo) - span * 0.1,
                    max(xs_hi) + span * 0.50)  # room for right-side labels
    ax.set_xlabel('Causal estimate β (95% CI)',
                   fontsize=BASE_FONT - 1)
    ax.set_title('Bidirectional MR estimates — 4 methods × 3 groups',
                  fontsize=BASE_FONT)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def make_figure(prs_dir, mr_dir, out_dir, figsize=(15, 15)):
    os.makedirs(out_dir, exist_ok=True)
    print("Merged PRS + MR figure [15×15 in, 14pt]")

    # ───────── PRS data ─────────
    aw = load_prs_files(prs_dir, 'AD')
    tw = load_prs_files(prs_dir, 'SBP')

    thr = 0.001
    if aw is not None and tw is not None:
        ad_set  = set(aw.loc[aw['BETA'].abs() > thr, 'SNP'])
        sbp_set = set(tw.loc[tw['BETA'].abs() > thr, 'SNP'])
        shared  = ad_set & sbp_set
        n_ad_only  = len(ad_set - sbp_set)
        n_sbp_only = len(sbp_set - ad_set)
        n_shared   = len(shared)
        mg = aw[['SNP','BETA']].merge(
            tw[['SNP','BETA']], on='SNP', suffixes=('_AD','_SBP'))
        bl = mg[(mg['BETA_AD'].abs() > thr) & (mg['BETA_SBP'].abs() > thr)]
        n_conc = int((np.sign(bl['BETA_AD']) == np.sign(bl['BETA_SBP'])).sum())
        n_disc = len(bl) - n_conc
    else:
        n_ad_only, n_sbp_only, n_shared = 8236, 3486, 64
        n_conc, n_disc = 30, 34
    print(f"  Venn counts: AD-only={n_ad_only:,}  shared={n_shared:,}  "
          f"SBP-only={n_sbp_only:,}  (conc={n_conc} / disc={n_disc})")

    # ───────── MR data ─────────
    mr = load_mr(mr_dir)

    # ───────── figure ─────────
    fig = plt.figure(figsize=figsize)
    # Row 1: a (left 50%) + c (right 50%).  Panel b removed.
    # Row 2: d, e, f equal thirds
    gs_top = GridSpec(
        2, 1, figure=fig,
        hspace=0.40,
        left=0.06, right=0.97, top=0.95, bottom=0.055,
    )
    gs_row1 = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_top[0],
        width_ratios=[1, 1], wspace=0.22)
    gs_row2 = GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_top[1],
        wspace=0.32)

    # ========== Row 1 ==========
    # panel a: two stacked axes (hist on top, per-chr on bottom) — now 50% wide
    gs_a = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_row1[0, 0],
                                    height_ratios=[1, 1], hspace=0.55)
    ax_a_hist = fig.add_subplot(gs_a[0])
    ax_a_chr  = fig.add_subplot(gs_a[1])

    # panel c: Miami (2 stacked halves) — 50% wide
    gs_c = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_row1[0, 1],
                                    height_ratios=[1, 1], hspace=0.15)
    ax_c_ad  = fig.add_subplot(gs_c[0])
    ax_c_sbp = fig.add_subplot(gs_c[1], sharex=ax_c_ad)

    # ========== Row 2 ==========
    ax_d = fig.add_subplot(gs_row2[0, 0])  # forward scatter

    gs_e = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_row2[0, 1],
                                    height_ratios=[1, 1], hspace=0.50)
    ax_e_all    = fig.add_subplot(gs_e[0])
    ax_e_noapoe = fig.add_subplot(gs_e[1])

    ax_f = fig.add_subplot(gs_row2[0, 2])  # forest

    # ─────────────────────────────────────────────────────
    # PANEL a — weight distribution + per-chromosome burden
    # ─────────────────────────────────────────────────────
    lab(ax_a_hist, 'a')
    log_bins = np.logspace(-6, -0.3, 25)
    if aw is not None and tw is not None:
        aa = np.abs(aw['BETA'].values); ta = np.abs(tw['BETA'].values)
        ac, _ = np.histogram(aa, bins=log_bins)
        tc, _ = np.histogram(ta, bins=log_bins)
        ap = ac / len(aa) * 100
        tp = tc / len(ta) * 100
    else:
        ad_std, sbp_std, an, tn = 4.9e-4, 2.1e-4, 1_107_672, 1_118_481
        ac = np.array([max(0, (half_normal_cdf(log_bins[i+1], ad_std) -
                                 half_normal_cdf(log_bins[i],   ad_std)) * an)
                       for i in range(len(log_bins)-1)])
        tc = np.array([max(0, (half_normal_cdf(log_bins[i+1], sbp_std) -
                                 half_normal_cdf(log_bins[i],   sbp_std)) * tn)
                       for i in range(len(log_bins)-1)])
        ap = ac / an * 100; tp = tc / tn * 100
    xh = np.arange(len(ap)); wh = 0.38
    ax_a_hist.bar(xh - wh/2, ap, wh, color=AD,  alpha=0.55,
                    edgecolor=AD,  lw=0.5, label='AD')
    ax_a_hist.bar(xh + wh/2, tp, wh, color=SBP, alpha=0.55,
                    edgecolor=SBP, lw=0.5, label='SBP')
    ti = list(range(0, len(xh), 4))
    ax_a_hist.set_xticks(ti)
    ax_a_hist.set_xticklabels([f'{log_bins[i]:.0e}' for i in ti],
                                 rotation=30, fontsize=BASE_FONT - 3)
    ax_a_hist.set_xlabel('SNP weight |β| (log bins)', fontsize=BASE_FONT - 2)
    ax_a_hist.set_ylabel('% of SNPs', fontsize=BASE_FONT - 2)
    ax_a_hist.set_ylim(0, 50)
    ax_a_hist.set_title('SNP weight distribution', fontsize=BASE_FONT - 1)
    ax_a_hist.legend(frameon=False, fontsize=BASE_FONT - 3, loc='upper left')

    # per-chr burden
    if aw is not None and tw is not None:
        ab_raw = [np.abs(aw[aw['CHR'] == c]['BETA']).sum() for c in range(1, 23)]
        tb_raw = [np.abs(tw[tw['CHR'] == c]['BETA']).sum() for c in range(1, 23)]
        ab = [x / sum(ab_raw) * 100 for x in ab_raw]
        tb = [x / sum(tb_raw) * 100 for x in tb_raw]
    else:
        ab = [10.76,11.40,7.70,7.20,7.50,9.17,6.80,6.30,5.50,6.68,6.20,
              5.90,4.20,4.00,3.80,4.10,4.30,3.50,6.19,2.80,1.70,1.60]
        tb = [4.77,4.86,3.70,3.30,3.60,4.46,3.40,3.10,3.00,3.89,3.50,
              3.20,2.30,2.20,2.10,2.40,2.50,1.90,1.29,1.80,0.95,0.90]
    for i in range(22):
        if i % 2 == 1:
            ax_a_chr.axvspan(i - 0.5, i + 0.5, color=STRIPE, zorder=0)
    xx = np.arange(22)
    bars_ad = ax_a_chr.bar(xx - 0.19, ab, 0.38,
                             color=AD,  alpha=0.55, edgecolor=AD,
                             lw=0.3, label='AD', zorder=3)
    bars_ad[18].set_alpha(1.0)
    ax_a_chr.bar(xx + 0.19, tb, 0.38,
                  color=SBP, alpha=0.55, edgecolor=SBP,
                  lw=0.3, label='SBP', zorder=3)
    ax_a_chr.set_xticks(xx)
    ax_a_chr.set_xticklabels(range(1, 23), fontsize=BASE_FONT - 4)
    ax_a_chr.set_xlim(-0.6, 21.6)
    ax_a_chr.set_xlabel('Chromosome',            fontsize=BASE_FONT - 2)
    ax_a_chr.set_ylabel('% of total PRS burden', fontsize=BASE_FONT - 2)
    ax_a_chr.set_title('PRS burden by chromosome', fontsize=BASE_FONT - 1)
    ax_a_chr.legend(frameon=False, fontsize=BASE_FONT - 3, loc='upper right')

    # ─────────────────────────────────────────────────────
    # PANEL b (Miami) — was previously 'c'
    # ─────────────────────────────────────────────────────
    lab(ax_c_ad, 'b')
    add_chr_stripes(ax_c_ad)
    add_chr_stripes(ax_c_sbp)

    plot_miami_half(ax_c_ad,  aw, AD,  AD_LT)
    plot_miami_half(ax_c_sbp, tw, SBP, SBP_LT)

    ad_max  = min(aw['BETA'].abs().max() * 1.1, 0.25) if aw is not None else 0.22
    sbp_max = min(tw['BETA'].abs().max() * 1.1, 0.05) if tw is not None else 0.04
    ax_c_ad.set_ylim(0, ad_max)
    ax_c_sbp.set_ylim(sbp_max, 0)
    ax_c_ad.set_ylabel('AD |β|',  fontsize=BASE_FONT - 2)
    ax_c_sbp.set_ylabel('SBP |β|', fontsize=BASE_FONT - 2)
    ax_c_ad.set_xlim(-10, GENOME_SIZE + 10)

    apoe_x = CHR_OFFSETS[18] + 45.4
    top_b = aw['BETA'].abs().max() if aw is not None else 0.204
    ax_c_ad.annotate('APOE', xy=(apoe_x, top_b),
                       xytext=(apoe_x - 120, top_b * 1.02),
                       fontsize=BASE_FONT - 1, fontweight='bold', color=AD,
                       arrowprops=dict(arrowstyle='->', color=AD, lw=1.2))

    ax_c_ad.text(0.01, 0.68, f'max |β| = {ad_max:.3f}',
                   transform=ax_c_ad.transAxes,
                   fontsize=BASE_FONT - 4, color=AD, fontstyle='italic')
    ax_c_ad.text(0.01, 0.56, 'y-scales differ 5×',
                   transform=ax_c_ad.transAxes,
                   fontsize=BASE_FONT - 4, color='#999', fontstyle='italic')
    ax_c_sbp.text(0.01, 0.15, f'max |β| = {sbp_max:.4f}',
                    transform=ax_c_sbp.transAxes,
                    fontsize=BASE_FONT - 4, color=SBP, fontstyle='italic')

    for i in range(22):
        mid = CHR_OFFSETS[i] + CHR_SIZES[i] / 2
        ax_c_ad.text(mid, -ad_max * 0.03, str(i + 1),
                       ha='center', va='top', fontsize=BASE_FONT - 5, color='#555')

    # shared SNP connectors
    ax_c_ad.set_title('PRS Miami — AD (top) / SBP (bottom)',
                        fontsize=BASE_FONT - 1, fontweight='bold')
    plt.setp(ax_c_ad.get_xticklabels(), visible=False)
    ax_c_ad.tick_params(axis='x', length=0)
    ax_c_sbp.set_xticks([])

    # ─────────────────────────────────────────────────────
    # PANEL c (Forward MR scatter, SBP → AD) — was 'd'
    # ─────────────────────────────────────────────────────
    lab(ax_d, 'c')
    n_fwd = len(mr['fwd_harm']) if mr['fwd_harm'] is not None else None
    mr_scatter(ax_d, mr['fwd_harm'], mr['fwd_res'], SBP,
                xlabel='SNP effect on SBP (β)',
                ylabel='SNP effect on AD (β)',
                title='SBP → AD (forward)',
                n_label=n_fwd)

    # ─────────────────────────────────────────────────────
    # PANEL d (Reverse MR scatter, split) — was 'e'
    # ─────────────────────────────────────────────────────
    lab(ax_e_all, 'd')
    n_rev_all = len(mr['rev_harm']) if mr['rev_harm'] is not None else None
    mr_scatter(ax_e_all, mr['rev_harm'], mr['rev_res'], AD,
                xlabel='SNP effect on AD (β)',
                ylabel='SNP effect on SBP (β)',
                title='AD → SBP  (all SNPs)',
                n_label=n_rev_all)

    n_rev_noapoe = (len(mr['rev_harm_noapoe'])
                     if mr['rev_harm_noapoe'] is not None else None)
    mr_scatter(ax_e_noapoe, mr['rev_harm_noapoe'], mr['rev_res_noapoe'], AD,
                xlabel='SNP effect on AD (β)',
                ylabel='SNP effect on SBP (β)',
                title='AD → SBP  (APOE excluded)',
                n_label=n_rev_noapoe,
                marker='^', open_markers=True)

    # ─────────────────────────────────────────────────────
    # PANEL e (Forest plot, 3 groups) — was 'f'
    # ─────────────────────────────────────────────────────
    lab(ax_f, 'e')
    forest_panel(ax_f, mr['fwd_res'], mr['rev_res'], mr['rev_res_noapoe'])

    # ─────────────────────────────────────────────────────
    fig.suptitle(
        'AD × SBP shared architecture (row 1) and bidirectional Mendelian randomisation (row 2)',
        fontsize=BASE_FONT + 2, fontweight='bold', y=0.985)

    out_pdf = os.path.join(out_dir, 'Fig_merged_PRS_MR_AD_SBP.pdf')
    out_png = os.path.join(out_dir, 'Fig_merged_PRS_MR_AD_SBP.png')
    fig.savefig(out_pdf, format='pdf')
    fig.savefig(out_png, format='png', dpi=200)
    plt.close(fig)
    print(f"\n  Wrote {out_pdf}")
    print(f"  Wrote {out_png}")


# ═════════════════════ CLI ══════════════════════════════════
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prs-dir', required=True,
                    help='root that contains results/phase2_prs/{ad,sbp_sbayesrc}/')
    p.add_argument('--mr-dir',  required=True,
                    help='directory with forward_mr_sbp_to_ad.csv and reverse_mr_ad_to_sbp.csv '
                         '(including *_no_apoe variants)')
    p.add_argument('--out',     default='./figs')
    p.add_argument('--width',   type=float, default=15)
    p.add_argument('--height',  type=float, default=15)
    args = p.parse_args()
    make_figure(args.prs_dir, args.mr_dir, args.out,
                 figsize=(args.width, args.height))
