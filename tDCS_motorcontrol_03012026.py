#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Reproducible Analysis Script

Effects of Transcranial Direct Current Stimulation over the Left Sensorimotor
Cortex on Bimanual Force Control: A Computational and Experimental Investigation

Lima, V.M.S., Arthur, E.F., Gonzaga, R.R.D., Diniz, L.F.,
Pedreiro, R.C.M., Pinto Neto, O.

Correspondence: osmar@csusm.edu
================================================================================

INSTRUCTIONS:
  1. Place the data file 'data_tdcs_12212025.csv' and the raw force files
     (e.g., DAS_CONSTANTE_21_10_PRE.txt) in a folder of your choice.
  2. Update DATA_PATH below to point to that folder.
  3. Run this script:  python reproducible_analysis.py
  4. All figures and statistical output will be saved to DATA_PATH/results/

Requirements:
  pip install numpy pandas scipy matplotlib

================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, filtfilt, welch, coherence
from scipy import stats
from scipy.stats import f as f_dist
import warnings
import os
import sys

warnings.filterwarnings('ignore')

# =============================================================================
# USER CONFIGURATION — CHANGE THIS PATH
# =============================================================================

DATA_PATH = r"C:\Users\osmar\OneDrive\Documents\PESQUISAS\Vinicius"

# Raw force files for Figure 1 (optional — schematic used if not found)
# NOTE: On the authors' system these are in a different directory tree
RAW_DATA_PATH = DATA_PATH
EXAMPLE_PARTICIPANT = "DAS"      # participant ID for Figure 1
EXAMPLE_DATE = "21_10"            # date string in filename (underscore, not dot)

# =============================================================================
# SETUP
# =============================================================================

DATA_FILE = os.path.join(DATA_PATH, "data_tdcs_12212025.csv")
RESULTS_DIR = os.path.join(DATA_PATH, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Publication-quality figure defaults
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'legend.frameon': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# =============================================================================
# 1. DATA LOADING AND PREPARATION
# =============================================================================

def load_data(filepath):
    """Load experimental data and compute derived metrics."""
    if not os.path.exists(filepath):
        print(f"\nERROR: Data file not found at:\n  {filepath}")
        print(f"\nPlease update DATA_PATH at the top of this script.")
        sys.exit(1)

    df = pd.read_csv(filepath, encoding='utf-8-sig')

    # Standardize group label
    df['Group'] = df['Group'].replace('Placebo', 'Sham')

    # Compute force undershoot (%)
    df['Undershoot_pct'] = (
        100 * (df['OF_F_alvo_mean'] - df['OF_Total_mean_raw'])
        / df['OF_F_alvo_mean']
    )

    print(f"Loaded {len(df)} observations from {df['participant'].nunique()} "
          f"participants ({', '.join(df['Group'].unique())})")
    return df


def load_raw_force_file(filepath):
    """Load raw force data from Vernier .txt file."""
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    time_v, f1_v, f2_v, ft_v = [], [], [], []
    target = None
    for line in lines[7:]:
        line = line.strip().replace('\r', '')
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) >= 4:
            try:
                time_v.append(float(parts[0].replace(',', '.')))
                f1_v.append(float(parts[1].replace(',', '.')))
                f2_v.append(float(parts[2].replace(',', '.')))
                ft_v.append(float(parts[3].replace(',', '.')))
                if len(parts) >= 5 and parts[4].strip():
                    try:
                        target = float(parts[4].replace(',', '.'))
                    except ValueError:
                        pass
            except ValueError:
                continue
    t_arr = np.array(time_v)
    fs = 1.0 / np.mean(np.diff(t_arr)) if len(t_arr) > 1 else 100.0
    return {'time': t_arr, 'force1': np.array(f1_v), 'force2': np.array(f2_v),
            'total': np.array(ft_v), 'target': target, 'fs': fs}


def load_participant_raw(participant, date_str):
    """Load PRE and POST raw force files for a participant."""
    result = {}
    for epoch, suffix in [('PRE', 'PRE'), ('POST', 'POS')]:
        fpath = os.path.join(RAW_DATA_PATH,
                             f"{participant}_CONSTANTE_{date_str}_{suffix}.txt")
        if os.path.exists(fpath):
            result[epoch] = load_raw_force_file(fpath)
    return result if result else None


# =============================================================================
# 2. STATISTICAL ANALYSES
# =============================================================================

def compute_descriptives(df):
    """Compute cell means and SDs for all primary metrics."""
    metrics = {
        'Undershoot_pct': 'Undershoot (%)',
        'OF_Total_RMSE_raw': 'RMSE (N)',
        'OF_Total_P_1_3Hz': 'Power 1-3 Hz (N²/Hz)',
    }
    rows = []
    for col, label in metrics.items():
        for grp in ['Sham', 'tDCS']:
            for ep in ['PRE', 'POS']:
                vals = df[(df['Group'] == grp) & (df['Epoch'] == ep)][col]
                rows.append({
                    'Metric': label, 'Group': grp, 'Epoch': ep,
                    'Mean': vals.mean(), 'SD': vals.std(),
                    'SEM': vals.std() / np.sqrt(len(vals)), 'N': len(vals)
                })
    return pd.DataFrame(rows)


def run_interaction_tests(df):
    """
    Test Group × Epoch interaction using independent t-test on change scores.
    F(1,22) = t² for the between-group comparison of (POST − PRE) differences.
    """
    metrics = {
        'Undershoot_pct': 'Undershoot (%)',
        'OF_Total_RMSE_raw': 'RMSE (N)',
        'OF_Total_P_1_3Hz': 'Power 1-3 Hz',
    }
    results = []
    for col, label in metrics.items():
        wide = df.pivot_table(index='participant', columns='Epoch',
                              values=col, aggfunc='first')
        wide['Group'] = (df.drop_duplicates('participant')
                         .set_index('participant')['Group'])
        wide['change'] = wide['POS'] - wide['PRE']

        tdcs = wide[wide['Group'] == 'tDCS']['change']
        sham = wide[wide['Group'] == 'Sham']['change']
        t_stat, p_val = stats.ttest_ind(tdcs, sham)

        results.append({
            'Metric': label,
            'tDCS_change': f"{tdcs.mean():.3f} ± {tdcs.std():.3f}",
            'Sham_change': f"{sham.mean():.3f} ± {sham.std():.3f}",
            'F(1,22)': t_stat ** 2,
            'p_interaction': p_val,
        })
    return pd.DataFrame(results)


def run_posthoc_paired(df):
    """Within-group paired t-tests (PRE vs POST) with Holm correction."""
    metrics = {
        'Undershoot_pct': 'Undershoot (%)',
        'OF_Total_RMSE_raw': 'RMSE (N)',
        'OF_Total_P_1_3Hz': 'Power 1-3 Hz',
    }
    results = []
    for col, label in metrics.items():
        wide = df.pivot_table(index='participant', columns='Epoch',
                              values=col, aggfunc='first')
        wide['Group'] = (df.drop_duplicates('participant')
                         .set_index('participant')['Group'])
        for grp in ['tDCS', 'Sham']:
            sub = wide[wide['Group'] == grp]
            t_stat, p_val = stats.ttest_rel(sub['PRE'], sub['POS'])
            diff = sub['POS'] - sub['PRE']
            d = diff.mean() / diff.std()
            results.append({
                'Metric': label, 'Group': grp,
                'PRE_mean': sub['PRE'].mean(), 'PRE_sd': sub['PRE'].std(),
                'POST_mean': sub['POS'].mean(), 'POST_sd': sub['POS'].std(),
                't': t_stat, 'df': len(sub) - 1,
                'p_uncorrected': p_val, 'Cohen_d': d,
            })

    res_df = pd.DataFrame(results)
    # Holm–Bonferroni correction across all 6 tests
    sorted_idx = res_df['p_uncorrected'].argsort().values
    n = len(res_df)
    p_holm = np.ones(n)
    for rank, idx in enumerate(sorted_idx):
        p_holm[idx] = min(res_df['p_uncorrected'].iloc[idx] * (n - rank), 1.0)
    for i in range(1, n):
        p_holm[sorted_idx[i]] = max(p_holm[sorted_idx[i]], p_holm[sorted_idx[i - 1]])
    res_df['p_holm'] = p_holm
    return res_df


def run_correlations(df):
    """Pearson correlations between change scores, by group."""
    wide_u = df.pivot_table(index='participant', columns='Epoch',
                            values='Undershoot_pct', aggfunc='first')
    wide_p = df.pivot_table(index='participant', columns='Epoch',
                            values='OF_Total_P_1_3Hz', aggfunc='first')
    wide_r = df.pivot_table(index='participant', columns='Epoch',
                            values='OF_Total_RMSE_raw', aggfunc='first')
    groups = (df.drop_duplicates('participant')
              .set_index('participant')['Group'])

    delta_u = wide_u['POS'] - wide_u['PRE']
    delta_p = wide_p['POS'] - wide_p['PRE']
    delta_r = wide_r['POS'] - wide_r['PRE']

    results = []
    for grp in ['tDCS', 'Sham']:
        mask = groups == grp
        r1, p1 = stats.pearsonr(delta_p[mask], delta_u[mask])
        r2, p2 = stats.pearsonr(delta_p[mask], delta_r[mask])
        results.append({'Group': grp, 'Comparison': 'ΔPower vs ΔUndershoot',
                        'r': r1, 'p': p1})
        results.append({'Group': grp, 'Comparison': 'ΔPower vs ΔRMSE',
                        'r': r2, 'p': p2})
    return pd.DataFrame(results)


def run_ancova(df):
    """ANCOVA: POST ~ Group + PRE (baseline covariate)."""
    metrics = {
        'Undershoot_pct': 'Undershoot (%)',
        'OF_Total_RMSE_raw': 'RMSE (N)',
        'OF_Total_P_1_3Hz': 'Power 1-3 Hz',
    }
    results = []
    for col, label in metrics.items():
        wide = df.pivot_table(index='participant', columns='Epoch',
                              values=col, aggfunc='first')
        wide['Group'] = (df.drop_duplicates('participant')
                         .set_index('participant')['Group'])
        wide['Group_code'] = (wide['Group'] == 'tDCS').astype(float)

        X = np.column_stack([
            np.ones(len(wide)),
            wide['Group_code'].values,
            wide['PRE'].values,
        ])
        y = wide['POS'].values
        # Manual OLS for minimal dependencies
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ beta
        residuals = y - y_hat
        n, k = X.shape
        mse = np.sum(residuals ** 2) / (n - k)
        se = np.sqrt(mse * np.diag(np.linalg.inv(X.T @ X)))
        t_vals = beta / se
        p_vals = 2 * stats.t.sf(np.abs(t_vals), df=n - k)

        results.append({
            'Metric': label,
            'Group_beta': beta[1], 'Group_SE': se[1],
            'Group_t': t_vals[1], 'Group_p': p_vals[1],
            'Baseline_beta': beta[2], 'Baseline_p': p_vals[2],
        })
    return pd.DataFrame(results)


def compute_reliability(df):
    """Test-retest reliability (ICC) from Sham group PRE-POST."""
    metrics = {
        'Undershoot_pct': 'Undershoot (%)',
        'OF_Total_RMSE_raw': 'RMSE (N)',
        'OF_Total_P_1_3Hz': 'Power 1-3 Hz',
        'OF_Coh_1_3Hz': 'Coherence 1-3 Hz',
    }
    results = []
    sham = df[df['Group'] == 'Sham']
    for col, label in metrics.items():
        wide = sham.pivot_table(index='participant', columns='Epoch',
                                values=col, aggfunc='first')
        pre, post = wide['PRE'].values, wide['POS'].values
        n = len(pre)
        # ICC(3,1) — two-way mixed, single measures, consistency
        grand_mean = np.mean(np.concatenate([pre, post]))
        ss_between = 2 * np.sum((np.mean([pre, post], axis=0) - grand_mean) ** 2)
        ss_within = np.sum((pre - np.mean([pre, post], axis=0)) ** 2 +
                           (post - np.mean([pre, post], axis=0)) ** 2)
        ms_between = ss_between / (n - 1)
        ms_within = ss_within / n
        icc = (ms_between - ms_within) / (ms_between + ms_within)
        results.append({'Metric': label, 'ICC': icc, 'N': n})
    return pd.DataFrame(results)


# =============================================================================
# 3. COMPUTATIONAL MODEL
# =============================================================================

def butter_lowpass(x, fs, fc, order=4):
    """Apply a low-pass Butterworth filter."""
    b, a = butter(order, fc / (fs / 2), btype='low')
    return filtfilt(b, a, x)


def simulate_bimanual_force(G_proprio=0.5, target_N=45.0, duration=40.0,
                            fs=100.0, seed=None):
    """
    Closed-loop computational model of bimanual isometric force control.

    Matches the model described in Section 2.6 of the manuscript:
      - Vision ON (t < 20 s): visual feedback control with gain K_vis
      - Vision OFF (t >= 20 s): proprioceptive feedback with gain G_proprio,
        internal target drift (lambda_drift_base = 0.0045), and
        proprioceptive delay (delta)
      - Between-trial variability in lambda_drift and K_correction

    Parameters
    ----------
    G_proprio : float
        Proprioceptive feedback gain [0.2, 0.8]. Sham ≈ 0.25, tDCS ≈ 0.70.
    target_N : float
        Per-hand target force in Newtons (total target = 2 × target_N).
    duration : float
        Trial duration in seconds.
    fs : float
        Sampling rate in Hz.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict with keys: time, force_L, force_R, total_force, target,
                    undershoot_pct, rmse, power_1_3Hz, freqs_psd, psd,
                    freqs_coh, coh_spectrum, G_proprio
    """
    if seed is not None:
        np.random.seed(seed)

    dt = 1.0 / fs
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs
    target_total = target_N * 2

    force_L = np.zeros(n_samples)
    force_R = np.zeros(n_samples)

    # ---- Fixed parameters (Table in Section 2.6) ----
    K_vis = 0.12          # Visual feedback gain
    K_base = 0.15         # Base correction gain
    w_common = 0.15       # Common drive weight
    delay_samples = int(0.1 * fs)  # Proprioceptive delay δ = 100 ms
    sigma_proprio = 0.15   # Proprioceptive noise SD
    sigma_motor_von = 0.22   # Motor noise during Vision ON
    sigma_motor_voff = 0.25  # Motor noise during Vision OFF

    # ---- Between-trial variability (Section 2.6) ----
    lambda_drift_base = 0.0045 * (1 - 0.5 * G_proprio)
    lambda_drift = max(0.0005,
                       lambda_drift_base + np.random.randn() * 0.0008)
    K_correction = max(0.02,
                       K_base * G_proprio + np.random.randn() * 0.008)

    # ---- Shared signals ----
    common_drive = butter_lowpass(np.random.randn(n_samples) * 0.25, fs, 2)
    tremor_L = 0.06 * np.sin(2 * np.pi * 10.0 * t +
                              np.random.rand() * 2 * np.pi)
    tremor_R = 0.06 * np.sin(2 * np.pi * 10.2 * t +
                              np.random.rand() * 2 * np.pi)

    # ---- Simulation ----
    ramp_end = int(3 * fs)
    vision_off_start = int(20 * fs)
    internal_target = target_N

    for i in range(1, n_samples):
        if i < ramp_end:
            # Ramp-up phase
            frac = i / ramp_end
            force_L[i] = target_N * frac + np.random.randn() * 0.15
            force_R[i] = target_N * frac + np.random.randn() * 0.15

        elif i < vision_off_start:
            # Vision ON — Eq. (1) in manuscript
            err_L = target_N - force_L[i - 1]
            err_R = target_N - force_R[i - 1]
            force_L[i] = (force_L[i - 1]
                          + K_vis * err_L
                          + common_drive[i] * w_common
                          + np.random.randn() * sigma_motor_von
                          + tremor_L[i])
            force_R[i] = (force_R[i - 1]
                          + K_vis * err_R
                          + common_drive[i] * w_common
                          + np.random.randn() * sigma_motor_von
                          + tremor_R[i])

        else:
            # Vision OFF — Eqs. (2-4) in manuscript
            internal_target *= (1 - lambda_drift * dt)

            # Delayed proprioceptive estimate
            idx_delay = max(0, i - delay_samples)
            proprio_L = force_L[idx_delay] + np.random.randn() * sigma_proprio
            proprio_R = force_R[idx_delay] + np.random.randn() * sigma_proprio

            err_L = internal_target - proprio_L
            err_R = internal_target - proprio_R

            force_L[i] = (force_L[i - 1]
                          + K_correction * err_L
                          + common_drive[i] * w_common
                          + np.random.randn() * sigma_motor_voff
                          + tremor_L[i])
            force_R[i] = (force_R[i - 1]
                          + K_correction * err_R
                          + common_drive[i] * w_common
                          + np.random.randn() * sigma_motor_voff
                          + tremor_R[i])

        force_L[i] = np.clip(force_L[i], target_N * 0.7, target_N * 1.3)
        force_R[i] = np.clip(force_R[i], target_N * 0.7, target_N * 1.3)

    # ---- Compute metrics from Vision OFF analysis window (23–40 s) ----
    total_force = force_L + force_R
    of_start = int(23 * fs)
    of_end = int(40 * fs)
    of_force = total_force[of_start:of_end]

    undershoot = 100 * (target_total - np.mean(of_force)) / target_total
    rmse = np.sqrt(np.mean((of_force - target_total) ** 2))

    freqs_psd, psd = welch(of_force - np.mean(of_force), fs=fs,
                           nperseg=min(1024, len(of_force)), noverlap=512)
    mask_1_3 = (freqs_psd >= 1) & (freqs_psd <= 3)
    _trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    power_1_3 = _trapz(psd[mask_1_3], freqs_psd[mask_1_3]) if mask_1_3.sum() > 1 else 0

    freqs_coh, coh = coherence(force_L[of_start:of_end],
                               force_R[of_start:of_end],
                               fs=fs, nperseg=min(512, len(of_force)))

    return {
        'time': t, 'force_L': force_L, 'force_R': force_R,
        'total_force': total_force, 'target': target_total,
        'undershoot_pct': undershoot, 'rmse': rmse,
        'power_1_3Hz': power_1_3,
        'freqs_psd': freqs_psd, 'psd': psd,
        'freqs_coh': freqs_coh, 'coh_spectrum': coh,
        'G_proprio': G_proprio,
    }


def run_dose_response(G_values, n_seeds=20, target_N=45.0):
    """Run simulations across a range of G_proprio values."""
    rows = []
    for G in G_values:
        for seed in range(n_seeds):
            sim = simulate_bimanual_force(G_proprio=G, target_N=target_N,
                                          seed=seed * 1000 + int(G * 10000))
            rows.append({
                'G_proprio': G, 'seed': seed,
                'undershoot_pct': sim['undershoot_pct'],
                'rmse': sim['rmse'],
                'power_1_3Hz': sim['power_1_3Hz'],
            })
    return pd.DataFrame(rows)


# =============================================================================
# 4. FIGURES
# =============================================================================

# Color palette
C_SHAM = '#3182bd'
C_TDCS = '#e31a1c'


def fig1_experimental_design(raw_data, save_path):
    """Figure 1: Experimental design (montage + timeline + real force trace)."""
    print("  Creating Figure 1: Experimental Design...")

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, height_ratios=[1, 1.2], width_ratios=[1, 1.2])

    # Panel A: tDCS Montage schematic
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('A. tDCS Electrode Montage', fontweight='bold')
    head = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax.add_patch(head)
    ax.plot([0, 0], [1, 1.2], 'k-', lw=2)
    ax.text(0, 1.3, 'Nose', ha='center', fontsize=9)
    anode = plt.Circle((-0.25, 0.2), 0.22, color='red', alpha=0.7)
    ax.add_patch(anode)
    ax.text(-0.25, 0.2, '+', ha='center', va='center', fontsize=20,
            color='white', fontweight='bold')
    ax.text(-0.25, -0.12, 'Left of Cz\n(Anode)', ha='center', fontsize=9)
    cathode = plt.Circle((0.5, 0.7), 0.15, color='blue', alpha=0.7)
    ax.add_patch(cathode)
    ax.text(0.5, 0.7, '\u2013', ha='center', va='center', fontsize=18,
            color='white', fontweight='bold')
    ax.text(0.72, 0.85, 'Fp2\n(Cathode)', ha='center', fontsize=9)
    ax.plot(-1, 0, 'ko', ms=4); ax.text(-1.15, 0, 'T3', ha='right', fontsize=8)
    ax.plot(1, 0, 'ko', ms=4); ax.text(1.15, 0, 'T4', ha='left', fontsize=8)
    ax.text(-1.3, -0.8, "2 mA, 20 min\n35 cm\u00B2 electrodes\nSaline sponges",
            fontsize=9, bbox=dict(boxstyle='round', fc='lightgray', alpha=0.8),
            va='top')

    # Panel B: Task Timeline
    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(-2, 42); ax.set_ylim(-0.5, 2); ax.axis('off')
    ax.set_title('B. Task Timeline', fontweight='bold')
    ax.add_patch(FancyBboxPatch((0, 0.5), 20, 1, boxstyle="round,pad=0.02",
                 fc='lightblue', ec='blue', lw=2))
    ax.text(10, 1, 'Vision ON\n(0\u201320 s)', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.add_patch(FancyBboxPatch((20, 0.5), 20, 1, boxstyle="round,pad=0.02",
                 fc='lightyellow', ec='orange', lw=2))
    ax.text(30, 1, 'Vision OFF\n(20\u201340 s)', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.add_patch(FancyBboxPatch((23, 0.1), 17, 0.35, boxstyle="round,pad=0.02",
                 fc='lightgreen', ec='green', lw=1.5))
    ax.text(31.5, 0.27, 'Analysis Window (23\u201340 s)', ha='center',
            va='center', fontsize=9)
    ax.annotate('', xy=(42, 0), xytext=(-2, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(42, -0.15, 'Time (s)', ha='right', fontsize=10)
    for t_mark in [0, 3, 20, 23, 40]:
        ax.plot([t_mark, t_mark], [-0.1, 0.1], 'k-', lw=1)
        ax.text(t_mark, -0.25, str(t_mark), ha='center', fontsize=9)

    # Panel C: Real (or schematic) force trace
    ax = fig.add_subplot(gs[1, :])
    if raw_data is not None and 'POST' in raw_data:
        data = raw_data['POST']
        time_arr = data['time']
        total = data['total']
        target = data['target']
        fs_raw = data['fs']
        total_sm = butter_lowpass(total, fs_raw, 5)
        mask40 = time_arr <= 40
        time_arr, total_sm = time_arr[mask40], total_sm[mask40]
        ax.set_title('C. Bimanual Force-Matching Task (Representative Trial)',
                     fontweight='bold')
        ax.axvspan(0, 20, alpha=0.15, color='blue', label='Vision ON')
        ax.axvspan(20, 40, alpha=0.15, color='orange', label='Vision OFF')
        ax.plot(time_arr, total_sm, 'b-', lw=1, label='Total Force')
        ax.axhline(target, color='green', ls='--', lw=2,
                   label=f'Target ({target:.1f} N)')
        of_mask = (time_arr >= 23) & (time_arr <= 40)
        if of_mask.sum() > 0:
            mean_of = np.mean(total_sm[of_mask])
            u_pct = 100 * (target - mean_of) / target
            ax.annotate('', xy=(32, mean_of-10), xytext=(32, target-10),
                        arrowprops=dict(arrowstyle='<->', color='red', lw=2.5))
            ax.text(33.5, (mean_of + target) / 2-10,
                    f'Undershoot\n{u_pct:.1f}%', fontsize=14,
                    color='red', va='center', fontweight='bold')
            ax.axhline(mean_of, xmin=0.575, xmax=1.0, color='red', ls=':',
                       lw=1.5, alpha=0.7)
        ax.text(10, target + 4, 'Visual feedback\navailable',
                ha='center', fontsize=10)
        ax.text(30, target + 4, 'Proprioceptive\nfeedback only',
                ha='center', fontsize=10)
        ax.set_ylim(np.min(total_sm) - 5, target + 40)
    else:
        ax.set_title('C. Bimanual Force-Matching Task (Schematic)',
                     fontweight='bold')
        fs_s = 100; t_s = np.arange(0, 40, 1 / fs_s); tgt = 45.0
        force = np.zeros_like(t_s)
        force[t_s < 3] = tgt * (t_s[t_s < 3] / 3)
        m_on = (t_s >= 3) & (t_s < 20)
        force[m_on] = tgt + np.random.randn(m_on.sum()) * 0.8
        force[m_on] = butter_lowpass(force[m_on], fs_s, 3)
        force[m_on] += tgt - np.mean(force[m_on])
        m_off = t_s >= 20
        force[m_off] = tgt - 0.15 * (t_s[m_off] - 20) + np.random.randn(m_off.sum()) * 1
        force[m_off] = butter_lowpass(force[m_off], fs_s, 3)
        force = butter_lowpass(force, fs_s, 5)
        ax.axvspan(0, 20, alpha=0.15, color='blue', label='Vision ON')
        ax.axvspan(20, 40, alpha=0.15, color='orange', label='Vision OFF')
        ax.plot(t_s, force, 'b-', lw=1.5, label='Total Force')
        ax.axhline(tgt, color='green', ls='--', lw=2, label='Target (30% MVC)')
        m_of = np.mean(force[t_s >= 23])
        ax.annotate('', xy=(35, m_of), xytext=(35, tgt),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        ax.text(36.5, (m_of + tgt) / 2, 'Undershoot', fontsize=14,
                color='red', va='center')
        ax.text(10, tgt + 3, 'Visual feedback\navailable', ha='center',
                fontsize=10)
        ax.text(30, tgt + 3, 'Proprioceptive\nfeedback only', ha='center',
                fontsize=10)
        ax.set_ylim(38, 50)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (N)')
    ax.set_xlim(0, 42)
    ax.legend(loc='lower right')
    plt.tight_layout()
    fpath = os.path.join(save_path, 'Figure1_Experimental_Design.png')
    plt.savefig(fpath, facecolor='white')
    plt.savefig(fpath.replace('.png', '.svg'), format='svg')
    print(f"    Saved: {fpath}")
    plt.close()


def fig2_experimental_results(df, save_path):
    """Figure 2: Experimental results (Group × Epoch bar plots)."""
    print("  Creating Figure 2: Experimental Results...")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    epochs = ['PRE', 'POS']
    x = np.array([0, 1])
    w = 0.35

    metric_specs = [
        ('Undershoot_pct', 'Undershoot (%)', 'A. Force Undershoot', axes[0, 0]),
        ('OF_Total_RMSE_raw', 'RMSE (N)', 'B. Root Mean Square Error', axes[0, 1]),
        ('OF_Total_P_1_3Hz', 'Power (N²/Hz)', 'C. Spectral Power (1–3 Hz)', axes[1, 0]),
    ]

    for col, ylabel, title, ax in metric_specs:
        for gi, (grp, color) in enumerate(
                [('Sham', C_SHAM), ('tDCS', C_TDCS)]):
            means, sems = [], []
            for ep in epochs:
                vals = df[(df['Group'] == grp) & (df['Epoch'] == ep)][col]
                means.append(vals.mean())
                sems.append(vals.std() / np.sqrt(len(vals)))
            offset = -w / 2 if gi == 0 else w / 2
            ax.bar(x + offset, means, w, yerr=sems, capsize=5,
                   color=color, alpha=0.6, label=grp, edgecolor='black',
                   error_kw={'elinewidth': 1.5, 'capthick': 1.5})

        # Add within-tDCS significance bracket
        posthoc = run_posthoc_paired(df)
        row = posthoc[(posthoc['Metric'] == ylabel.split(' (')[0] + 
                       (' (%)' if '%' in ylabel else ' (N)' if 'N' in ylabel else ''))
                      & (posthoc['Group'] == 'tDCS')]
        # Simpler approach: just mark tDCS POST bar
        tdcs_post = df[(df['Group'] == 'tDCS') & (df['Epoch'] == 'POS')][col]
        tdcs_pre = df[(df['Group'] == 'tDCS') & (df['Epoch'] == 'PRE')][col]
        t_val, p_val = stats.ttest_rel(tdcs_pre, tdcs_post)
        if p_val < 0.05:
            ymax = max(tdcs_pre.mean(), tdcs_post.mean()) + \
                   max(tdcs_pre.std(), tdcs_post.std()) / np.sqrt(12)
            ax.plot([0 + w/2, 1 + w/2], [ymax * 1.08, ymax * 1.08],
                    'k-', linewidth=1.5)
            stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
            ax.text(0.5 + w/2, ymax * 1.10, stars, ha='center', fontsize=14)

        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['PRE', 'POST'])
        ax.legend()

    # Panel D: Coherence
    ax = axes[1, 1]
    bands = ['0–1 Hz', '1–3 Hz', '3–7 Hz', '7–12 Hz']
    coh_cols = ['OF_Coh_0_1Hz', 'OF_Coh_1_3Hz', 'OF_Coh_3_7Hz', 'OF_Coh_7_12Hz']
    x_coh = np.arange(len(bands))
    bw = 0.2

    for ei, (ep, alpha) in enumerate(zip(['PRE', 'POS'], [0.4, 0.8])):
        for gi, (grp, color) in enumerate([('Sham', C_SHAM), ('tDCS', C_TDCS)]):
            means = [df[(df['Group'] == grp) & (df['Epoch'] == ep)][c].mean()
                     for c in coh_cols]
            sems = [df[(df['Group'] == grp) & (df['Epoch'] == ep)][c].std()
                    / np.sqrt(12) for c in coh_cols]
            offset = (-1.5 + ei + gi * 2) * bw
            label = f'{grp} {ep.replace("POS", "POST")}'
            ax.bar(x_coh + offset, means, bw, yerr=sems, capsize=3,
                   color=color, alpha=alpha, label=label, edgecolor='black',
                   error_kw={'elinewidth': 1, 'capthick': 1})

    ax.set_ylabel('Coherence')
    ax.set_title('D. Inter-hand Coherence', fontweight='bold')
    ax.set_xticks(x_coh)
    ax.set_xticklabels(bands)
    ax.legend(ncol=2, fontsize=9)
    ax.set_ylim(0, 0.7)

    plt.tight_layout()
    fpath = os.path.join(save_path, 'Figure2_Experimental_Results.png')
    plt.savefig(fpath, facecolor='white')
    plt.savefig(fpath.replace('.png', '.svg'), format='svg')
    print(f"    Saved: {fpath}")
    plt.close()


def fig3_correlations(df, save_path):
    """Figure 3: Individual-level correlations (ΔPower vs ΔUndershoot/ΔRMSE)."""
    print("  Creating Figure 3: Correlations...")

    wide_u = df.pivot_table(index='participant', columns='Epoch',
                            values='Undershoot_pct', aggfunc='first')
    wide_p = df.pivot_table(index='participant', columns='Epoch',
                            values='OF_Total_P_1_3Hz', aggfunc='first')
    wide_r = df.pivot_table(index='participant', columns='Epoch',
                            values='OF_Total_RMSE_raw', aggfunc='first')
    groups = df.drop_duplicates('participant').set_index('participant')['Group']

    delta_u = wide_u['POS'] - wide_u['PRE']
    delta_p = wide_p['POS'] - wide_p['PRE']
    delta_r = wide_r['POS'] - wide_r['PRE']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {'Sham': C_SHAM, 'tDCS': C_TDCS}

    for ax, (delta_y, ylabel, title) in zip(axes, [
        (delta_u, 'Δ Undershoot (%)', 'A. Δ Power vs Δ Undershoot'),
        (delta_r, 'Δ RMSE (N)', 'B. Δ Power vs Δ RMSE'),
    ]):
        for grp in ['Sham', 'tDCS']:
            mask = groups == grp
            xv, yv = delta_p[mask], delta_y[mask]
            ax.scatter(xv, yv, c=colors[grp], s=100, edgecolors='k',
                       label=grp, zorder=5)
            if len(xv) > 2:
                slope, intercept, r, p, _ = stats.linregress(xv, yv)
                xline = np.array([xv.min(), xv.max()])
                ls = '-' if p < 0.05 else '--'
                ax.plot(xline, intercept + slope * xline,
                        color=colors[grp], linestyle=ls, alpha=0.7, lw=2)
                sig = '*' if p < 0.05 else ''
                ax.text(0.98, 0.95 if grp == 'tDCS' else 0.85,
                        f'{grp}: r = {r:.2f}, p = {p:.3f}{sig}',
                        transform=ax.transAxes, ha='right', fontsize=10,
                        color=colors[grp])

        ax.axhline(0, color='gray', ls=':', alpha=0.5)
        ax.axvline(0, color='gray', ls=':', alpha=0.5)
        ax.set_xlabel('Δ Power 1–3 Hz (N²/Hz)')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.legend()

    plt.tight_layout()
    fpath = os.path.join(save_path, 'Figure3_Correlations.png')
    plt.savefig(fpath, facecolor='white')
    plt.savefig(fpath.replace('.png', '.svg'), format='svg')
    print(f"    Saved: {fpath}")
    plt.close()


def fig4_model_results(df, save_path):
    """Figure 4: Computational model simulated time series (2 panels)."""
    print("  Creating Figure 4: Model Results...")

    sim_sham = simulate_bimanual_force(G_proprio=0.25, seed=42)
    sim_tdcs = simulate_bimanual_force(G_proprio=0.70, seed=42)
    t = sim_sham['time']
    target = sim_sham['target']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: Full time series
    ax = axes[0]
    ax.axvspan(0, 20, alpha=0.12, color='blue', label='Vision ON')
    ax.axvspan(20, 40, alpha=0.12, color='orange', label='Vision OFF')
    ax.plot(t, sim_sham['total_force'], color=C_SHAM, lw=0.8, alpha=0.8,
            label='Sham (G = 0.25)')
    ax.plot(t, sim_tdcs['total_force'], color=C_TDCS, lw=0.8, alpha=0.8,
            label='tDCS (G = 0.70)')
    ax.axhline(target, color='green', ls='--', lw=2, label='Target')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Total Force (N)')
    ax.set_title('A. Simulated Force Time Series', fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 40)
    ax.set_ylim(target * 0.87, target * 1.07)

    # Panel B: Vision OFF detail
    ax = axes[1]
    of_mask = (t >= 23) & (t <= 40)
    ax.plot(t[of_mask], sim_sham['total_force'][of_mask], color=C_SHAM,
            lw=1.2, label='Sham')
    ax.plot(t[of_mask], sim_tdcs['total_force'][of_mask], color=C_TDCS,
            lw=1.2, label='tDCS')
    ax.axhline(target, color='green', ls='--', lw=2)

    m_s = np.mean(sim_sham['total_force'][of_mask])
    m_t = np.mean(sim_tdcs['total_force'][of_mask])
    u_s = 100 * (target - m_s) / target
    u_t = 100 * (target - m_t) / target
    ax.axhline(m_s, color=C_SHAM, ls=':', alpha=0.7,
               label=f'Sham mean: {m_s:.1f} N ({u_s:.1f}%)')
    ax.axhline(m_t, color=C_TDCS, ls=':', alpha=0.7,
               label=f'tDCS mean: {m_t:.1f} N ({u_t:.1f}%)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Total Force (N)')
    ax.set_title('B. Vision OFF Epoch Detail', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(target * 0.90, target * 1.05)

    plt.tight_layout()
    fpath = os.path.join(save_path, 'Figure4_Model_Results.png')
    plt.savefig(fpath, facecolor='white')
    plt.savefig(fpath.replace('.png', '.svg'), format='svg')
    print(f"    Saved: {fpath}")
    plt.close()


def fig5_model_dose_response(save_path):
    """Figure 5: Computational model dose-response."""
    print("  Creating Figure 5: Model Dose-Response...")

    G_values = [0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    sweep = run_dose_response(G_values, n_seeds=20)
    grouped = sweep.groupby('G_proprio')
    G_vals = sorted(sweep['G_proprio'].unique())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (metric, ylabel, title, scale) in zip(axes, [
        ('undershoot_pct', 'Undershoot (%)', 'A. Undershoot vs G_proprio', 1),
        ('power_1_3Hz', 'Power 1–3 Hz (×10 N²/Hz)',
         'B. Corrective Power vs G_proprio', 10),
    ]):
        means = [grouped.get_group(g)[metric].mean() * scale for g in G_vals]
        stds = [grouped.get_group(g)[metric].std() * scale for g in G_vals]
        ax.errorbar(G_vals, means, yerr=stds, fmt='o-', capsize=5,
                    color='gray', lw=2, ms=8, alpha=0.8, capthick=1.5,
                    elinewidth=1.5, label='Model sweep')

        # Highlight Sham and tDCS
        for G, color, marker, label in [
            (0.25, C_SHAM, 'o', 'Sham (G = 0.25)'),
            (0.70, C_TDCS, 's', 'tDCS (G = 0.70)'),
        ]:
            g = grouped.get_group(G)[metric]
            ax.errorbar([G], [g.mean() * scale], yerr=[g.std() * scale],
                        fmt=marker, capsize=8, color=color, ms=14,
                        markeredgewidth=2, markeredgecolor='black',
                        capthick=2.5, elinewidth=2.5, label=label)

        ax.set_xlabel('G_proprio (Proprioceptive Gain)')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Panel C: Combined dose-response
    ax = axes[2]
    ax2 = ax.twinx()
    means_u = [grouped.get_group(g)['undershoot_pct'].mean() for g in G_vals]
    means_p = [grouped.get_group(g)['power_1_3Hz'].mean() * 10 for g in G_vals]

    l1, = ax.plot(G_vals, means_u, 'o-', color=C_SHAM, lw=2, ms=7,
                  label='Undershoot (%)')
    l2, = ax2.plot(G_vals, means_p, 's-', color=C_TDCS, lw=2, ms=7,
                   label='Power 1–3 Hz (×10)')

    for G, label_txt in [(0.25, 'Sham'), (0.70, 'tDCS')]:
        g = grouped.get_group(G)
        ax.plot(G, g['undershoot_pct'].mean(), 'o', color=C_SHAM, ms=14,
                markeredgecolor='black', markeredgewidth=2, zorder=10)
        ax2.plot(G, g['power_1_3Hz'].mean() * 10, 's', color=C_TDCS, ms=14,
                 markeredgecolor='black', markeredgewidth=2, zorder=10)
        ax.annotate(label_txt, xy=(G, g['undershoot_pct'].mean()),
                    xytext=(0, 12), textcoords='offset points',
                    ha='center', fontweight='bold', fontsize=11)

    ax.set_xlabel('G_proprio (Proprioceptive Gain)')
    ax.set_ylabel('Undershoot (%)', color=C_SHAM)
    ax2.set_ylabel('Power 1–3 Hz (×10)', color=C_TDCS)
    ax.set_title('C. Dose-Response Relationship', fontweight='bold')
    ax.tick_params(axis='y', labelcolor=C_SHAM)
    ax2.tick_params(axis='y', labelcolor=C_TDCS)
    ax.legend([l1, l2], [l1.get_label(), l2.get_label()], loc='center right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fpath = os.path.join(save_path, 'Figure5_Model_DoseResponse.png')
    plt.savefig(fpath, facecolor='white')
    plt.savefig(fpath.replace('.png', '.svg'), format='svg')
    print(f"    Saved: {fpath}")
    plt.close()



# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 75)
    print("REPRODUCIBLE ANALYSIS — tDCS Bimanual Force Control")
    print("=" * 75)

    # ---- Load data ----
    df = load_data(DATA_FILE)
    print()

    # ---- Statistical analyses ----
    print("Running statistical analyses...")

    desc = compute_descriptives(df)
    print("\n--- Descriptive Statistics ---")
    print(desc.to_string(index=False))

    interactions = run_interaction_tests(df)
    print("\n--- Group × Epoch Interactions ---")
    print(interactions.to_string(index=False))

    posthoc = run_posthoc_paired(df)
    print("\n--- Post-hoc Paired t-tests (Holm-corrected) ---")
    cols = ['Metric', 'Group', 'PRE_mean', 'POST_mean', 't', 'df',
            'p_uncorrected', 'p_holm', 'Cohen_d']
    print(posthoc[cols].to_string(index=False))

    corr = run_correlations(df)
    print("\n--- Correlations (change scores) ---")
    print(corr.to_string(index=False))

    ancova = run_ancova(df)
    print("\n--- ANCOVA (POST ~ Group + Baseline) ---")
    print(ancova.to_string(index=False))

    reliability = compute_reliability(df)
    print("\n--- Test-Retest Reliability (Sham ICC) ---")
    print(reliability.to_string(index=False))

    # ---- Save statistical tables ----
    desc.to_csv(os.path.join(RESULTS_DIR, 'descriptives.csv'), index=False)
    interactions.to_csv(os.path.join(RESULTS_DIR, 'interactions.csv'), index=False)
    posthoc.to_csv(os.path.join(RESULTS_DIR, 'posthoc.csv'), index=False)
    corr.to_csv(os.path.join(RESULTS_DIR, 'correlations.csv'), index=False)
    ancova.to_csv(os.path.join(RESULTS_DIR, 'ancova.csv'), index=False)
    reliability.to_csv(os.path.join(RESULTS_DIR, 'reliability.csv'), index=False)
    print(f"\nStatistical tables saved to: {RESULTS_DIR}")

    # ---- Generate figures ----
    print("\nGenerating figures...")

    # Figure 1: Experimental Design (uses raw force files if available)
    raw_data = None
    try:
        raw_data = load_participant_raw(EXAMPLE_PARTICIPANT,
                                        EXAMPLE_DATE)
        if raw_data:
            print(f"  Loaded raw data for {EXAMPLE_PARTICIPANT} "
                  f"({len(raw_data)} epochs)")
    except Exception as e:
        print(f"  Raw data not found ({e}); using schematic for Figure 1")
    fig1_experimental_design(raw_data, RESULTS_DIR)

    # Figure 3: Experimental Results
    fig2_experimental_results(df, RESULTS_DIR)

    # Figure 4: Correlations
    fig3_correlations(df, RESULTS_DIR)

    # Figure 5: Computational Model Results
    fig4_model_results(df, RESULTS_DIR)

    # Figure 6: Model Dose-Response
    fig5_model_dose_response(RESULTS_DIR)

    print("\n" + "=" * 75)
    print("ANALYSIS COMPLETE")
    print(f"All outputs saved to: {RESULTS_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    main()