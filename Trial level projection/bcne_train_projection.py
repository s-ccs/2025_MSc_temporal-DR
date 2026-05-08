"""
bcne_train.py
Complete BCNE pipeline — auto-detects number of conditions, continuous levels,
channels, timepoints from the CSV. Works with any number of conditions.

Run: python bcne_train.py
     python bcne_train.py data/file.csv
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
from matplotlib.colors      import Normalize
from matplotlib.collections import LineCollection
from tensorflow.keras.models     import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models     import Model as KModel
from temporal_pro import TimeCORR
from spatial_pro import (createMeshDistance,createInteractionMatrix)
from spatial_OPT import (create_space_distributions,gromov_wasserstein_adjusted_norm)
from recursiveBCN_utils import (create_model,create_kl_divergence,train_model_with_patient,  calculate_low_para_for_input,
 calculate_low_para_for_layer)
from projection import apply_fixed_projection
import tensorflow as tf

SEED = 42
os.environ['PYTHONHASHSEED']        = str(SEED)
os.environ['TF_DETERMINISTIC_OPS']  = '1'
os.environ['TF_CUDNN_DETERMINISTIC']= '1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# file configuration
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else 'data/DR_EEG_10conditions_noise2.csv'

from datetime import datetime
_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR  = f'results/bcne/{_run_id}'
PEAK_MAP = {"P100": 100, "N170": 170, "P300": 300, "N400": 400}
BALANCE  = 1
RECUR    = 3
EPOCHS   = 200
PATIENT  = 20
N_COMP   = 2
os.makedirs(OUT_DIR, exist_ok=True)
with open(f'{OUT_DIR}/csv_source.txt', 'w') as _f:
    _f.write(f"csv: {CSV_PATH}\n")
    _f.write(f"recur: {RECUR}\n")
    _f.write(f"n_comp: {N_COMP}\n")
    _f.write(f"balance: {BALANCE}\n")
# STEP 1 — Load data + auto-detect structure
print("=" * 50)
print("STEP 1 — Load data")
print("=" * 50)

df = pd.read_csv(CSV_PATH).sort_values(
         ["Trial_ID", "Time_ms"]).reset_index(drop=True)

# Auto-detect meta vs EEG channel columns
_known_meta = ["Time_ms", "Trial_ID", "Condition", "Continuous"]
meta_cols   = [c for c in _known_meta if c in df.columns]
eeg_cols    = [c for c in df.columns  if c not in meta_cols]

# Auto-detect optional columns
HAS_CONDITION  = "Condition"  in df.columns
HAS_CONTINUOUS = "Continuous" in df.columns

n_trials = df["Trial_ID"].nunique()
n_time   = df["Time_ms"].nunique()
n_chan   = len(eeg_cols)
t_ms     = np.array(sorted(df["Time_ms"].unique()))
ROW = COL = int(np.ceil(np.sqrt(n_chan)))

X_3d = df[eeg_cols].values.reshape(n_trials, n_time, n_chan)
# Auto-detect conditions
if HAS_CONDITION:
    cond_labels  = (df.drop_duplicates("Trial_ID")
                      .sort_values("Trial_ID")
                      .reset_index(drop=True)["Condition"].values)
    conditions   = sorted(df["Condition"].unique())
    n_conditions = len(conditions)
    print(f"  Detected Condition column with {len(np.unique(cond_labels))} unique conditions")
    cond_masks   = {c: cond_labels == c for c in conditions} #array of true and false to select the rows of specific condition from the trials
    # Assign a distinct colour per condition
    _cmap_cond   = mplcm.get_cmap('tab10', n_conditions)
    cond_colors  = {c: _cmap_cond(i) for i, c in enumerate(conditions)}
else:
    conditions   = []
    n_conditions = 0
    cond_masks   = {}
    cond_colors  = {}

# Auto-detect continuous levels
if HAS_CONTINUOUS:
    cont_labels  = (df.drop_duplicates("Trial_ID")
                      .sort_values("Trial_ID")
                      .reset_index(drop=True)["Continuous"].values)
    cont_levels  = sorted(df["Continuous"].unique())
else:
    cont_labels  = None
    cont_levels  = []

print(f"  CSV            : {CSV_PATH}")
print(f"  X_3d shape     : {X_3d.shape}")
print(f"  n_trials       : {n_trials}")
print(f"  n_time         : {n_time}")
print(f"  n_channels     : {n_chan}  → grid {ROW}×{COL}")
print(f"  t_ms range     : {t_ms[0]}ms to {t_ms[-1]}ms")
print(f"  Conditions     : {conditions}  (n={n_conditions})")
print(f"  Cont levels    : {len(cont_levels)}  {[round(l,2) for l in cont_levels]}")

# STEP 2 — Grand average

print()
print("=" * 50)
print("STEP 2 — Grand average")
print("=" * 50)

X_avg      = X_3d.mean(axis=0)  #grand average across trials to get clean ERP signal 
n          = n_time
X_train    = X_avg[:n]
t_ms_train = t_ms[:n]
batch_size = n // BALANCE

print(f"  X_avg shape  : {X_avg.shape}")
print(f"  X_train shape: {X_train.shape}")
print(f"  batch_size   : {batch_size}")
print(f"  Averaged {n_trials} trials → clean ERP signal")

# ── Butterfly plot ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4), facecolor='white')
ax.plot(t_ms_train, X_train, color='steelblue', alpha=0.3, lw=0.8)
ax.plot(t_ms_train, X_train.mean(axis=1), color='black', lw=2, label='Mean')
ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.4)
ax.set(xlabel='Time (ms)', ylabel='Amplitude (μV)',
       title=f'Butterfly plot — Grand average ({n_chan} channels)')
ax.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()
_ylim = ax.get_ylim()
for name, ms in PEAK_MAP.items():
    if ms <= t_ms_train[-1]:
        ax.axvline(ms, color='gray', ls=':', alpha=0.6)
        ax.text(ms + 3, _ylim[1], name, rotation=90, va='top', fontsize=8, color='gray')
plt.savefig(f'{OUT_DIR}/butterfly_plot.png', dpi=150, bbox_inches='tight')
plt.show()

# STEP 3 — Compute fixed matrices
print()
print("=" * 50)
print("STEP 3 — Compute fixed matrices")
print("=" * 50)

print("  Computing autocorr_map...")
autocorr_map = TimeCORR(X=X_train, smooth_window=1)
X_train_auto = np.matmul(autocorr_map, X_train)
print(f"  autocorr_map shape: {autocorr_map.shape}")

print("  Computing Gromov-Wasserstein matrix (spatial mapping grid)")
distMat     = createMeshDistance(ROW, COL)
interactMat = createInteractionMatrix(X_train_auto)
n_eff       = min(n_chan, ROW * COL)
p, q        = create_space_distributions(n_eff, n_eff)
T = gromov_wasserstein_adjusted_norm(
        np.zeros((n_eff, n_eff)), interactMat,
        distMat[:n_eff, :n_eff], p, q,
        loss_fun='kl_loss', epsilon=0.0, max_iter=200)
print(f"  T shape: {T.shape}")

np.save(f'{OUT_DIR}/autocorr_map.npy', autocorr_map)
np.save(f'{OUT_DIR}/T_matrix.npy',     T)
print(f"  Saved autocorr_map.npy and T_matrix.npy")


# STEP 4 — Fixed projection function
#turn the ERP signal into image using the fixed matrices computed from the grand average
print()
print("=" * 50)
print("STEP 4 — Define apply_fixed_projection")
print("=" * 50)

X_train_proj = apply_fixed_projection(X_train, autocorr_map, T, ROW, COL)
print(f"  Test — Input: {X_train.shape}  Output: {X_train_proj.shape}")
print(f"  Min: {X_train_proj.min():.4f}  Max: {X_train_proj.max():.4f}")

# ── Neuromap visualisation ─────────────────────────────────────────────────
_vis_labels = {name: int(np.argmin(np.abs(t_ms_train - ms)))
               for name, ms in PEAK_MAP.items() if ms <= t_ms_train[-1]}
_n_show = len(_vis_labels)
fig_nm, axes_nm = plt.subplots(1, _n_show, figsize=(3 * _n_show, 3),
                               facecolor='white')
if _n_show == 1:
    axes_nm = [axes_nm]
fig_nm.suptitle("Neuromaps at ERP peaks (grand average)",
                fontsize=11, fontweight='bold')
for ax_nm, (name, idx) in zip(axes_nm, _vis_labels.items()):
    im = ax_nm.imshow(X_train_proj[idx, :, :, 0], cmap='RdBu_r',
                      aspect='equal', interpolation='nearest')
    ax_nm.set_title(f"{name}\n({t_ms_train[idx]:.0f} ms)", fontsize=9)
    ax_nm.axis('off')
    fig_nm.colorbar(im, ax=ax_nm, shrink=0.7, pad=0.02)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/neuromaps_erp_peaks.png', dpi=150, bbox_inches='tight')
plt.show()


# STEP 5 — Training projection

print()
print("=" * 50)
print("STEP 5 — Training projection")
print("=" * 50)

input_shape = (ROW, COL, 1)
print(f"  X_train_proj shape: {X_train_proj.shape}")

# STEP 6 — Build and train BCNE
print()
print("=" * 50)
print("STEP 6 — Build and train BCNE")
print("=" * 50)

model_design = [3, [3, 16, 32], (256, 128, 64, 8), N_COMP]
model = create_model(
    input_shape     = input_shape,
    num_conv_layers = model_design[0],
    filters_list    = model_design[1],
    kernel_size     = 3,
    alpha           = 0.05,
    dense_units     = model_design[2],
    final_units     = model_design[3]
)
kl_loss = create_kl_divergence(batch_size, N_COMP)
model.compile(loss=kl_loss, optimizer=Adam(learning_rate=0.0005))
print(f"  Model parameters: {model.count_params():,}")

out_paths = [f'{OUT_DIR}/m{i}_avg.h5' for i in range(1, RECUR + 1)]
HD_type   = 'sherlock'

print("  ── Recursion 1 ──")
model = train_model_with_patient(
    model, X_train_proj, out_paths[0],
    calculate_low_para_for_input,
    EPOCHS, PATIENT, n, batch_size, HD_type)

if RECUR > 1:
    print("  ── Recursion 2 ──")
    model = load_model(out_paths[0], custom_objects={'KLdivergence': kl_loss})
    model.compile(loss=kl_loss, optimizer=Adam(learning_rate=0.0005))
    model = train_model_with_patient(
        model, X_train_proj, out_paths[1],
        lambda m, X, n, bs, hd: calculate_low_para_for_layer(m, X, 'Dense1', n, bs, hd),
        EPOCHS, PATIENT, n, batch_size, HD_type)

if RECUR > 2:
    print("  ── Recursion 3 ──")
    model = load_model(out_paths[1], custom_objects={'KLdivergence': kl_loss})
    model.compile(loss=kl_loss, optimizer=Adam(learning_rate=0.0005))
    model = train_model_with_patient(
        model, X_train_proj, out_paths[2],
        lambda m, X, n, bs, hd: calculate_low_para_for_layer(m, X, 'Dense2', n, bs, hd),
        EPOCHS, PATIENT, n, batch_size, HD_type)

print("  Training complete!")

# STEP 7 — Grand average embedding

print()
print("=" * 50)
print("STEP 7 — Grand average embedding")
print("=" * 50)

best_model = load_model(out_paths[RECUR - 1],
                        custom_objects={'KLdivergence': kl_loss})
emb_model  = KModel(inputs=best_model.input, outputs=best_model.output)
grand_emb  = emb_model.predict(X_train_proj, verbose=0)   # (n_time, 2)

np.save(f'{OUT_DIR}/grand_emb.npy',  grand_emb)
np.save(f'{OUT_DIR}/t_ms_train.npy', t_ms_train)

erp_idx = {name: int(np.argmin(np.abs(t_ms_train - ms)))
           for name, ms in PEAK_MAP.items() if ms <= t_ms_train[-1]}

print(f"  grand_emb shape: {grand_emb.shape}")
print(f"  {'Peak':<8} {'dim1':>10} {'dim2':>10}")
print("  " + "-" * 32)
for name, idx in erp_idx.items():
    print(f"  {name:<8} {grand_emb[idx,0]:>10.4f} {grand_emb[idx,1]:>10.4f}")


# STEP 8 — Project condition averages (auto N conditions)

print()
print("=" * 50)
print("STEP 8 — Project condition averages")
print("=" * 50)

cond_emb = {}
cond_dev = {}

if HAS_CONDITION:
    for cond in conditions:
        mask = cond_masks[cond]
        print(f"  {cond} ({mask.sum()} trials)...")
        X_avg_cond     = X_3d[mask].mean(axis=0)
        proj           = apply_fixed_projection(X_avg_cond, autocorr_map, T, ROW, COL)
        cond_emb[cond] = emb_model.predict(proj, verbose=0)
        cond_dev[cond] = np.linalg.norm(
                             cond_emb[cond] - grand_emb, axis=1)

    np.save(f'{OUT_DIR}/cond_emb.npy', cond_emb)
    print()
    print(f"  Deviation from grand average at ERP peaks:")
    header = f"  {'Peak':<8}" + "".join(f"{c:>12}" for c in conditions)
    print(header)
    print("  " + "-" * (8 + 12 * n_conditions))
    for name, idx in erp_idx.items():
        row = f"  {name:<8}"
        for cond in conditions:
            row += f"{cond_dev[cond][idx]:>12.4f}"
        print(row)
    print(f"\n  Saved cond_emb.npy")
else:
    print("  No Condition column — skipping")


# STEP 9 — Project continuous level averages
print()
print("=" * 50)
print("STEP 9 — Project continuous level averages")
print("=" * 50)

cont_emb = {}
cont_dev = {}

if HAS_CONTINUOUS:
    for lvl in cont_levels:
        lvl_mask      = cont_labels == lvl
        X_lvl         = X_3d[lvl_mask].mean(axis=0)
        lvl_proj      = apply_fixed_projection(X_lvl, autocorr_map, T, ROW, COL)
        cont_emb[lvl] = emb_model.predict(lvl_proj, verbose=0)
        print(f"  Level {lvl:+.2f} ({lvl_mask.sum()} trials) → {cont_emb[lvl].shape}")

    ref      = cont_emb[cont_levels[0]]
    cont_dev = {lvl: np.linalg.norm(cont_emb[lvl] - ref, axis=1)
                for lvl in cont_levels}

    np.save(f'{OUT_DIR}/cont_emb.npy', cont_emb)

    print()
    print("  Deviations at P300:")
    p300_idx = erp_idx.get("P300", list(erp_idx.values())[0])
    for lvl in cont_levels:
        note = " ← reference" if lvl == cont_levels[0] \
               else (" ← expect largest" if lvl == cont_levels[-1] else "")
        print(f"  {lvl:>+8.2f}  {cont_dev[lvl][p300_idx]:>10.4f}{note}")
    print(f"\n  Saved cont_emb.npy")
else:
    print("  No Continuous column — skipping")

# STEP 10 — Plot
print()
print("=" * 50)
print("STEP 10 — Plotting")
print("=" * 50)

from matplotlib.gridspec import GridSpec

cmap_time = plt.cm.gnuplot2
cmap_cont = plt.cm.coolwarm
norm_time = Normalize(vmin=t_ms_train[0], vmax=t_ms_train[-1])
norm_cont = Normalize(vmin=min(cont_levels) if cont_levels else 0,
                      vmax=max(cont_levels) if cont_levels else 1)

def colored_line(ax, traj, lw=2.5):
    pts  = traj.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc   = LineCollection(segs, cmap=cmap_time, norm=norm_time, lw=lw, zorder=5)
    lc.set_array(t_ms_train[:-1])
    ax.add_collection(lc); ax.autoscale()
    return lc

def mark_peaks_2d(ax, traj):
    _yrange = traj[:, 1].max() - traj[:, 1].min()
    _offset = _yrange * 0.03
    for name, idx in erp_idx.items():
        ax.scatter(traj[idx,0], traj[idx,1], s=80, facecolors='white',
                   edgecolors='black', lw=1.5, zorder=8)
        ax.text(traj[idx,0], traj[idx,1] + _offset, name,
                ha='center', fontsize=8, fontweight='bold', zorder=9)

def mark_peaks_time(ax, y_pos):
    for name, idx in erp_idx.items():
        ax.axvline(t_ms_train[idx], color='gray', ls=':', alpha=0.5)
        ax.text(t_ms_train[idx]+3, y_pos, name,
                fontsize=8, color='gray', rotation=90, va='bottom')

def cont_colorbar(ax):
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_cont, norm=norm_cont),
                 ax=ax, label='Continuous value', shrink=0.8)

n_cols = 1 + (1 if HAS_CONDITION else 0) + (1 if HAS_CONTINUOUS else 0)
n_rows = 1 + (1 if HAS_CONTINUOUS else 0)
fig    = plt.figure(figsize=(6*n_cols, 5*n_rows), facecolor='white')
fig.suptitle("BCNE — Unsupervised learning\nTrain: grand average only",
             fontsize=13, fontweight='bold')
gs  = GridSpec(n_rows, n_cols, figure=fig)
col = 0

# Panel 1 — grand average
ax = fig.add_subplot(gs[0, col]); col += 1
lc = colored_line(ax, grand_emb, lw=3)
ax.scatter(grand_emb[:,0], grand_emb[:,1], c=t_ms_train, cmap=cmap_time,
           norm=norm_time, s=30, edgecolors='black', lw=0.4, zorder=5)
mark_peaks_2d(ax, grand_emb)
ax.set(title='Grand average\n(training — no labels used)',
       xlabel='BCNE dim 1', ylabel='BCNE dim 2')
ax.grid(True, ls='--', alpha=0.25)
fig.colorbar(lc, ax=ax, label='Time (ms)', shrink=0.8)

# Panel 2 — conditions
if HAS_CONDITION:
    ax = fig.add_subplot(gs[0, col]); col += 1
    colored_line(ax, grand_emb, lw=1.2)
    for i, cond in enumerate(conditions):
        ax.plot(cond_emb[cond][:,0], cond_emb[cond][:,1],
                color=cond_colors[cond], lw=2.5,
                ls=['-','--',':','-.'][ i % 4],
                label=f'{cond} avg', zorder=4)
    mark_peaks_2d(ax, grand_emb)
    ax.set(title=f"{n_conditions} conditions\n(Use label after training)",
           xlabel='BCNE dim 1', ylabel='BCNE dim 2')
    ax.legend(fontsize=8, loc='best'); ax.grid(True, ls='--', alpha=0.25)

# Panel 3 & 4 — continuous
if HAS_CONTINUOUS:
    ax = fig.add_subplot(gs[0, col])
    for lvl in cont_levels:
        ax.plot(cont_emb[lvl][:,0], cont_emb[lvl][:,1],
                color=cmap_cont(norm_cont(lvl)), lw=1.8, alpha=0.85)
    mark_peaks_2d(ax, grand_emb)
    ax.set(title='Continuous levels\n(Use label after training)',
           xlabel='BCNE dim 1', ylabel='BCNE dim 2')
    ax.grid(True, ls='--', alpha=0.25); cont_colorbar(ax)

    ax = fig.add_subplot(gs[1, 0])
    for lvl in cont_levels:
        ax.plot(t_ms_train, cont_dev[lvl],
                color=cmap_cont(norm_cont(lvl)), lw=2, alpha=0.85)
    max_dev  = max(v.max() for v in cont_dev.values())
    mark_peaks_time(ax, y_pos=max_dev * 0.05)
    mean_dev  = np.mean(np.stack(list(cont_dev.values())), axis=0)
    peak_name = min(erp_idx, key=lambda n: abs(erp_idx[n] - int(np.argmax(mean_dev)))) if erp_idx else ''
    ax.set(xlabel='Time (ms)', ylabel=f'Distance from level {cont_levels[0]:.2f}',
           title=f'Continuous effect peak at {peak_name}')
    ax.grid(True, ls='--', alpha=0.25); cont_colorbar(ax)

plt.tight_layout()
outfile = f'{OUT_DIR}/bcne_results.png'

fig.text(0.99, 0.01, outfile, ha='right', va='bottom',
         fontsize=6, color='gray', alpha=0.6)
plt.savefig(outfile, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: {outfile}")

if HAS_CONDITION:
    fig_dev, ax_dev = plt.subplots(figsize=(10, 4), facecolor='white')

    for cond in conditions:
        ax_dev.plot(t_ms_train, cond_dev[cond],
                    color=cond_colors[cond], lw=2, label=cond)

    _dev_all = np.concatenate(list(cond_dev.values()))
    _dev_min, _dev_max = _dev_all.min(), _dev_all.max()
    _dev_offset = _dev_min + (_dev_max - _dev_min) * 0.05
    for name, idx in erp_idx.items():
        ax_dev.axvline(t_ms_train[idx], color='gray', ls=':', alpha=0.5)
        ax_dev.text(t_ms_train[idx] + 3, _dev_offset, name,
                    fontsize=8, color='gray', rotation=90, va='bottom')

    ax_dev.set(xlabel='Time (ms)',
               ylabel='Distance from grand average',
               title='Condition deviation over time\n'
                     '(peaks at N170 and N400)')
    ax_dev.legend(fontsize=8)
    ax_dev.grid(True, ls='--', alpha=0.25)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/condition_deviation.png', dpi=150, bbox_inches='tight')
    plt.show()
print()
print("=" * 50)
print(f"COMPLETE — check {outfile}")
print("=" * 50)
