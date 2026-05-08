"""
single_trial_projection.py  —  Run AFTER bcne_train.py
Project a single trial through the full pipeline and plot its trajectory
against the grand average.

    python single_trial_projection.py                                         # latest run, trial 0
    python single_trial_projection.py results/bcne/20260429_193713
    python single_trial_projection.py results/bcne/20260429_193713 42
    python single_trial_projection.py results/bcne/20260429_193713 42 data/DR_8April_with_outliers.csv
"""
import sys, os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors      import Normalize
from matplotlib.collections import LineCollection
from tensorflow.keras.models import load_model, Model as KModel
from recursiveBCN_utils import create_kl_divergence
from projection import apply_fixed_projection

PEAK_MAP = {"P100": 100, "N170": 170, "P300": 300, "N400": 400}

OUT_DIR      = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob('results/bcne/2*'))[-1]
TRIAL_IDX    = int(sys.argv[2]) if len(sys.argv) > 2 else 0
CSV_OVERRIDE = sys.argv[3] if len(sys.argv) > 3 else None
print(f"Run directory : {OUT_DIR}")
print(f"Trial index   : {TRIAL_IDX}")

with open(f'{OUT_DIR}/csv_source.txt') as _f:
    _src = {l.split(':')[0].strip(): l.split(':', 1)[1].strip() for l in _f if ':' in l}
CSV_PATH = CSV_OVERRIDE if CSV_OVERRIDE else _src['csv']
N_COMP   = int(_src.get('n_comp',  2))
RECUR    = int(_src.get('recur',   3))
BALANCE  = int(_src.get('balance', 1))
print(f"CSV           : {CSV_PATH}{' (override)' if CSV_OVERRIDE else ''}")

autocorr_map = np.load(f'{OUT_DIR}/autocorr_map.npy')
T            = np.load(f'{OUT_DIR}/T_matrix.npy')
t_ms_train   = np.load(f'{OUT_DIR}/t_ms_train.npy')
n            = len(t_ms_train)
ROW = COL    = int(np.ceil(np.sqrt(T.shape[0])))
print(f"  autocorr_map:{autocorr_map.shape}  T:{T.shape}  n:{n}")

kl_loss    = create_kl_divergence(n // BALANCE, N_COMP)
best_model = load_model(f'{OUT_DIR}/m{RECUR}_avg.h5',
                        custom_objects={'KLdivergence': kl_loss})
emb_model  = KModel(inputs=best_model.input, outputs=best_model.output)
print(f"  Loaded m{RECUR}_avg.h5  |  params: {best_model.count_params():,}")

df = pd.read_csv(CSV_PATH).sort_values(["Trial_ID", "Time_ms"]).reset_index(drop=True)
_known_meta    = ["Time_ms", "Trial_ID", "Condition", "Continuous"]
eeg_cols       = [c for c in df.columns if c not in _known_meta]
HAS_CONDITION  = "Condition"  in df.columns
HAS_CONTINUOUS = "Continuous" in df.columns
n_trials = df["Trial_ID"].nunique()
n_time   = df["Time_ms"].nunique()
X_3d     = df[eeg_cols].values.reshape(n_trials, n_time, len(eeg_cols))
print(f"  X_3d: {X_3d.shape}  →  using trial index {TRIAL_IDX}")

if TRIAL_IDX >= n_trials:
    raise ValueError(f"TRIAL_IDX {TRIAL_IDX} out of range — only {n_trials} trials available")

trial_meta = df.drop_duplicates("Trial_ID").sort_values("Trial_ID").reset_index(drop=True)
trial_info = trial_meta.iloc[TRIAL_IDX]
trial_id   = trial_info["Trial_ID"]
condition  = trial_info["Condition"]  if HAS_CONDITION  else "N/A"
cont_val   = trial_info["Continuous"] if HAS_CONTINUOUS else "N/A"
print(f"  Trial_ID: {trial_id}  |  Condition: {condition}  |  Continuous: {cont_val}")

print(f"\nProjecting trial {TRIAL_IDX}...")
X_trial    = X_3d[TRIAL_IDX, :n, :]
trial_proj = apply_fixed_projection(X_trial, autocorr_map, T, ROW, COL)
trial_emb  = emb_model.predict(trial_proj, verbose=0)
print(f"  trial_emb: {trial_emb.shape}")

# load all trial embeddings to compute condition mean
emb_path = f'{OUT_DIR}/embedding_3d.npy'
if os.path.exists(emb_path):
    embedding_3d = np.load(emb_path)
    base_cond    = condition.split("_OUTLIER")[0]
    cond_labels  = trial_meta["Condition"].apply(lambda c: c.split("_OUTLIER")[0]).values
    cond_mask    = (cond_labels == base_cond) & np.array(["OUTLIER" not in c
                    for c in trial_meta["Condition"].values])
    ref_traj     = embedding_3d[cond_mask].mean(axis=0)
    ref_label    = f'{base_cond} mean'
else:
    print("  embedding_3d.npy not found — run trial_analysis.py first for condition mean")
    ref_traj  = None
    ref_label = None

erp_idx = {name: int(np.argmin(np.abs(t_ms_train - ms)))
           for name, ms in PEAK_MAP.items() if ms <= t_ms_train[-1]}

cmap_time = plt.cm.gnuplot2
norm_time = Normalize(vmin=t_ms_train[0], vmax=t_ms_train[-1])

def colored_line(ax, traj, lw=2.5):
    pts  = traj.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc   = LineCollection(segs, cmap=cmap_time, norm=norm_time, lw=lw, zorder=5)
    lc.set_array(t_ms_train[:-1]); ax.add_collection(lc); ax.autoscale()
    return lc

def mark_peaks(ax, traj):
    _off = (traj[:, 1].max() - traj[:, 1].min()) * 0.03
    for name, idx in erp_idx.items():
        ax.scatter(traj[idx, 0], traj[idx, 1], s=80, facecolors='white',
                   edgecolors='black', lw=1.5, zorder=8)
        ax.text(traj[idx, 0], traj[idx, 1] + _off, name,
                ha='center', fontsize=8, fontweight='bold', zorder=9)

ref = ref_traj if ref_traj is not None else np.load(f'{OUT_DIR}/grand_emb.npy')
ref_name = ref_label if ref_label is not None else 'grand avg'

all_pts = np.concatenate([ref, trial_emb], axis=0)
pad  = 0.25
xlim = (all_pts[:, 0].min() - pad * (all_pts[:, 0].ptp() or 1),
        all_pts[:, 0].max() + pad * (all_pts[:, 0].ptp() or 1))
ylim = (all_pts[:, 1].min() - pad * (all_pts[:, 1].ptp() or 1),
        all_pts[:, 1].max() + pad * (all_pts[:, 1].ptp() or 1))

fig, ax = plt.subplots(figsize=(8, 7), facecolor='white')
fig.suptitle(
    f"BCNE — Single Trial Projection\n"
    f"Trial index: {TRIAL_IDX}  |  Trial_ID: {trial_id}  |  "
    f"Condition: {condition}  |  Continuous: {cont_val}  |  Run: {os.path.basename(OUT_DIR)}",
    fontsize=12, fontweight='bold'
)

colored_line(ax, ref, lw=2)
ax.scatter(ref[:, 0], ref[:, 1], c=t_ms_train, cmap=cmap_time,
           norm=norm_time, s=20, edgecolors='black', lw=0.3, zorder=5, alpha=0.5,
           label=ref_name)

lc3 = colored_line(ax, trial_emb, lw=2.5)
ax.scatter(trial_emb[:, 0], trial_emb[:, 1], c=t_ms_train, cmap=cmap_time,
           norm=norm_time, s=60, edgecolors='red', lw=1.2, zorder=7,
           label=f'trial {trial_id}')

mark_peaks(ax, ref)
ax.set(title=f'Trial ID {trial_id} (red border) vs {ref_name}\n'
             f'Condition: {condition}  |  Continuous: {cont_val}',
       xlabel='BCNE dim 1', ylabel='BCNE dim 2', xlim=xlim, ylim=ylim)
ax.legend(fontsize=8)
ax.grid(True, ls='--', alpha=0.25)
fig.colorbar(lc3, ax=ax, label='Time (ms)', shrink=0.8)

plt.tight_layout()
out_png = f'{OUT_DIR}/single_trial_{TRIAL_IDX}.png'
fig.text(0.99, 0.01, out_png, ha='right', va='bottom',
         fontsize=6, color='gray', alpha=0.6)
plt.savefig(out_png, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved: {out_png}\nDone.")
