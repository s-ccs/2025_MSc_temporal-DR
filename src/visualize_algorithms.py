"""
Paper-quality figure for ERP Temporal Dimensionality Reduction.
Adapted from BCNE paper Figure 5 (Macaque dataset) for ERP data.

Two-phase approach:
  Phase 1 (base env):  python visualize_algorithms.py --extract-bcne
  Phase 2 (TDR env):   conda run -n TDR python visualize_algorithms.py --plot
  Or both:             python visualize_algorithms.py

Reads from the latest results/erp/<timestamp>/ directory.
Saves output to results/erp/<timestamp>/compare_algorithms/.
Does NOT modify existing results.
"""
import os
import sys
import numpy as np
import pandas as pd
import random
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ============================================================
# Config
# ============================================================
CSV_PATH = 'data/eeg_32ch_allERP_15March.csv'
BALANCE = 4


def find_latest_bcne_dir():
    base = 'results/erp'
    if not os.path.exists(base):
        return None
    runs = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))],
                  key=lambda d: os.path.getmtime(os.path.join(base, d)))
    return os.path.join(base, runs[-1]) if runs else None


# ============================================================
# Evaluation helpers
# ============================================================
def knn_accuracy(embedding, labels, k=8):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    return np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=k), embedding, labels, cv=10))




# ============================================================
# Phase 1: Extract BCNE embeddings + save flat data (base env)
# ============================================================
def extract_bcne():
    import tensorflow as tf
    tf.config.experimental_run_functions_eagerly(True)
    tf.random.set_seed(SEED)
    from tensorflow.keras.models import load_model
    from recursiveBCN_utils import create_kl_divergence
    from data_preprocess import erp_data_preprocess, erp_data_preprocess_compare

    bcne_dir = find_latest_bcne_dir()
    if not bcne_dir:
        print("No BCNE results found. Run main.py first.")
        return
    out_dir = f'{bcne_dir}/compare_algorithms'
    os.makedirs(out_dir, exist_ok=True)

    X_proj, labels_cond, labels_cont, n, n_tp = erp_data_preprocess(
        CSV_PATH, BALANCE, normalization='global')
    n_groups = len(labels_cond)
    print(f"  BCNE data: {X_proj.shape} ({n_groups} groups x {n_tp} timepoints)")

    batch_size = n // BALANCE
    kl_loss = create_kl_divergence(batch_size, 2)

    for m in [1, 2, 3, 4]:
        model_path = f'{bcne_dir}/m{m}.h5'
        if not os.path.exists(model_path):
            continue
        model = load_model(model_path, custom_objects={'KLdivergence': kl_loss})
        emb = model.predict(X_proj, verbose=0)
        np.save(f'{out_dir}/BCNE_m{m}_embedding.npy', emb)
        print(f"  BCNE m{m}: {emb.shape}")

    X_flat, labels_cond, labels_cont, n_flat, n_tp = erp_data_preprocess_compare(CSV_PATH)
    np.save(f'{out_dir}/X_flat.npy', X_flat)
    np.save(f'{out_dir}/labels_cond.npy', labels_cond)
    np.save(f'{out_dir}/labels_cont.npy', labels_cont)
    np.savez(f'{out_dir}/metadata.npz', n_tp=n_tp, n_groups=n_groups)

    # Save per-group waveforms and metadata for panel (a)
    df = pd.read_csv(CSV_PATH)
    meta_cols = ['Time_ms', 'Trial_ID', 'Condition', 'Continuous']
    channel_cols = [c for c in df.columns if c not in meta_cols]
    conditions = df['Condition'].unique()
    cond_car = conditions[0]
    cond_face = conditions[1]
    erp_times_arr = df[df['Condition'] == cond_car].groupby('Time_ms')[channel_cols].mean().index.values
    np.save(f'{out_dir}/erp_times.npy', erp_times_arr)
    np.save(f'{out_dir}/channel_names.npy', np.array(channel_cols))
    np.save(f'{out_dir}/cond_names.npy', np.array([str(cond_car), str(cond_face)]))

    # Save per-group averaged waveforms (20 groups: 2 conditions x 10 continuous levels)
    trial_meta = df.groupby('Trial_ID').first()[['Condition', 'Continuous']].reset_index()
    groups = trial_meta.groupby(['Condition', 'Continuous']).groups
    sorted_keys = sorted(groups.keys(), key=lambda x: (0 if x[0] == 'car' else 1, x[1]))

    group_waveforms = []  # (20, 60, 32)
    group_keys = []       # (20, 2) - condition, continuous

    for key in sorted_keys:
        cond, cont = key
        trial_id_vals = trial_meta.loc[groups[key].tolist(), 'Trial_ID'].values
        trial_data_list = [df[df['Trial_ID'] == tid].sort_values('Time_ms')[channel_cols].values
                           for tid in trial_id_vals]
        avg_waveform = np.mean(trial_data_list, axis=0)  # (60, 32)
        group_waveforms.append(avg_waveform)
        group_keys.append([cond, cont])

    group_waveforms = np.array(group_waveforms)  # (20, 60, 32)
    group_keys = np.array(group_keys)
    np.save(f'{out_dir}/erp_group_waveforms.npy', group_waveforms)
    np.save(f'{out_dir}/erp_group_keys.npy', group_keys)

    print(f"Saved to: {out_dir}")
    print(f"  Per-group waveforms: {group_waveforms.shape}")


# ============================================================
# Phase 2: Run methods + paper figure (TDR env)
# ============================================================
def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.colors as mcolors
    from sklearn.decomposition import PCA
    import phate
    import tphate

    bcne_dir = find_latest_bcne_dir()
    out_dir = f'{bcne_dir}/compare_algorithms' if bcne_dir else None
    if not out_dir or not os.path.exists(f'{out_dir}/X_flat.npy'):
        print("Run --extract-bcne first.")
        return

    # Load data
    X_flat = np.load(f'{out_dir}/X_flat.npy')
    labels_cond = np.load(f'{out_dir}/labels_cond.npy')
    labels_cont = np.load(f'{out_dir}/labels_cont.npy')
    meta = np.load(f'{out_dir}/metadata.npz', allow_pickle=True)
    n_tp = int(meta['n_tp'])
    n_groups = int(meta['n_groups'])
    labels_cond_full = np.repeat(labels_cond, n_tp)
    labels_cont_full = np.repeat(labels_cont, n_tp)

    erp_times = np.load(f'{out_dir}/erp_times.npy')
    channel_names = np.load(f'{out_dir}/channel_names.npy', allow_pickle=True)
    cond_names_path = f'{out_dir}/cond_names.npy'
    if os.path.exists(cond_names_path):
        cond_names = np.load(cond_names_path, allow_pickle=True)
        cond_name_0, cond_name_1 = str(cond_names[0]).capitalize(), str(cond_names[1]).capitalize()
    else:
        cond_name_0, cond_name_1 = 'Car', 'Face'

    print(f"Data: {X_flat.shape}, {n_groups} groups, {n_tp} timepoints")

    # --- Run methods ---
    embeddings = {}
    methods = []

    # Order: PCA, t-SNE, UMAP, PHATE, T-PHATE, BCNE m1, BCNE m4
    print("Running PCA...")
    embeddings['PCA'] = PCA(n_components=2, random_state=SEED).fit_transform(X_flat)
    methods.append('PCA')

    print("Running t-SNE...")
    from sklearn.manifold import TSNE
    embeddings['t-SNE'] = TSNE(n_components=2, random_state=SEED, n_jobs=-1, verbose=0).fit_transform(X_flat)
    methods.append('t-SNE')

    print("Running UMAP...")
    import umap
    embeddings['UMAP'] = umap.UMAP(n_components=2, random_state=SEED, n_jobs=-1, verbose=0).fit_transform(X_flat)
    methods.append('UMAP')

    print("Running PHATE...")
    embeddings['PHATE'] = phate.PHATE(n_components=2, t='auto', knn=5, decay=40,
                                       n_jobs=-1, verbose=0, random_state=SEED).fit_transform(X_flat)
    methods.append('PHATE')

    print("Running T-PHATE...")
    embeddings['T-PHATE'] = tphate.TPHATE(n_components=2, t='auto', knn=5, decay=40,
                                           n_jobs=-1, verbose=0, random_state=SEED).fit_transform(X_flat)
    methods.append('T-PHATE')

    for m in [1, 4]:
        p = f'{out_dir}/BCNE_m{m}_embedding.npy'
        if os.path.exists(p):
            embeddings[f'BCNE m{m}'] = np.load(p)
            methods.append(f'BCNE m{m}')

    n_methods = len(methods)

    # --- Evaluate ---
    print("\nMetrics:")
    results = {}
    for m in methods:
        emb = embeddings[m]
        cond = knn_accuracy(emb, labels_cond_full, k=8)
        results[m] = {'knn_cond_k8': cond}
        print(f"  {m:12s}  cond={cond:.4f}")

    for m in methods:
        np.save(f'{out_dir}/{m.replace(" ", "_")}_embedding.npy', embeddings[m])
    pd.DataFrame([{'method': m, **v} for m, v in results.items()]).to_csv(
        f'{out_dir}/comparison_metrics.csv', index=False)


    fig = plt.figure(figsize=(4.2 * n_methods + 2, 26))
    gs_main = gridspec.GridSpec(5, 1, height_ratios=[0.6, 1, 1, 1, 0.8],
                                 hspace=0.28, left=0.06, right=0.94, top=0.95, bottom=0.05)

    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 13, 'font.family': 'sans-serif'})

    # ---- Panel labels ----
    panel_labels = ['a', 'b', 'c', 'd', 'e']
    panel_y = [0.96, 0.82, 0.66, 0.50, 0.34]
    for lbl, y in zip(panel_labels, panel_y):
        fig.text(0.01, y, lbl, fontsize=18, fontweight='bold', va='top')

    # ================================================================
    # 20 Group-Averaged Waveforms (10 continuous levels per condition)
    # ================================================================
    gs_a = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_main[0],
                                            wspace=0.35, width_ratios=[1, 1, 0.6])

    # Load per-group waveforms
    group_waveforms = np.load(f'{out_dir}/erp_group_waveforms.npy')  # (20, 60, 32)
    group_keys = np.load(f'{out_dir}/erp_group_keys.npy', allow_pickle=True)  # (20, 2)

    # Pick representative channel (Fz for N400 condition effect)
    ch_name = 'Fz'
    ch_idx = list(channel_names).index(ch_name) if ch_name in channel_names else 31

    # ERP component times
    comp_times = {'P100': 100, 'N170': 170, 'P300': 300, 'N400': 400}

    # Colormap for continuous levels (-5 to +5)
    cont_cmap = plt.cm.get_cmap('viridis')

    # Compute shared y-axis range across all 20 groups
    all_waveforms_ch = group_waveforms[:, :, ch_idx]
    y_min = np.floor(all_waveforms_ch.min()) - 0.5
    y_max = np.ceil(all_waveforms_ch.max()) + 0.5

    # Shared continuous norm across both conditions
    all_conts = [float(group_keys[i][1]) for i in range(len(group_keys))]
    cont_norm = mcolors.Normalize(vmin=min(all_conts), vmax=max(all_conts))

    # Panel 1: Car condition (10 lines, continuous level -5 → +5)
    ax_car = fig.add_subplot(gs_a[0])
    car_indices = [i for i, key in enumerate(group_keys) if str(key[0]) == 'car']

    for idx in car_indices:
        waveform = group_waveforms[idx, :, ch_idx]
        cont_val = float(group_keys[idx][1])
        color = cont_cmap(cont_norm(cont_val))
        ax_car.plot(erp_times, waveform, color=color, lw=1.5, alpha=0.8)

    ax_car.set_ylim(y_min, y_max)
    ax_car.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
    for comp_name, ct in comp_times.items():
        if ct <= erp_times.max():
            ax_car.axvline(ct, color='gray', lw=0.7, ls=':', alpha=0.5)
            ax_car.text(ct, y_max - (y_max - y_min) * 0.05, comp_name,
                        fontsize=7, ha='center', color='gray')

    ax_car.set_title(f'{cond_name_0} (channel {ch_name})', fontweight='bold', fontsize=12)
    ax_car.set_xlabel('Time (ms)', fontsize=10)
    ax_car.set_ylabel('Amplitude', fontsize=10)
    ax_car.spines['top'].set_visible(False)
    ax_car.spines['right'].set_visible(False)

    # Panel 2: Face condition (10 lines, continuous level -5 → +5)
    ax_face = fig.add_subplot(gs_a[1])
    face_indices = [i for i, key in enumerate(group_keys) if str(key[0]) == 'face']

    for idx in face_indices:
        waveform = group_waveforms[idx, :, ch_idx]
        cont_val = float(group_keys[idx][1])
        color = cont_cmap(cont_norm(cont_val))
        ax_face.plot(erp_times, waveform, color=color, lw=1.5, alpha=0.8)

    ax_face.set_ylim(y_min, y_max)
    ax_face.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
    for comp_name, ct in comp_times.items():
        if ct <= erp_times.max():
            ax_face.axvline(ct, color='gray', lw=0.7, ls=':', alpha=0.5)

    ax_face.set_title(f'{cond_name_1} (channel {ch_name})', fontweight='bold', fontsize=12)
    ax_face.set_xlabel('Time (ms)', fontsize=10)
    ax_face.spines['top'].set_visible(False)
    ax_face.spines['right'].set_visible(False)

    # Colorbar for continuous levels
    cbar_ax = fig.add_axes([0.92, 0.88, 0.015, 0.08])
    sm = plt.cm.ScalarMappable(cmap=cont_cmap, norm=cont_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Continuous', fontsize=9)

    # Info text panel
    ax_info = fig.add_subplot(gs_a[2])
    ax_info.axis('off')
    info_text = (
        f"ERP Dataset\n"
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"Channels: 32\n"
        f"Timepoints: {n_tp}\n"
        f"Groups: {n_groups}\n"
        f"  (2 cond \u00d7 10 levels)\n"
        f"Trials/group: 100\n"
        f"Components:\n"
        f"  P100, N170, P300, N400"
    )
    ax_info.text(0.1, 0.9, info_text, transform=ax_info.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', alpha=0.8))

    # ================================================================
    # 2D scatter — CONDITION colors (Car blue, Face red)
    # ================================================================
    gs_b = gridspec.GridSpecFromSubplotSpec(1, n_methods, subplot_spec=gs_main[1], wspace=0.08)

    car_mask = labels_cond_full == 0
    face_mask = labels_cond_full == 1
    tp_labels = np.tile(np.arange(n_tp), n_groups)

    time_norm = mcolors.Normalize(vmin=0, vmax=n_tp - 1)
    cond_colors = ['#1f77b4', '#d62728']  # blue for car, red for face

    for col, method in enumerate(methods):
        ax = fig.add_subplot(gs_b[col])
        emb = embeddings[method]
        n_emb = len(emb)

        # Plot car points (circles) in blue
        car_idx = np.where(car_mask[:n_emb])[0]
        ax.scatter(emb[car_idx, 0], emb[car_idx, 1],
                  c=cond_colors[0], s=15, marker='o', alpha=0.7,
                  rasterized=True, zorder=1, label=cond_name_0)

        # Plot face points (squares) in red
        face_idx = np.where(face_mask[:n_emb])[0]
        ax.scatter(emb[face_idx, 0], emb[face_idx, 1],
                  c=cond_colors[1], s=15, marker='s', alpha=0.7,
                  rasterized=True, zorder=1, label=cond_name_1)

        ax.set_title(method, fontweight='bold', fontsize=13, pad=6)
        if col == 0:
            ax.set_ylabel('Condition', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.3); sp.set_color('#cccccc')

    # ================================================================
    # 2D scatter — TIME progression for CAR (plasma, early→late)
    # ================================================================
    gs_c = gridspec.GridSpecFromSubplotSpec(1, n_methods, subplot_spec=gs_main[2], wspace=0.08)

    for col, method in enumerate(methods):
        ax = fig.add_subplot(gs_c[col])
        emb = embeddings[method]
        n_emb = len(emb)

        # Plot car points (circles) colored by time
        car_idx = np.where(car_mask[:n_emb])[0]
        ax.scatter(emb[car_idx, 0], emb[car_idx, 1],
                  c=tp_labels[car_idx], cmap='plasma', norm=time_norm,
                  s=15, marker='o', alpha=0.8, edgecolors='none',
                  rasterized=True, zorder=1)

        if col == 0:
            ax.set_ylabel(cond_name_0, fontsize=12, fontweight='bold', color='#1f77b4')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.3); sp.set_color('#cccccc')

    # Colorbar for car time
    cbar_ax = fig.add_axes([0.955, 0.44, 0.010, 0.14])
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=time_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Time (ms)', fontsize=9)
    cbar.set_ticks([0, 15, 30, 45, 59])
    cbar.set_ticklabels(['0', '150', '300', '450', '590'])

    # ================================================================
    #  TIME progression for FACE (plasma, early→late)
    # ================================================================
    gs_d = gridspec.GridSpecFromSubplotSpec(1, n_methods, subplot_spec=gs_main[3], wspace=0.08)

    for col, method in enumerate(methods):
        ax = fig.add_subplot(gs_d[col])
        emb = embeddings[method]
        n_emb = len(emb)

        # Plot face points (squares) colored by time
        face_idx = np.where(face_mask[:n_emb])[0]
        ax.scatter(emb[face_idx, 0], emb[face_idx, 1],
                  c=tp_labels[face_idx], cmap='plasma', norm=time_norm,
                  s=15, marker='s', alpha=0.8, edgecolors='none',
                  rasterized=True, zorder=1)

        if col == 0:
            ax.set_ylabel(cond_name_1, fontsize=12, fontweight='bold', color='#d62728')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.3); sp.set_color('#cccccc')

    # Colorbar for face time
    cbar_ax = fig.add_axes([0.955, 0.25, 0.010, 0.14])
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=time_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Time (ms)', fontsize=9)
    cbar.set_ticks([0, 15, 30, 45, 59])
    cbar.set_ticklabels(['0', '150', '300', '450', '590'])

    # ================================================================
    # Method Parameters Table
    # ================================================================
    gs_e = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_main[4])
    ax_e = fig.add_subplot(gs_e[0])
    ax_e.axis('off')

    param_data = [
        ['Method', 'Key Parameters', 'KNN Acc (k=8)'],
        ['PCA', 'n_components=2', f"{results['PCA']['knn_cond_k8']:.3f}"],
        ['t-SNE', 'perplexity=30, random_state', f"{results['t-SNE']['knn_cond_k8']:.3f}"],
        ['UMAP', 'n_neighbors=15, min_dist=0.1', f"{results['UMAP']['knn_cond_k8']:.3f}"],
        ['PHATE', 'knn=5, decay=40, t=auto', f"{results['PHATE']['knn_cond_k8']:.3f}"],
        ['T-PHATE', 'knn=5, decay=40, t=auto', f"{results['T-PHATE']['knn_cond_k8']:.3f}"],
        ['BCNE m1', 'recur=1, perp=30, bal=6', f"{results['BCNE m1']['knn_cond_k8']:.3f}"],
        ['BCNE m4', 'recur=4, perp=30, bal=6', f"{results['BCNE m4']['knn_cond_k8']:.3f}"],
    ]

    table = ax_e.table(cellText=param_data, cellLoc='center', loc='center',
                       colWidths=[0.15, 0.50, 0.20], bbox=[0.05, 0.1, 0.9, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#1f77b4')
        table[(0, i)].set_text_props(weight='bold', color='white', ha='center')

    # Alternate row colors and borders
    for i in range(1, len(param_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#e8f4f8')
            else:
                table[(i, j)].set_facecolor('#ffffff')
            table[(i, j)].set_edgecolor('#cccccc')
            table[(i, j)].set_linewidth(0.5)
            if j == 0:
                table[(i, j)].set_text_props(weight='bold', ha='center')
            else:
                table[(i, j)].set_text_props(ha='center')

    # ---- Title ----
    fig.suptitle('ERP Temporal Dimensionality Reduction',
                 fontsize=18, fontweight='bold', y=0.975)

    fig_path = f'{out_dir}/compare_algorithms.png'
    plt.savefig(fig_path, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nFigure saved: {fig_path}")
    print(f"Metrics: {out_dir}/comparison_metrics.csv")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    if '--extract-bcne' in sys.argv:
        extract_bcne()
    elif '--plot' in sys.argv:
        plot()
    else:
        import subprocess
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script = os.path.join(script_dir, 'visualize_algorithms.py')

        print("Phase 1: Extracting BCNE embeddings (base env)")
        r1 = subprocess.run([sys.executable, script, '--extract-bcne'], cwd=script_dir)
        if r1.returncode != 0:
            sys.exit(1)

        print("\nPhase 2: Running PCA/PHATE/T-PHATE + paper figure (TDR env)")
        r2 = subprocess.run(['conda', 'run', '-n', 'TDR', 'python', script, '--plot'], cwd=script_dir)
        if r2.returncode != 0:
            sys.exit(1)

        print("\nDone!")
