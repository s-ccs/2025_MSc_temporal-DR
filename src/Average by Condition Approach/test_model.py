"""
Model testing for ERP EEG BCNE pipeline.
Test on grouped training data (20 groups: Condition x Continuous).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.keras.models import load_model
from recursiveBCN_utils import create_kl_divergence
from data_preprocess import erp_data_preprocess

import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)


def test_erp(balance, csv_path, output_dir, normalization='global'):
    """
    Test BCNE ERP models on grouped training data.
    Generates: 2 rows (condition, continuous) x 4 columns (m1-m4).
    """
    X_train_proj, labels_cond, labels_cont, n, n_timepoints = erp_data_preprocess(
        csv_path, balance, normalization=normalization)

    n_groups = len(labels_cond)

    batch_size = n // balance
    kl_divergence_loss = create_kl_divergence(batch_size, 2)

    labels_cond_full = np.repeat(labels_cond, n_timepoints)
    labels_cont_full = np.repeat(labels_cont, n_timepoints)

    # Collect predictions for all available models
    preds = {}
    for model_num in range(1, 5):
        model_path = f'{output_dir}/m{model_num}.h5'
        if not os.path.exists(model_path):
            continue
        model = load_model(model_path, custom_objects={'KLdivergence': kl_divergence_loss})
        preds[model_num] = model.predict(X_train_proj)
        np.save(f'{output_dir}/m{model_num}_embedding.npy', preds[model_num])

    if not preds:
        print("No models found.")
        return

    model_nums = sorted(preds.keys())
    n_models = len(model_nums)

    fig = plt.figure(figsize=(5 * n_models, 10))
    gs = gridspec.GridSpec(2, n_models, hspace=0.25, wspace=0.20)

    for col, m_num in enumerate(model_nums):
        pred = preds[m_num]

        # Row 1: Condition
        ax = fig.add_subplot(gs[0, col])
        car_mask = labels_cond_full == 0
        face_mask = labels_cond_full == 1
        ax.scatter(pred[car_mask, 0], pred[car_mask, 1], c='deepskyblue', s=12, alpha=0.5, label='Car')
        ax.scatter(pred[face_mask, 0], pred[face_mask, 1], c='hotpink', s=12, alpha=0.5, label='Face')
        ax.set_title(f'm{m_num} - Condition', fontweight='bold')
        ax.set_xlabel('BCNE1'); ax.set_ylabel('BCNE2')
        if col == 0:
            ax.legend(fontsize=8, markerscale=2)

        # Row 2: Continuous
        ax = fig.add_subplot(gs[1, col])
        sc = ax.scatter(pred[:, 0], pred[:, 1], c=labels_cont_full, cmap='viridis', s=12, alpha=0.5)
        ax.set_title(f'm{m_num} - Continuous', fontweight='bold')
        ax.set_xlabel('BCNE1'); ax.set_ylabel('BCNE2')
        if col == n_models - 1:
            plt.colorbar(sc, ax=ax, fraction=0.046)

    fig.suptitle(f'BCNE ERP Results ({n_groups} groups, {n_timepoints} timepoints)',
                 fontsize=16, fontweight='bold', y=0.98)
    fig_path = f'{output_dir}/results_combined.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Combined figure saved: {fig_path}")
