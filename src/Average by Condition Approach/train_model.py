"""
Training pipeline for ERP EEG BCNE model.
Recursive CNN training with KL divergence loss (t-SNE style).
"""
import os
import numpy as np
import random
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from data_preprocess import erp_data_preprocess
from recursiveBCN_utils import (create_model, create_kl_divergence,
                                 calculate_low_para_for_input, calculate_low_para_for_layer)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
my_seed = 0
np.random.seed(my_seed)
random.seed(my_seed)
tf.random.set_seed(my_seed)
tf.config.experimental_run_functions_eagerly(True)

# Dense layer names for recursive stages 2-4
LAYER_NAMES = ['Dense1', 'Dense2', 'Dense3']


def train_stage(model, X_train, out_path, low_para_func,
                epochs, n, batch_size, HD_type, stage_name,
                patience_threshold=None):
    """
    Train one recursive stage.
    If patience_threshold is set, uses early stopping. Otherwise runs fixed epochs.
    """
    batch_num = int(n // batch_size)
    loss_temp = 100
    patience = 0

    pbar = tqdm(range(epochs), desc=stage_name, unit="epoch", ncols=120, ascii=True)
    for epoch in pbar:
        if patience_threshold and patience >= patience_threshold:
            pbar.set_postfix_str(f"Early stop | best_loss={loss_temp/batch_num:.6f}")
            pbar.close()
            print(f"  Early stopping at epoch {epoch} (patience={patience_threshold})")
            break
        if epoch == 0:
            low_para = low_para_func(model, X_train, n, batch_size, HD_type)
        loss = 0
        for i in range(0, n, batch_size):
            loss += model.train_on_batch(X_train[i:i + batch_size], low_para[i // batch_size])
        avg_loss = loss / batch_num

        if patience_threshold:
            if loss < loss_temp:
                loss_temp = loss
                patience = 0
                model.save(out_path)
                pbar.set_postfix_str(f"KL={avg_loss:.6f} | best={loss_temp/batch_num:.6f} | pat=0 *saved*")
            else:
                patience += 1
                pbar.set_postfix_str(f"KL={avg_loss:.6f} | best={loss_temp/batch_num:.6f} | pat={patience}/{patience_threshold}")
        else:
            pbar.set_postfix_str(f"KL={avg_loss:.6f}")

    if not patience_threshold:
        model.save(out_path)
    return model


def run_recursive_stages(model, X_train, out_paths, kl_divergence_loss,
                         recur, n, batch_size, HD_type,
                         epochs, patience_threshold=None, perplexity=None):
    """
    Run all recursive stages (up to 4).
    Stage 1: P computed from raw input.
    Stages 2-4: P computed from Dense1/Dense2/Dense3 layer activations.
    """
    stage_names = ["Recursion 1 (Input)", "Recursion 2 (Dense1)",
                   "Recursion 3 (Dense2)", "Recursion 4 (Dense3)"]

    if isinstance(epochs, int):
        epochs_list = [epochs] * 4
    else:
        epochs_list = epochs

    # Stage 1: from input
    print(f"\n--- {stage_names[0]} ---")
    input_func = lambda m, X, n, bs, hd: calculate_low_para_for_input(m, X, n, bs, hd, perplexity=perplexity)
    model = train_stage(model, X_train, out_paths[0], input_func,
                        epochs_list[0], n, batch_size, HD_type, stage_names[0],
                        patience_threshold=patience_threshold)

    # Stages 2-4: from intermediate dense layers
    for stage_idx in range(1, min(recur, 4)):
        print(f"\n--- {stage_names[stage_idx]} ---")
        model = load_model(out_paths[stage_idx - 1], custom_objects={'KLdivergence': kl_divergence_loss})
        model.compile(loss=kl_divergence_loss, optimizer=Adam(learning_rate=0.0005))
        layer_name = LAYER_NAMES[stage_idx - 1]
        layer_func = lambda m, X, n, bs, hd, ln=layer_name: calculate_low_para_for_layer(m, X, ln, n, bs, hd, perplexity=perplexity)
        model = train_stage(model, X_train, out_paths[stage_idx], layer_func,
                            epochs_list[stage_idx], n, batch_size, HD_type, stage_names[stage_idx],
                            patience_threshold=patience_threshold)

    return model


def main_erp(n_components=2, recur=4, balance=4, train_mode=0,
             csv_path='data/NonLinearBCNE_22march_noise4_2.csv',
             model_design=None, output_dir=None,
             perplexity=None, normalization='global'):
    """
    Train BCNE on ERP EEG data.
    20 groups (Condition x Continuous).

    Returns (model, output_dir)
    """
    if model_design is None:
        model_design = [4, [3, 16, 32, 64], (1024, 512, 256, 8), n_components]

    HD_type = 'erp'

    if output_dir is None:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_dir = f'results/erp/{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    perp_str = perplexity if perplexity else 'default(10)'
    print(f"\n{'='*60}")
    print(f"ERP Training — 20 groups, recur={recur}, perplexity={perp_str}, norm={normalization}")
    print(f"Results: {output_dir}")
    print(f"{'='*60}")

    X_train_proj, _, _, n, _ = erp_data_preprocess(
        csv_path, balance, normalization=normalization)

    batch_size = n // balance
    input_shape = (6, 6, 1)
    model = create_model(
        input_shape=input_shape,
        num_conv_layers=model_design[0],
        filters_list=model_design[1],
        kernel_size=3,
        alpha=0.05,
        dense_units=model_design[2],
        final_units=model_design[3]
    )

    kl_divergence_loss = create_kl_divergence(batch_size, n_components)
    model.compile(loss=kl_divergence_loss, optimizer=Adam(learning_rate=0.0005))

    out_paths = [f'{output_dir}/m{i}.h5' for i in range(1, 5)]

    if train_mode == 0:
        model = run_recursive_stages(model, X_train_proj, out_paths, kl_divergence_loss,
                                     recur, n, batch_size, HD_type,
                                     epochs=200, patience_threshold=50,
                                     perplexity=perplexity)
    else:
        model = run_recursive_stages(model, X_train_proj, out_paths, kl_divergence_loss,
                                     recur, n, batch_size, HD_type,
                                     epochs=[150, 100, 50, 50],
                                     perplexity=perplexity)

    print(f"Training complete. Models saved in: {output_dir}")
    return model, output_dir
