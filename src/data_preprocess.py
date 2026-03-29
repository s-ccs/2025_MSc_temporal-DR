import numpy as np
import pandas as pd
from temporal_pro import TimeCORR
from sklearn.feature_selection import VarianceThreshold
from spatial_pro import construct_neuromap, construct_neuromap_with_projmat, apply_neuromap


def spatiotemporal_projection(X_train, rowNum, colNum):
    autocorr_map = TimeCORR(X=X_train, smooth_window=1)
    X_train_auto = np.matmul(autocorr_map, X_train)

    nump = rowNum * colNum
    if nump < X_train_auto.shape[1]:
        selector = VarianceThreshold()
        var_threshold = selector.fit(X_train_auto)
        top_n_indices = var_threshold.get_support(indices=True)
        X_train_auto = X_train_auto[:, top_n_indices[:nump]]

    NeuroMaps = construct_neuromap(X_train_auto, rowNum, colNum, epsilon=0.0, num_iter=200)
    return NeuroMaps


def erp_data_preprocess(csv_path, balance_degree=4, normalization='per_group'):
    df = pd.read_csv(csv_path)
    meta_cols = ['Time_ms', 'Trial_ID', 'Condition', 'Continuous']
    channel_cols = [c for c in df.columns if c not in meta_cols]
    n_timepoints = df[df['Trial_ID'] == df['Trial_ID'].iloc[0]].shape[0]

    trial_meta = df.groupby('Trial_ID').first()[['Condition', 'Continuous']].reset_index()
    groups = trial_meta.groupby(['Condition', 'Continuous']).groups
    sorted_keys = sorted(groups.keys(), key=lambda x: (0 if x[0] == 'car' else 1, x[1]))

    proj_size = 6
    labels_condition, labels_continuous = [], []
    avg_vector_proj = None

    for gidx, key in enumerate(sorted_keys):
        cond, cont = key
        trial_id_vals = trial_meta.loc[groups[key].tolist(), 'Trial_ID'].values

        trial_data_list = [df[df['Trial_ID'] == tid].sort_values('Time_ms')[channel_cols].values
                           for tid in trial_id_vals]
        avg_vector = np.mean(trial_data_list, axis=0)

        avg_vector_st = spatiotemporal_projection(avg_vector, proj_size, proj_size)

        avg_vector_proj = avg_vector_st if avg_vector_proj is None else np.concatenate((avg_vector_proj, avg_vector_st), axis=0)
        labels_condition.append(0 if cond == 'car' else 1)
        labels_continuous.append(cont)

    X_train_proj = np.array(avg_vector_proj)
    labels_condition = np.array(labels_condition)
    labels_continuous = np.array(labels_continuous)

    if normalization == 'per_group':
        for i in range(len(labels_condition)):
            idx = slice(i * n_timepoints, (i + 1) * n_timepoints)
            bmin, bmax = X_train_proj[idx].min(), X_train_proj[idx].max()
            if bmax - bmin > 1e-12:
                X_train_proj[idx] = (X_train_proj[idx] - bmin) / (bmax - bmin)
    elif normalization == 'global':
        xmin, xmax = X_train_proj.min(), X_train_proj.max()
        X_train_proj = (X_train_proj - xmin) / (xmax - xmin)

    return X_train_proj, labels_condition, labels_continuous, X_train_proj.shape[0], n_timepoints


def compute_fixed_projection(csv_path, rowNum=6, colNum=6):
    df = pd.read_csv(csv_path)
    channel_cols = [c for c in df.columns if c not in ['Time_ms', 'Trial_ID', 'Condition', 'Continuous']]

    all_trials = [tdf.sort_values('Time_ms')[channel_cols].values
                  for _, tdf in df.groupby('Trial_ID')]
    global_avg = np.mean(all_trials, axis=0)

    autocorr_map = TimeCORR(X=global_avg, smooth_window=1)
    X_smooth = np.matmul(autocorr_map, global_avg)

    nump = rowNum * colNum
    selector = VarianceThreshold()
    selector.fit(X_smooth)
    var_indices = selector.get_support(indices=True)[:nump]
    X_smooth = X_smooth[:, var_indices]

    _, projMat = construct_neuromap_with_projmat(X_smooth, rowNum, colNum, epsilon=0.0, num_iter=200)
    return projMat, var_indices


def erp_data_preprocess_compare(csv_path):
    df = pd.read_csv(csv_path)
    channel_cols = [c for c in df.columns if c not in ['Time_ms', 'Trial_ID', 'Condition', 'Continuous']]
    n_timepoints = df[df['Trial_ID'] == df['Trial_ID'].iloc[0]].shape[0]

    trial_meta = df.groupby('Trial_ID').first()[['Condition', 'Continuous']].reset_index()
    groups = trial_meta.groupby(['Condition', 'Continuous']).groups
    sorted_keys = sorted(groups.keys(), key=lambda x: (0 if x[0] == 'car' else 1, x[1]))

    labels_cond, labels_cont, avg_data_all = [], [], []

    for key in sorted_keys:
        trial_id_vals = trial_meta.loc[groups[key].tolist(), 'Trial_ID'].values
        trial_data_list = [df[df['Trial_ID'] == tid].sort_values('Time_ms')[channel_cols].values
                           for tid in trial_id_vals]
        avg_data_all.append(np.mean(trial_data_list, axis=0))
        labels_cond.append(0 if key[0] == 'car' else 1)
        labels_cont.append(key[1])

    X_train = np.concatenate(avg_data_all, axis=0)
    xmin, xmax = X_train.min(), X_train.max()
    X_train = (X_train - xmin) / (xmax - xmin)

    return X_train, np.array(labels_cond), np.array(labels_cont), X_train.shape[0], n_timepoints
