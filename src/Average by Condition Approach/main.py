"""
Main entry point for ERP EEG BCNE pipeline.
Train and test BCNE on ERP data.
20 groups (Condition x Continuous), independent projection, per-group normalization.
"""
import numpy as np
import os
import tensorflow as tf
from train_model import main_erp
from test_model import test_erp

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.config.experimental_run_functions_eagerly(True)
np.seterr(divide='ignore', invalid='ignore')

if __name__ == "__main__":

    CSV_PATH = 'data/eeg_32ch_allERP_15March.csv'

    erp_model, erp_output_dir = main_erp(
        n_components=2, recur=4, balance=4, train_mode=1,
        csv_path=CSV_PATH,
        perplexity=10,
        normalization='global',
    )
    test_erp(balance=4, csv_path=CSV_PATH, output_dir=erp_output_dir,
             normalization='global')
