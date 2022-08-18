import numpy as np
import pandas as pd

import scipy.stats

def error_windows_generator(sequence, h):

    windows = []

    for i in range(len(sequence) - h):
        windows.append(sequence[i:i+h+1])
    
    return np.array(windows)

def error_moving_average(error_windows, w):
    
    moving_windows = error_windows[:, -w:]
    moving_windows_mean = np.mean(moving_windows, axis=1)

    return moving_windows_mean

def ewma(error_windows, alpha):
    ewma_error_windows = np.zeros((error_windows.shape))
    for i in range(error_windows.shape[0]):
        for j in range(error_windows.shape[1]):
            if j == 0:
                ewma_error_windows[i,j] = error_windows[i, j]
            else:
                ewma_error_windows[i,j] = alpha*error_windows[i, j] + (1-alpha)*ewma_error_windows[i,j-1]
    return ewma_error_windows

def gaussian_tail_rule(timestamp, y, y_pred, error, y_label, h=1000, w=10, threshold=0.99):

    timestamp = list(timestamp)
    y = list(y)
    y_pred = list(y_pred)
    error = list(error)
    y_label = list(y_label)

    error_windows = error_windows_generator(error, h)

    past_errors = error_windows[:, :-1]
    moving_average_errors = error_moving_average(error_windows, w)

    past_errors_mean = np.mean(past_errors, axis=1)
    past_errors_std = np.std(past_errors, axis=1)
    z_score_actual_errors = (moving_average_errors - past_errors_mean) / past_errors_std
    fault_scores = 1 - scipy.stats.norm.sf(np.abs(z_score_actual_errors))
    y_pred_label = np.where(fault_scores > threshold, 1, 0)

    df_detection = pd.DataFrame({'timestamp': timestamp[h:],
                'y': y[h:],
                'y_pred': y_pred[h:],
                'fault_score': fault_scores,
                'y_label': y_label[h:],
                'y_label_pred': y_pred_label})

    return df_detection

def threshold_calc(window):

    mu = np.mean(window)
    sigma = np.std(window)
    z_options = np.arange(0.5, 10.5, 0.5)

    argmax = -9999
    threshold = 0
    for z in z_options:

        eps = mu + z*sigma
        delta_mu = mu - np.mean(window[window < eps])
        delta_sigma = sigma - np.std(window[window < eps])
        e_a = len(window[window > eps])
        e_a_bool = window > eps

        # E_seq calculation
        seq_flag = 0
        seq_counter = 0
        for i in e_a_bool:
            if i == 0:
                if seq_flag >= 2:
                    seq_counter += 1
                seq_flag = 0
            else:
                seq_flag += i
        if seq_flag >= 2:
            seq_counter += 1

        E_seq = seq_counter

        arg = ((delta_mu / mu) + (delta_sigma / sigma)) / (e_a + E_seq**2) 

        if arg > argmax:
            argmax = arg
            threshold = eps

    return threshold

def dynamic_error_threshold(timestamp, y, y_pred, error, y_label, h=1000, alpha=0.1):

    timestamp = list(timestamp)
    y = list(y)
    y_pred = list(y_pred)
    error = list(error)
    y_label = list(y_label)

    error_windows = error_windows_generator(error, h)
    errors = error_windows[:, -1] # e_t
    print(f'Error windows shape: {error_windows.shape}')

    ewma_error_windows = ewma(error_windows, alpha)
    print(f'EWMA error windows shape: {ewma_error_windows.shape}')

    thresholds = np.apply_along_axis(threshold_calc, 1, ewma_error_windows)
    print(f'Thresholds shape: {thresholds.shape}')

    y_pred_label = np.where(errors > thresholds, 1, 0)

    df_detection = pd.DataFrame({'timestamp': timestamp[h:],
                'y': y[h:],
                'y_pred': y_pred[h:],
                'threshold': thresholds,
                'y_label': y_label[h:],
                'y_label_pred': y_pred_label})

    return df_detection

def mahalanobis_distance_rule(timestamp, y, y_samples, y_label, z_threshold=2):

    y = np.squeeze(y, axis=-1) # deve ter rank 1 para funcionar

    y_samples_mean = y_samples.mean(axis=1)
    y_samples_std = y_samples.std(axis=1)

    # fault score computation
    z_score_2 = ((y - y_samples_mean)**2) / (y_samples_std**2)
    y_pred_label = np.where(z_score_2 > z_threshold**2, 1, 0)

    df_detection = pd.DataFrame({'timestamp': timestamp,
                'y': y,
                'y_pred_mean': y_samples_mean,
                'y_pred_std': y_samples_std,
                'fault_score': z_score_2,
                'y_label': y_label,
                'y_label_pred': y_pred_label})

    return df_detection