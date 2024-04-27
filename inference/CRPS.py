import numpy as np

def crps(y_true, y_pred, sample_weight=None):
    """Calculate Continuous Ranked Probability Score
    Data based on size (N_samples, N_features)
    Args:
        * y_true : np.array (N_samples, N_features) ground truth
        * y_pred : np.array (N_ensemble, N_samples, N_features) predictions from N_ensemble members
        * sample_weight : np.array (N_samples) weighting for samples e.g., area weighting
    Returns:
        * CRPS : np.array (N_features)
    """
    num_samples = y_pred.shape[0]
    y_pred = np.sort(y_pred, axis=0)
    diff = y_pred[1:] - y_pred[:-1]
    weight = np.arange(1, num_samples) * np.arange(num_samples - 1, 0, -1)
    weight = np.expand_dims(weight, (-2,-1))
    y_true = np.expand_dims(y_true, 0)
    absolute_error = np.mean(np.abs(y_pred - y_true), axis=(0))
    per_obs_crps = absolute_error - np.sum(diff * weight, axis=0) / num_samples**2
    return np.average(per_obs_crps, axis=0, weights=sample_weight)