"""
Fairness metrics for evaluating machine learning models.
This module implements various fairness metrics and information-theoretic measures.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, 
    precision_score, recall_score, f1_score
)
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def demographic_parity_difference(y_pred: np.ndarray, 
                                   sensitive: np.ndarray,
                                   favorable_label: int = 1) -> float:
    """
    Calculate Demographic Parity Difference (DPD).
    
    DPD = |P(Y_hat=1 | S=0) - P(Y_hat=1 | S=1)|
    
    Parameters:
    -----------
    y_pred : np.ndarray
        Predicted labels
    sensitive : np.ndarray
        Sensitive attribute values (binary)
    favorable_label : int
        The favorable/positive label
    
    Returns:
    --------
    float : Demographic parity difference (0 = perfect parity)
    """
    mask_0 = sensitive == 0
    mask_1 = sensitive == 1
    
    rate_0 = np.mean(y_pred[mask_0] == favorable_label) if mask_0.sum() > 0 else 0
    rate_1 = np.mean(y_pred[mask_1] == favorable_label) if mask_1.sum() > 0 else 0
    
    return abs(rate_0 - rate_1)


def demographic_parity_ratio(y_pred: np.ndarray,
                             sensitive: np.ndarray,
                             favorable_label: int = 1) -> float:
    """
    Calculate Demographic Parity Ratio (Disparate Impact Ratio).
    
    DPR = min(P(Y_hat=1|S=0)/P(Y_hat=1|S=1), P(Y_hat=1|S=1)/P(Y_hat=1|S=0))
    
    Parameters:
    -----------
    y_pred : np.ndarray
        Predicted labels
    sensitive : np.ndarray
        Sensitive attribute values (binary)
    favorable_label : int
        The favorable/positive label
    
    Returns:
    --------
    float : Demographic parity ratio (1 = perfect parity)
    """
    mask_0 = sensitive == 0
    mask_1 = sensitive == 1
    
    rate_0 = np.mean(y_pred[mask_0] == favorable_label) if mask_0.sum() > 0 else 0
    rate_1 = np.mean(y_pred[mask_1] == favorable_label) if mask_1.sum() > 0 else 0
    
    if rate_0 == 0 or rate_1 == 0:
        return 0.0
    
    return min(rate_0 / rate_1, rate_1 / rate_0)


def equalized_odds_difference(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               sensitive: np.ndarray) -> Dict[str, float]:
    """
    Calculate Equalized Odds Difference.
    
    Measures the difference in TPR and FPR across groups.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    sensitive : np.ndarray
        Sensitive attribute values (binary)
    
    Returns:
    --------
    Dict containing TPR difference, FPR difference, and average
    """
    mask_0 = sensitive == 0
    mask_1 = sensitive == 1
    
    # Calculate TPR for each group
    tpr_0 = recall_score(y_true[mask_0], y_pred[mask_0], zero_division=0)
    tpr_1 = recall_score(y_true[mask_1], y_pred[mask_1], zero_division=0)
    
    # Calculate FPR for each group
    tn_0, fp_0, _, _ = confusion_matrix(y_true[mask_0], y_pred[mask_0], labels=[0, 1]).ravel() if mask_0.sum() > 0 else (0, 0, 0, 0)
    tn_1, fp_1, _, _ = confusion_matrix(y_true[mask_1], y_pred[mask_1], labels=[0, 1]).ravel() if mask_1.sum() > 0 else (0, 0, 0, 0)
    
    fpr_0 = fp_0 / (fp_0 + tn_0) if (fp_0 + tn_0) > 0 else 0
    fpr_1 = fp_1 / (fp_1 + tn_1) if (fp_1 + tn_1) > 0 else 0
    
    tpr_diff = abs(tpr_0 - tpr_1)
    fpr_diff = abs(fpr_0 - fpr_1)
    
    return {
        'tpr_difference': tpr_diff,
        'fpr_difference': fpr_diff,
        'equalized_odds_difference': (tpr_diff + fpr_diff) / 2,
        'tpr_group_0': tpr_0,
        'tpr_group_1': tpr_1,
        'fpr_group_0': fpr_0,
        'fpr_group_1': fpr_1
    }


def calibration_difference(y_true: np.ndarray,
                           y_prob: np.ndarray,
                           sensitive: np.ndarray,
                           n_bins: int = 10) -> float:
    """
    Calculate Calibration Difference across groups.
    
    Measures how well-calibrated predictions are within each group.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_prob : np.ndarray
        Predicted probabilities for positive class
    sensitive : np.ndarray
        Sensitive attribute values (binary)
    n_bins : int
        Number of bins for calibration
    
    Returns:
    --------
    float : Average calibration difference across bins
    """
    mask_0 = sensitive == 0
    mask_1 = sensitive == 1
    
    bins = np.linspace(0, 1, n_bins + 1)
    calibration_errors = []
    
    for i in range(n_bins):
        bin_mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        
        bin_mask_0 = bin_mask & mask_0
        bin_mask_1 = bin_mask & mask_1
        
        if bin_mask_0.sum() > 0 and bin_mask_1.sum() > 0:
            calib_0 = y_true[bin_mask_0].mean()
            calib_1 = y_true[bin_mask_1].mean()
            calibration_errors.append(abs(calib_0 - calib_1))
    
    return np.mean(calibration_errors) if calibration_errors else 0.0


def predictive_parity_difference(y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  sensitive: np.ndarray) -> float:
    """
    Calculate Predictive Parity Difference.
    
    Measures the difference in Positive Predictive Value (Precision) across groups.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    sensitive : np.ndarray
        Sensitive attribute values (binary)
    
    Returns:
    --------
    float : Absolute difference in precision between groups
    """
    mask_0 = sensitive == 0
    mask_1 = sensitive == 1
    
    ppv_0 = precision_score(y_true[mask_0], y_pred[mask_0], zero_division=0)
    ppv_1 = precision_score(y_true[mask_1], y_pred[mask_1], zero_division=0)
    
    return abs(ppv_0 - ppv_1)


def compute_all_fairness_metrics(y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  y_prob: Optional[np.ndarray],
                                  sensitive: np.ndarray) -> Dict[str, float]:
    """
    Compute all fairness metrics at once.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_prob : np.ndarray, optional
        Predicted probabilities
    sensitive : np.ndarray
        Sensitive attribute values
    
    Returns:
    --------
    Dict containing all fairness metrics
    """
    eo_metrics = equalized_odds_difference(y_true, y_pred, sensitive)
    
    metrics = {
        'demographic_parity_difference': demographic_parity_difference(y_pred, sensitive),
        'demographic_parity_ratio': demographic_parity_ratio(y_pred, sensitive),
        'predictive_parity_diff': predictive_parity_difference(y_true, y_pred, sensitive),
        'equalized_odds_difference': {
            'tpr_difference': eo_metrics['tpr_difference'],
            'fpr_difference': eo_metrics['fpr_difference'],
            'average': eo_metrics['equalized_odds_difference']
        }
    }
    
    if y_prob is not None:
        metrics['calibration_difference'] = calibration_difference(y_true, y_prob, sensitive)
    
    return metrics


def compute_performance_metrics(y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute standard performance metrics.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_prob : np.ndarray, optional
        Predicted probabilities
    
    Returns:
    --------
    Dict containing performance metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_prob is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['auc_roc'] = 0.5
    
    return metrics


def compute_group_performance(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               sensitive: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Compute performance metrics for each demographic group.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    sensitive : np.ndarray
        Sensitive attribute values
    
    Returns:
    --------
    Dict containing metrics for each group
    """
    results = {}
    
    for group in np.unique(sensitive):
        mask = sensitive == group
        group_metrics = {
            'accuracy': accuracy_score(y_true[mask], y_pred[mask]),
            'precision': precision_score(y_true[mask], y_pred[mask], zero_division=0),
            'recall': recall_score(y_true[mask], y_pred[mask], zero_division=0),
            'f1': f1_score(y_true[mask], y_pred[mask], zero_division=0),
            'positive_rate': np.mean(y_pred[mask]),
            'base_rate': np.mean(y_true[mask])
        }
        results[f'group_{group}'] = group_metrics
    
    return results


def information_loss_metric(original_mi: float, 
                            transformed_mi: float) -> float:
    """
    Calculate relative information loss after transformation.
    
    Parameters:
    -----------
    original_mi : float
        Mutual information before transformation
    transformed_mi : float
        Mutual information after transformation
    
    Returns:
    --------
    float : Relative information loss (0 = no loss, 1 = total loss)
    """
    if original_mi == 0:
        return 0.0
    return max(0, (original_mi - transformed_mi) / original_mi)


def fairness_performance_tradeoff(fairness_metric: float,
                                   performance_metric: float,
                                   fairness_weight: float = 0.5) -> float:
    """
    Compute a combined fairness-performance score.
    
    Parameters:
    -----------
    fairness_metric : float
        Fairness metric (lower is better, e.g., DPD)
    performance_metric : float
        Performance metric (higher is better, e.g., accuracy)
    fairness_weight : float
        Weight for fairness (1 - fairness_weight for performance)
    
    Returns:
    --------
    float : Combined score (higher is better)
    """
    fairness_score = 1 - fairness_metric  # Convert to higher-is-better
    combined = fairness_weight * fairness_score + (1 - fairness_weight) * performance_metric
    return combined
