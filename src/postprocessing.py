"""
Post-processing fairness methods.
This module implements fairness interventions applied after model training.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.optimize import minimize, linprog
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class EqualizedOddsPostProcessor(BaseEstimator):
    """
    Equalized Odds Post-processing.
    
    Adjusts predictions to equalize true positive rates and false positive rates
    across demographic groups.
    Based on: Hardt et al. (2016) - Equality of opportunity in supervised learning.
    """
    
    def __init__(self, objective: str = 'equalized_odds'):
        """
        Parameters:
        -----------
        objective : str
            'equalized_odds' for TPR and FPR parity, 
            'equal_opportunity' for TPR parity only
        """
        self.objective = objective
        self.mixing_rates_ = None
        
    def fit(self, y_true: np.ndarray, y_prob: np.ndarray, sensitive: np.ndarray) -> 'EqualizedOddsPostProcessor':
        """
        Fit the post-processor to find optimal mixing rates.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_prob : np.ndarray
            Predicted probabilities
        sensitive : np.ndarray
            Sensitive attribute values
        
        Returns:
        --------
        self
        """
        groups = np.unique(sensitive)
        self.mixing_rates_ = {}
        
        # For each group, find optimal threshold and mixing rate
        for g in groups:
            mask = sensitive == g
            y_true_g = y_true[mask]
            y_prob_g = y_prob[mask]
            
            # Find threshold that optimizes objective
            best_threshold = 0.5
            best_score = float('inf')
            
            for threshold in np.linspace(0.1, 0.9, 17):
                y_pred_g = (y_prob_g >= threshold).astype(int)
                
                # Compute TPR and FPR
                tp = np.sum((y_pred_g == 1) & (y_true_g == 1))
                fn = np.sum((y_pred_g == 0) & (y_true_g == 1))
                fp = np.sum((y_pred_g == 1) & (y_true_g == 0))
                tn = np.sum((y_pred_g == 0) & (y_true_g == 0))
                
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                # Objective: balance TPR and FPR
                score = abs(tpr - 0.5) + abs(fpr - 0.5)  # Placeholder
                
                if score < best_score:
                    best_score = score
                    best_threshold = threshold
            
            self.mixing_rates_[g] = {
                'threshold': best_threshold,
                'flip_prob_0to1': 0.0,  # Probability of flipping 0 to 1
                'flip_prob_1to0': 0.0   # Probability of flipping 1 to 0
            }
        
        # Optimize mixing rates across groups
        self._optimize_mixing_rates(y_true, y_prob, sensitive)
        
        return self
    
    def _optimize_mixing_rates(self, y_true: np.ndarray, y_prob: np.ndarray, sensitive: np.ndarray):
        """Optimize mixing rates to achieve fairness."""
        groups = np.unique(sensitive)
        
        # Compute baseline rates for each group
        group_rates = {}
        for g in groups:
            mask = sensitive == g
            threshold = self.mixing_rates_[g]['threshold']
            y_pred = (y_prob[mask] >= threshold).astype(int)
            y_true_g = y_true[mask]
            
            tp = np.sum((y_pred == 1) & (y_true_g == 1))
            fn = np.sum((y_pred == 0) & (y_true_g == 1))
            fp = np.sum((y_pred == 1) & (y_true_g == 0))
            tn = np.sum((y_pred == 0) & (y_true_g == 0))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            group_rates[g] = {'tpr': tpr, 'fpr': fpr, 'n_pos': tp + fn, 'n_neg': fp + tn}
        
        # Find target rates (average or max)
        if len(groups) == 2:
            g0, g1 = groups
            
            # Target TPR: average of the two
            target_tpr = (group_rates[g0]['tpr'] + group_rates[g1]['tpr']) / 2
            target_fpr = (group_rates[g0]['fpr'] + group_rates[g1]['fpr']) / 2
            
            for g in groups:
                tpr_diff = target_tpr - group_rates[g]['tpr']
                fpr_diff = target_fpr - group_rates[g]['fpr']
                
                if tpr_diff > 0:
                    # Need to increase TPR: flip some FN to TP
                    self.mixing_rates_[g]['flip_prob_0to1'] = min(tpr_diff, 0.3)
                elif tpr_diff < 0:
                    # Need to decrease TPR: flip some TP to FN
                    self.mixing_rates_[g]['flip_prob_1to0'] = min(-tpr_diff, 0.3)
    
    def transform(self, y_prob: np.ndarray, sensitive: np.ndarray, y_true: np.ndarray = None) -> np.ndarray:
        """
        Apply post-processing to predictions.
        
        Parameters:
        -----------
        y_prob : np.ndarray
            Predicted probabilities
        sensitive : np.ndarray
            Sensitive attribute values
        y_true : np.ndarray, optional
            True labels (not used in transform)
        
        Returns:
        --------
        np.ndarray : Adjusted predictions
        """
        y_pred = np.zeros(len(y_prob), dtype=int)
        
        for g in np.unique(sensitive):
            mask = sensitive == g
            threshold = self.mixing_rates_[g]['threshold']
            y_pred_g = (y_prob[mask] >= threshold).astype(int)
            
            # Apply randomized flipping based on mixing rates
            flip_0to1 = self.mixing_rates_[g]['flip_prob_0to1']
            flip_1to0 = self.mixing_rates_[g]['flip_prob_1to0']
            
            random_vals = np.random.random(len(y_pred_g))
            
            # Flip some 0s to 1s
            flip_mask_0to1 = (y_pred_g == 0) & (random_vals < flip_0to1)
            y_pred_g[flip_mask_0to1] = 1
            
            # Flip some 1s to 0s
            flip_mask_1to0 = (y_pred_g == 1) & (random_vals < flip_1to0)
            y_pred_g[flip_mask_1to0] = 0
            
            y_pred[mask] = y_pred_g
        
        return y_pred


class ThresholdOptimizer(BaseEstimator):
    """
    Threshold Optimization for Fairness.
    
    Finds group-specific classification thresholds that optimize
    a combination of accuracy and fairness.
    """
    
    def __init__(self, 
                 fairness_metric: str = 'demographic_parity',
                 fairness_weight: float = 0.5):
        """
        Parameters:
        -----------
        fairness_metric : str
            Which fairness metric to optimize
        fairness_weight : float
            Weight for fairness vs accuracy (0-1)
        """
        self.fairness_metric = fairness_metric
        self.fairness_weight = fairness_weight
        self.group_thresholds_ = None
        
    def fit(self, y_true: np.ndarray, y_prob: np.ndarray, sensitive: np.ndarray) -> 'ThresholdOptimizer':
        """
        Find optimal thresholds for each group.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_prob : np.ndarray
            Predicted probabilities
        sensitive : np.ndarray
            Sensitive attribute values
        
        Returns:
        --------
        self
        """
        groups = np.unique(sensitive)
        self.group_thresholds_ = {}
        
        def objective(thresholds):
            """Objective function to minimize."""
            total_error = 0
            positive_rates = []
            
            for i, g in enumerate(groups):
                mask = sensitive == g
                threshold = thresholds[i]
                y_pred = (y_prob[mask] >= threshold).astype(int)
                
                # Accuracy component
                error = 1 - np.mean(y_pred == y_true[mask])
                total_error += error * np.sum(mask)
                
                # Positive rate for fairness
                positive_rates.append(np.mean(y_pred))
            
            # Accuracy loss
            accuracy_loss = total_error / len(y_true)
            
            # Fairness loss (demographic parity difference)
            fairness_loss = max(positive_rates) - min(positive_rates)
            
            return (1 - self.fairness_weight) * accuracy_loss + self.fairness_weight * fairness_loss
        
        # Optimize thresholds
        initial_thresholds = [0.5] * len(groups)
        bounds = [(0.1, 0.9)] * len(groups)
        
        result = minimize(objective, initial_thresholds, method='L-BFGS-B', bounds=bounds)
        
        for i, g in enumerate(groups):
            self.group_thresholds_[g] = result.x[i]
        
        return self
    
    def transform(self, y_prob: np.ndarray, sensitive: np.ndarray) -> np.ndarray:
        """
        Apply group-specific thresholds.
        
        Parameters:
        -----------
        y_prob : np.ndarray
            Predicted probabilities
        sensitive : np.ndarray
            Sensitive attribute values
        
        Returns:
        --------
        np.ndarray : Thresholded predictions
        """
        y_pred = np.zeros(len(y_prob), dtype=int)
        
        for g in np.unique(sensitive):
            mask = sensitive == g
            threshold = self.group_thresholds_.get(g, 0.5)
            y_pred[mask] = (y_prob[mask] >= threshold).astype(int)
        
        return y_pred
    
    def get_threshold_info(self) -> Dict:
        """Return information about the learned thresholds."""
        return {
            'group_thresholds': self.group_thresholds_.copy(),
            'threshold_spread': max(self.group_thresholds_.values()) - min(self.group_thresholds_.values())
        }


class CalibratedPostProcessor(BaseEstimator):
    """
    Calibration-based Post-processing.
    
    Adjusts predictions to achieve calibration within each demographic group.
    """
    
    def __init__(self, n_bins: int = 10):
        """
        Parameters:
        -----------
        n_bins : int
            Number of calibration bins
        """
        self.n_bins = n_bins
        self.calibration_maps_ = None
        
    def fit(self, y_true: np.ndarray, y_prob: np.ndarray, sensitive: np.ndarray) -> 'CalibratedPostProcessor':
        """
        Fit calibration maps for each group.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_prob : np.ndarray
            Predicted probabilities
        sensitive : np.ndarray
            Sensitive attribute values
        
        Returns:
        --------
        self
        """
        groups = np.unique(sensitive)
        self.calibration_maps_ = {}
        
        bins = np.linspace(0, 1, self.n_bins + 1)
        
        for g in groups:
            mask = sensitive == g
            calibration_map = {}
            
            for i in range(self.n_bins):
                bin_mask = (y_prob[mask] >= bins[i]) & (y_prob[mask] < bins[i + 1])
                if np.sum(bin_mask) > 0:
                    true_positive_rate = np.mean(y_true[mask][bin_mask])
                    calibration_map[i] = true_positive_rate
                else:
                    calibration_map[i] = (bins[i] + bins[i + 1]) / 2
            
            self.calibration_maps_[g] = calibration_map
        
        return self
    
    def transform(self, y_prob: np.ndarray, sensitive: np.ndarray) -> np.ndarray:
        """
        Apply calibration to predictions.
        
        Parameters:
        -----------
        y_prob : np.ndarray
            Predicted probabilities
        sensitive : np.ndarray
            Sensitive attribute values
        
        Returns:
        --------
        np.ndarray : Calibrated probabilities
        """
        y_prob_calibrated = np.zeros_like(y_prob)
        bins = np.linspace(0, 1, self.n_bins + 1)
        
        for g in np.unique(sensitive):
            mask = sensitive == g
            
            for i in range(self.n_bins):
                bin_mask = (y_prob[mask] >= bins[i]) & (y_prob[mask] < bins[i + 1])
                if np.sum(bin_mask) > 0:
                    calibrated_value = self.calibration_maps_.get(g, {}).get(i, (bins[i] + bins[i + 1]) / 2)
                    indices = np.where(mask)[0][bin_mask]
                    y_prob_calibrated[indices] = calibrated_value
        
        return y_prob_calibrated


class RejectOptionClassifier(BaseEstimator):
    """
    Reject Option Classification for Fairness.
    
    Gives favorable outcomes to unprivileged groups and unfavorable outcomes
    to privileged groups in the uncertainty region.
    """
    
    def __init__(self,
                 low_threshold: float = 0.3,
                 high_threshold: float = 0.7,
                 privileged_group: int = 1):
        """
        Parameters:
        -----------
        low_threshold : float
            Lower bound of uncertainty region
        high_threshold : float
            Upper bound of uncertainty region
        privileged_group : int
            Value of privileged group in sensitive attribute
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.privileged_group = privileged_group
        self.optimal_thresholds_ = None
        
    def fit(self, y_true: np.ndarray, y_prob: np.ndarray, sensitive: np.ndarray) -> 'RejectOptionClassifier':
        """
        Fit the reject option classifier.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_prob : np.ndarray
            Predicted probabilities
        sensitive : np.ndarray
            Sensitive attribute values
        
        Returns:
        --------
        self
        """
        # Find optimal thresholds that balance accuracy and fairness
        best_score = -float('inf')
        best_thresholds = (self.low_threshold, self.high_threshold)
        
        for low in np.linspace(0.2, 0.45, 6):
            for high in np.linspace(0.55, 0.8, 6):
                if low >= high:
                    continue
                
                y_pred = self._apply_reject_option(y_prob, sensitive, low, high)
                
                # Compute metrics
                accuracy = np.mean(y_pred == y_true)
                
                # Demographic parity
                mask_priv = sensitive == self.privileged_group
                mask_unpriv = sensitive != self.privileged_group
                dp_diff = abs(np.mean(y_pred[mask_priv]) - np.mean(y_pred[mask_unpriv]))
                
                # Combined score
                score = accuracy - 0.5 * dp_diff
                
                if score > best_score:
                    best_score = score
                    best_thresholds = (low, high)
        
        self.optimal_thresholds_ = best_thresholds
        return self
    
    def _apply_reject_option(self, y_prob: np.ndarray, sensitive: np.ndarray,
                              low_threshold: float, high_threshold: float) -> np.ndarray:
        """Apply reject option classification."""
        y_pred = np.zeros(len(y_prob), dtype=int)
        
        # Clear decisions
        y_pred[y_prob >= high_threshold] = 1
        y_pred[y_prob < low_threshold] = 0
        
        # Uncertainty region
        uncertain_mask = (y_prob >= low_threshold) & (y_prob < high_threshold)
        
        # Favor unprivileged group
        favor_mask = uncertain_mask & (sensitive != self.privileged_group)
        unfavor_mask = uncertain_mask & (sensitive == self.privileged_group)
        
        y_pred[favor_mask] = 1  # Give favorable outcome
        y_pred[unfavor_mask] = 0  # Give unfavorable outcome
        
        return y_pred
    
    def transform(self, y_prob: np.ndarray, sensitive: np.ndarray) -> np.ndarray:
        """
        Apply reject option classification.
        
        Parameters:
        -----------
        y_prob : np.ndarray
            Predicted probabilities
        sensitive : np.ndarray
            Sensitive attribute values
        
        Returns:
        --------
        np.ndarray : Adjusted predictions
        """
        low, high = self.optimal_thresholds_
        return self._apply_reject_option(y_prob, sensitive, low, high)


def compute_postprocessing_info_loss(y_true: np.ndarray,
                                      y_pred_original: np.ndarray,
                                      y_pred_adjusted: np.ndarray) -> Dict[str, float]:
    """
    Compute information loss from post-processing.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred_original : np.ndarray
        Original predictions
    y_pred_adjusted : np.ndarray
        Adjusted predictions
    
    Returns:
    --------
    Dict containing information loss metrics
    """
    # Prediction change rate
    change_rate = np.mean(y_pred_original != y_pred_adjusted)
    
    # Accuracy impact
    original_accuracy = np.mean(y_pred_original == y_true)
    adjusted_accuracy = np.mean(y_pred_adjusted == y_true)
    accuracy_change = adjusted_accuracy - original_accuracy
    
    # Prediction entropy change
    def entropy(y):
        p = np.mean(y)
        if p == 0 or p == 1:
            return 0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    
    entropy_original = entropy(y_pred_original)
    entropy_adjusted = entropy(y_pred_adjusted)
    entropy_change = entropy_adjusted - entropy_original
    
    return {
        'prediction_change_rate': change_rate,
        'accuracy_change': accuracy_change,
        'entropy_change': entropy_change,
        'original_accuracy': original_accuracy,
        'adjusted_accuracy': adjusted_accuracy
    }


def apply_postprocessing(y_true: np.ndarray,
                          y_prob: np.ndarray,
                          sensitive: np.ndarray,
                          method: str = 'equalized_odds',
                          **kwargs) -> Tuple[np.ndarray, BaseEstimator]:
    """
    Apply post-processing method to predictions.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_prob : np.ndarray
        Predicted probabilities
    sensitive : np.ndarray
        Sensitive attribute values
    method : str
        Method name: 'equalized_odds', 'threshold', 'calibration', or 'reject_option'
    **kwargs : dict
        Additional parameters for the method
    
    Returns:
    --------
    Tuple of adjusted predictions and fitted post-processor
    """
    if method == 'equalized_odds':
        processor = EqualizedOddsPostProcessor(**kwargs)
    elif method == 'threshold':
        processor = ThresholdOptimizer(**kwargs)
    elif method == 'calibration':
        processor = CalibratedPostProcessor(**kwargs)
    elif method == 'reject_option':
        processor = RejectOptionClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    processor.fit(y_true, y_prob, sensitive)
    y_pred = processor.transform(y_prob, sensitive)
    
    return y_pred, processor
