"""
Pre-processing fairness methods.
This module implements fairness interventions applied before model training.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Tuple, Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')


class Reweighing(BaseEstimator, TransformerMixin):
    """
    Reweighing algorithm for fair preprocessing.
    
    Assigns weights to training samples to ensure demographic parity in the training data.
    Based on: Kamiran & Calders (2012) - Data preprocessing techniques for classification without discrimination.
    """
    
    def __init__(self, sensitive_attr_idx: int = None):
        """
        Parameters:
        -----------
        sensitive_attr_idx : int
            Index of the sensitive attribute column
        """
        self.sensitive_attr_idx = sensitive_attr_idx
        self.weights_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, sensitive: np.ndarray = None) -> 'Reweighing':
        """
        Compute sample weights for reweighing.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        sensitive : np.ndarray, optional
            Sensitive attribute values. If None, uses sensitive_attr_idx
        
        Returns:
        --------
        self
        """
        if sensitive is None and self.sensitive_attr_idx is not None:
            sensitive = X[:, self.sensitive_attr_idx]
        
        n = len(y)
        self.weights_ = np.ones(n)
        
        # Get unique groups
        groups = np.unique(sensitive)
        labels = np.unique(y)
        
        # Calculate expected and observed probabilities
        for g in groups:
            for l in labels:
                # Expected: P(S=g) * P(Y=l)
                p_g = np.mean(sensitive == g)
                p_l = np.mean(y == l)
                expected = p_g * p_l
                
                # Observed: P(S=g, Y=l)
                mask = (sensitive == g) & (y == l)
                observed = np.mean(mask)
                
                # Weight = Expected / Observed
                if observed > 0:
                    weight = expected / observed
                    self.weights_[mask] = weight
        
        # Normalize weights
        self.weights_ = self.weights_ / np.mean(self.weights_)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Return the original features (transformation happens via weights).
        """
        return X
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray, sensitive: np.ndarray = None) -> np.ndarray:
        """Fit and return transformed data."""
        self.fit(X, y, sensitive)
        return self.transform(X)
    
    def get_weights(self) -> np.ndarray:
        """Return computed sample weights."""
        return self.weights_


class FeatureMasking(BaseEstimator, TransformerMixin):
    """
    Feature masking for fairness.
    
    Removes sensitive attributes and optionally their proxy features.
    """
    
    def __init__(self, 
                 sensitive_indices: List[int] = None,
                 proxy_indices: List[int] = None,
                 remove_proxies: bool = True):
        """
        Parameters:
        -----------
        sensitive_indices : list
            Indices of sensitive attribute columns
        proxy_indices : list
            Indices of proxy feature columns
        remove_proxies : bool
            Whether to also remove proxy features
        """
        self.sensitive_indices = sensitive_indices or []
        self.proxy_indices = proxy_indices or []
        self.remove_proxies = remove_proxies
        self.mask_ = None
        self.n_features_in_ = None
        self.n_features_out_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'FeatureMasking':
        """
        Fit the feature masker.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray, optional
            Target labels (not used)
        
        Returns:
        --------
        self
        """
        self.n_features_in_ = X.shape[1]
        
        # Determine which features to remove
        remove_indices = set(self.sensitive_indices)
        if self.remove_proxies:
            remove_indices.update(self.proxy_indices)
        
        # Create mask
        self.mask_ = np.array([i not in remove_indices for i in range(self.n_features_in_)])
        self.n_features_out_ = np.sum(self.mask_)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Remove masked features.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        
        Returns:
        --------
        np.ndarray : Feature matrix with masked features removed
        """
        return X[:, self.mask_]
    
    def get_information_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate information loss from feature masking.
        
        Parameters:
        -----------
        X : np.ndarray
            Original feature matrix
        y : np.ndarray
            Target labels
        
        Returns:
        --------
        float : Fraction of features removed
        """
        return 1 - (self.n_features_out_ / self.n_features_in_)


class DisparateImpactRemover(BaseEstimator, TransformerMixin):
    """
    Disparate Impact Remover.
    
    Modifies feature values to reduce correlation with sensitive attribute.
    Based on: Feldman et al. (2015) - Certifying and removing disparate impact.
    """
    
    def __init__(self, 
                 sensitive_attr_idx: int = 0,
                 repair_level: float = 1.0):
        """
        Parameters:
        -----------
        sensitive_attr_idx : int
            Index of the sensitive attribute column
        repair_level : float
            Level of repair (0 = no repair, 1 = full repair)
        """
        self.sensitive_attr_idx = sensitive_attr_idx
        self.repair_level = repair_level
        self.medians_ = None
        self.group_distributions_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray = None, sensitive: np.ndarray = None) -> 'DisparateImpactRemover':
        """
        Fit the disparate impact remover.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray, optional
            Target labels (not used)
        sensitive : np.ndarray, optional
            Sensitive attribute values
        
        Returns:
        --------
        self
        """
        if sensitive is None:
            sensitive = X[:, self.sensitive_attr_idx]
        
        self.medians_ = {}
        self.group_distributions_ = {}
        
        groups = np.unique(sensitive)
        n_features = X.shape[1]
        
        for feat_idx in range(n_features):
            if feat_idx == self.sensitive_attr_idx:
                continue
            
            # Store median for each group
            self.medians_[feat_idx] = {}
            self.group_distributions_[feat_idx] = {}
            
            for g in groups:
                mask = sensitive == g
                self.medians_[feat_idx][g] = np.median(X[mask, feat_idx])
                self.group_distributions_[feat_idx][g] = np.sort(X[mask, feat_idx])
        
        return self
    
    def transform(self, X: np.ndarray, sensitive: np.ndarray = None) -> np.ndarray:
        """
        Transform features to reduce disparate impact.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        sensitive : np.ndarray, optional
            Sensitive attribute values
        
        Returns:
        --------
        np.ndarray : Transformed feature matrix
        """
        if sensitive is None:
            sensitive = X[:, self.sensitive_attr_idx]
        
        X_transformed = X.copy()
        groups = np.unique(sensitive)
        n_features = X.shape[1]
        
        for feat_idx in range(n_features):
            if feat_idx == self.sensitive_attr_idx:
                continue
            
            # Compute overall median
            overall_median = np.median(X[:, feat_idx])
            
            for g in groups:
                mask = sensitive == g
                group_median = self.medians_[feat_idx][g]
                
                # Shift towards overall median
                shift = (overall_median - group_median) * self.repair_level
                X_transformed[mask, feat_idx] = X[mask, feat_idx] + shift
        
        return X_transformed


class LabelFlipping(BaseEstimator, TransformerMixin):
    """
    Label flipping for fairness.
    
    Flips labels in the training data to reduce bias.
    Selectively changes labels for samples near decision boundary.
    """
    
    def __init__(self, 
                 flip_rate: float = 0.1,
                 strategy: str = 'targeted'):
        """
        Parameters:
        -----------
        flip_rate : float
            Fraction of labels to flip
        strategy : str
            Flipping strategy: 'random', 'targeted', or 'equalize'
        """
        self.flip_rate = flip_rate
        self.strategy = strategy
        self.flip_indices_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, sensitive: np.ndarray) -> 'LabelFlipping':
        """
        Determine which labels to flip.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        sensitive : np.ndarray
            Sensitive attribute values
        
        Returns:
        --------
        self
        """
        n = len(y)
        groups = np.unique(sensitive)
        
        if self.strategy == 'random':
            n_flip = int(n * self.flip_rate)
            self.flip_indices_ = np.random.choice(n, n_flip, replace=False)
            
        elif self.strategy == 'targeted':
            # Flip labels for underrepresented group-label combinations
            self.flip_indices_ = []
            
            for g in groups:
                mask = sensitive == g
                positive_rate = np.mean(y[mask])
                overall_rate = np.mean(y)
                
                if positive_rate < overall_rate:
                    # This group is underrepresented in positive class
                    # Flip some negative labels to positive
                    candidates = np.where(mask & (y == 0))[0]
                    n_flip = int(len(candidates) * self.flip_rate * 2)
                    if len(candidates) > 0:
                        self.flip_indices_.extend(
                            np.random.choice(candidates, min(n_flip, len(candidates)), replace=False)
                        )
                elif positive_rate > overall_rate:
                    # Flip some positive labels to negative
                    candidates = np.where(mask & (y == 1))[0]
                    n_flip = int(len(candidates) * self.flip_rate * 2)
                    if len(candidates) > 0:
                        self.flip_indices_.extend(
                            np.random.choice(candidates, min(n_flip, len(candidates)), replace=False)
                        )
            
            self.flip_indices_ = np.array(self.flip_indices_)
            
        elif self.strategy == 'equalize':
            # Flip to equalize positive rates across groups
            self.flip_indices_ = []
            positive_rates = {g: np.mean(y[sensitive == g]) for g in groups}
            target_rate = np.mean(y)
            
            for g in groups:
                mask = sensitive == g
                rate_diff = positive_rates[g] - target_rate
                
                if rate_diff > 0:
                    # Too many positives, flip some to negative
                    candidates = np.where(mask & (y == 1))[0]
                    n_flip = int(abs(rate_diff) * np.sum(mask))
                    if len(candidates) > 0:
                        self.flip_indices_.extend(
                            np.random.choice(candidates, min(n_flip, len(candidates)), replace=False)
                        )
                else:
                    # Too few positives, flip some to positive
                    candidates = np.where(mask & (y == 0))[0]
                    n_flip = int(abs(rate_diff) * np.sum(mask))
                    if len(candidates) > 0:
                        self.flip_indices_.extend(
                            np.random.choice(candidates, min(n_flip, len(candidates)), replace=False)
                        )
            
            self.flip_indices_ = np.array(self.flip_indices_)
        
        return self
    
    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Apply label flipping.
        
        Parameters:
        -----------
        y : np.ndarray
            Original labels
        
        Returns:
        --------
        np.ndarray : Flipped labels
        """
        y_flipped = y.copy()
        if len(self.flip_indices_) > 0:
            y_flipped[self.flip_indices_] = 1 - y_flipped[self.flip_indices_]
        return y_flipped


class SamplingStrategy(BaseEstimator, TransformerMixin):
    """
    Sampling strategies for fairness.
    
    Implements oversampling, undersampling, and hybrid strategies.
    """
    
    def __init__(self, 
                 strategy: str = 'oversample',
                 target_ratio: float = 1.0):
        """
        Parameters:
        -----------
        strategy : str
            Sampling strategy: 'oversample', 'undersample', or 'hybrid'
        target_ratio : float
            Target ratio for group sizes
        """
        self.strategy = strategy
        self.target_ratio = target_ratio
        
    def fit_resample(self, 
                     X: np.ndarray, 
                     y: np.ndarray, 
                     sensitive: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Resample the data for fairness.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        sensitive : np.ndarray
            Sensitive attribute values
        
        Returns:
        --------
        Tuple of resampled X, y, and sensitive arrays
        """
        groups = np.unique(sensitive)
        labels = np.unique(y)
        
        # Count each group-label combination
        counts = {}
        for g in groups:
            for l in labels:
                counts[(g, l)] = np.sum((sensitive == g) & (y == l))
        
        # Determine target count
        max_count = max(counts.values())
        min_count = min(counts.values())
        
        if self.strategy == 'oversample':
            target_count = max_count
        elif self.strategy == 'undersample':
            target_count = min_count
        else:  # hybrid
            target_count = int((max_count + min_count) / 2)
        
        # Resample each group-label combination
        X_resampled = []
        y_resampled = []
        s_resampled = []
        
        for g in groups:
            for l in labels:
                mask = (sensitive == g) & (y == l)
                indices = np.where(mask)[0]
                current_count = len(indices)
                
                if current_count < target_count:
                    # Oversample
                    sampled_indices = np.random.choice(indices, target_count, replace=True)
                elif current_count > target_count and self.strategy in ['undersample', 'hybrid']:
                    # Undersample
                    sampled_indices = np.random.choice(indices, target_count, replace=False)
                else:
                    sampled_indices = indices
                
                X_resampled.append(X[sampled_indices])
                y_resampled.append(y[sampled_indices])
                s_resampled.append(sensitive[sampled_indices])
        
        return (
            np.vstack(X_resampled),
            np.concatenate(y_resampled),
            np.concatenate(s_resampled)
        )


def compute_preprocessing_info_loss(X_original: np.ndarray,
                                     X_transformed: np.ndarray,
                                     y: np.ndarray) -> Dict[str, float]:
    """
    Compute information loss metrics for preprocessing transformations.
    
    Parameters:
    -----------
    X_original : np.ndarray
        Original feature matrix
    X_transformed : np.ndarray
        Transformed feature matrix
    y : np.ndarray
        Target labels
    
    Returns:
    --------
    Dict containing various information loss metrics
    """
    from sklearn.feature_selection import mutual_info_classif
    
    # Mutual information loss
    mi_original = np.sum(mutual_info_classif(X_original, y, random_state=42))
    mi_transformed = np.sum(mutual_info_classif(X_transformed, y, random_state=42))
    mi_loss = (mi_original - mi_transformed) / mi_original if mi_original > 0 else 0
    
    # Variance retained (if same number of features)
    if X_original.shape[1] == X_transformed.shape[1]:
        var_original = np.var(X_original, axis=0).sum()
        var_transformed = np.var(X_transformed, axis=0).sum()
        var_retained = var_transformed / var_original if var_original > 0 else 1
    else:
        var_retained = X_transformed.shape[1] / X_original.shape[1]
    
    # Feature reduction
    feature_reduction = 1 - (X_transformed.shape[1] / X_original.shape[1])
    
    return {
        'mutual_information_loss': mi_loss,
        'variance_retained': var_retained,
        'feature_reduction': feature_reduction,
        'original_mi': mi_original,
        'transformed_mi': mi_transformed
    }
