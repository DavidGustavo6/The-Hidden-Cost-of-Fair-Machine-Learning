"""
In-processing fairness methods.
This module implements fairness interventions applied during model training.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from typing import Dict, Optional, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')


class FairnessRegularizedClassifier(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression with Fairness Regularization.
    
    Adds a fairness penalty term to the standard logistic loss.
    Penalizes demographic parity violations during optimization.
    """
    
    def __init__(self,
                 lambda_fairness: float = 1.0,
                 C: float = 1.0,
                 max_iter: int = 1000,
                 learning_rate: float = 0.01,
                 fairness_metric: str = 'demographic_parity'):
        """
        Parameters:
        -----------
        lambda_fairness : float
            Weight of the fairness regularization term
        C : float
            Inverse of regularization strength
        max_iter : int
            Maximum number of iterations
        learning_rate : float
            Learning rate for gradient descent
        fairness_metric : str
            Which fairness metric to optimize: 'demographic_parity' or 'equalized_odds'
        """
        self.lambda_fairness = lambda_fairness
        self.C = C
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.fairness_metric = fairness_metric
        self.weights_ = None
        self.bias_ = None
        self.loss_history_ = []
        self.fairness_history_ = []
        
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function."""
        return np.where(z >= 0, 
                       1 / (1 + np.exp(-z)), 
                       np.exp(z) / (1 + np.exp(z)))
    
    def _compute_fairness_gradient(self, 
                                    X: np.ndarray, 
                                    y_prob: np.ndarray,
                                    sensitive: np.ndarray) -> np.ndarray:
        """
        Compute gradient of fairness penalty.
        
        For demographic parity: penalize difference in mean predictions.
        """
        mask_0 = sensitive == 0
        mask_1 = sensitive == 1
        
        n_0 = max(np.sum(mask_0), 1)
        n_1 = max(np.sum(mask_1), 1)
        
        # Demographic parity gradient
        mean_pred_0 = np.sum(y_prob[mask_0]) / n_0
        mean_pred_1 = np.sum(y_prob[mask_1]) / n_1
        
        diff = mean_pred_0 - mean_pred_1
        
        # Gradient w.r.t. predictions
        grad_pred = np.zeros_like(y_prob)
        grad_pred[mask_0] = 2 * diff * y_prob[mask_0] * (1 - y_prob[mask_0]) / n_0
        grad_pred[mask_1] = -2 * diff * y_prob[mask_1] * (1 - y_prob[mask_1]) / n_1
        
        # Gradient w.r.t. weights
        grad_w = X.T @ grad_pred
        
        return grad_w
    
    def fit(self, X: np.ndarray, y: np.ndarray, sensitive: np.ndarray) -> 'FairnessRegularizedClassifier':
        """
        Fit the classifier with fairness regularization.
        
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
        n_samples, n_features = X.shape
        
        # Initialize weights
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0
        
        self.loss_history_ = []
        self.fairness_history_ = []
        
        for iteration in range(self.max_iter):
            # Forward pass
            z = X @ self.weights_ + self.bias_
            y_prob = self._sigmoid(z)
            
            # Clip probabilities for numerical stability
            y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)
            
            # Compute logistic loss
            log_loss = -np.mean(y * np.log(y_prob) + (1 - y) * np.log(1 - y_prob))
            
            # Compute fairness penalty
            mask_0 = sensitive == 0
            mask_1 = sensitive == 1
            dp_diff = abs(np.mean(y_prob[mask_0]) - np.mean(y_prob[mask_1]))
            fairness_penalty = dp_diff ** 2
            
            # Total loss
            total_loss = log_loss + self.lambda_fairness * fairness_penalty + (1 / (2 * self.C)) * np.sum(self.weights_ ** 2)
            
            self.loss_history_.append(total_loss)
            self.fairness_history_.append(dp_diff)
            
            # Compute gradients
            error = y_prob - y
            grad_w = (1 / n_samples) * (X.T @ error) + (1 / self.C) * self.weights_
            grad_b = np.mean(error)
            
            # Add fairness gradient
            if self.lambda_fairness > 0:
                fairness_grad = self._compute_fairness_gradient(X, y_prob, sensitive)
                grad_w += self.lambda_fairness * fairness_grad
            
            # Update weights
            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b
            
            # Check convergence
            if iteration > 0 and abs(self.loss_history_[-1] - self.loss_history_[-2]) < 1e-6:
                break
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        z = X @ self.weights_ + self.bias_
        prob_1 = self._sigmoid(z)
        prob_0 = 1 - prob_1
        return np.column_stack([prob_0, prob_1])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)


class AdversarialDebiasing(BaseEstimator, ClassifierMixin):
    """
    Adversarial Debiasing for Fair Classification.
    
    Uses an adversarial network to remove information about the sensitive
    attribute from the learned representations.
    Based on: Zhang et al. (2018) - Mitigating unwanted biases with adversarial learning.
    """
    
    def __init__(self,
                 hidden_size: int = 32,
                 adversary_weight: float = 1.0,
                 learning_rate: float = 0.001,
                 n_epochs: int = 100,
                 batch_size: int = 64,
                 use_pytorch: bool = True):
        """
        Parameters:
        -----------
        hidden_size : int
            Size of hidden layer
        adversary_weight : float
            Weight of adversarial loss
        learning_rate : float
            Learning rate for optimization
        n_epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        use_pytorch : bool
            Whether to use PyTorch (if False, uses simplified numpy implementation)
        """
        self.hidden_size = hidden_size
        self.adversary_weight = adversary_weight
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.use_pytorch = use_pytorch
        self.model_ = None
        self.adversary_ = None
        self.loss_history_ = []
        
    def _build_models_numpy(self, n_features: int):
        """Build simple numpy-based models."""
        # Classifier weights
        self.clf_w1_ = np.random.randn(n_features, self.hidden_size) * 0.01
        self.clf_b1_ = np.zeros(self.hidden_size)
        self.clf_w2_ = np.random.randn(self.hidden_size, 1) * 0.01
        self.clf_b2_ = np.zeros(1)
        
        # Adversary weights
        self.adv_w1_ = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
        self.adv_b1_ = np.zeros(self.hidden_size)
        self.adv_w2_ = np.random.randn(self.hidden_size, 1) * 0.01
        self.adv_b2_ = np.zeros(1)
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(z >= 0,
                       1 / (1 + np.exp(-z)),
                       np.exp(z) / (1 + np.exp(z)))
    
    def _relu(self, z: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, z)
    
    def _forward_numpy(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass using numpy."""
        # Hidden layer
        h1 = self._relu(X @ self.clf_w1_ + self.clf_b1_)
        
        # Classifier output
        y_logit = h1 @ self.clf_w2_ + self.clf_b2_
        y_pred = self._sigmoid(y_logit.flatten())
        
        # Adversary output
        a1 = self._relu(h1 @ self.adv_w1_ + self.adv_b1_)
        s_logit = a1 @ self.adv_w2_ + self.adv_b2_
        s_pred = self._sigmoid(s_logit.flatten())
        
        return y_pred, s_pred, h1
    
    def fit(self, X: np.ndarray, y: np.ndarray, sensitive: np.ndarray) -> 'AdversarialDebiasing':
        """
        Fit the adversarial debiasing model.
        
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
        n_samples, n_features = X.shape
        
        if self.use_pytorch:
            try:
                return self._fit_pytorch(X, y, sensitive)
            except ImportError:
                print("PyTorch not available, falling back to numpy implementation")
                self.use_pytorch = False
        
        self._build_models_numpy(n_features)
        
        self.loss_history_ = []
        
        for epoch in range(self.n_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            
            epoch_clf_loss = 0
            epoch_adv_loss = 0
            
            for i in range(0, n_samples, self.batch_size):
                batch_idx = indices[i:i + self.batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                s_batch = sensitive[batch_idx]
                
                # Forward pass
                y_pred, s_pred, h1 = self._forward_numpy(X_batch)
                
                y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
                s_pred = np.clip(s_pred, 1e-15, 1 - 1e-15)
                
                # Classifier loss
                clf_loss = -np.mean(y_batch * np.log(y_pred) + (1 - y_batch) * np.log(1 - y_pred))
                
                # Adversary loss
                adv_loss = -np.mean(s_batch * np.log(s_pred) + (1 - s_batch) * np.log(1 - s_pred))
                
                epoch_clf_loss += clf_loss
                epoch_adv_loss += adv_loss
                
                # Backward pass for classifier
                clf_error = y_pred - y_batch
                grad_clf_w2 = h1.T @ clf_error.reshape(-1, 1) / len(y_batch)
                grad_clf_b2 = np.mean(clf_error)
                
                # Backward pass for adversary
                adv_error = s_pred - s_batch
                a1 = self._relu(h1 @ self.adv_w1_ + self.adv_b1_)
                grad_adv_w2 = a1.T @ adv_error.reshape(-1, 1) / len(s_batch)
                grad_adv_b2 = np.mean(adv_error)
                
                # Update classifier (minimize clf_loss - adversary_weight * adv_loss)
                self.clf_w2_ -= self.learning_rate * grad_clf_w2
                self.clf_b2_ -= self.learning_rate * grad_clf_b2
                
                # Update adversary (maximize adv_loss = minimize -adv_loss)
                self.adv_w2_ -= self.learning_rate * grad_adv_w2
                self.adv_b2_ -= self.learning_rate * grad_adv_b2
            
            self.loss_history_.append({
                'classifier_loss': epoch_clf_loss / (n_samples // self.batch_size),
                'adversary_loss': epoch_adv_loss / (n_samples // self.batch_size)
            })
        
        return self
    
    def _fit_pytorch(self, X: np.ndarray, y: np.ndarray, sensitive: np.ndarray) -> 'AdversarialDebiasing':
        """Fit using PyTorch for better optimization."""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).to(device)
        s_tensor = torch.FloatTensor(sensitive).to(device)
        
        n_features = X.shape[1]
        
        # Build classifier
        self.model_ = nn.Sequential(
            nn.Linear(n_features, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        ).to(device)
        
        # Build adversary
        self.adversary_ = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        ).to(device)
        
        # Feature extractor
        self.feature_extractor_ = nn.Sequential(
            nn.Linear(n_features, self.hidden_size),
            nn.ReLU()
        ).to(device)
        
        # Classifier head
        self.classifier_head_ = nn.Sequential(
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        ).to(device)
        
        # Optimizers
        clf_optimizer = optim.Adam(
            list(self.feature_extractor_.parameters()) + list(self.classifier_head_.parameters()),
            lr=self.learning_rate
        )
        adv_optimizer = optim.Adam(self.adversary_.parameters(), lr=self.learning_rate)
        
        criterion = nn.BCELoss()
        
        self.loss_history_ = []
        n_samples = len(X)
        
        for epoch in range(self.n_epochs):
            indices = torch.randperm(n_samples)
            
            epoch_clf_loss = 0
            epoch_adv_loss = 0
            n_batches = 0
            
            for i in range(0, n_samples, self.batch_size):
                batch_idx = indices[i:i + self.batch_size]
                X_batch = X_tensor[batch_idx]
                y_batch = y_tensor[batch_idx]
                s_batch = s_tensor[batch_idx]
                
                # Forward pass
                features = self.feature_extractor_(X_batch)
                y_pred = self.classifier_head_(features).squeeze()
                s_pred = self.adversary_(features.detach()).squeeze()
                
                # Classifier loss
                clf_loss = criterion(y_pred, y_batch)
                
                # Adversary loss
                adv_loss = criterion(s_pred, s_batch)
                
                # Update classifier (minimize clf_loss + adversary_weight * (-adv_loss))
                clf_optimizer.zero_grad()
                s_pred_grad = self.adversary_(features).squeeze()
                total_clf_loss = clf_loss - self.adversary_weight * criterion(s_pred_grad, s_batch)
                total_clf_loss.backward()
                clf_optimizer.step()
                
                # Update adversary
                adv_optimizer.zero_grad()
                features_detached = self.feature_extractor_(X_batch).detach()
                s_pred_adv = self.adversary_(features_detached).squeeze()
                adv_loss_update = criterion(s_pred_adv, s_batch)
                adv_loss_update.backward()
                adv_optimizer.step()
                
                epoch_clf_loss += clf_loss.item()
                epoch_adv_loss += adv_loss.item()
                n_batches += 1
            
            self.loss_history_.append({
                'classifier_loss': epoch_clf_loss / n_batches,
                'adversary_loss': epoch_adv_loss / n_batches
            })
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.use_pytorch and self.feature_extractor_ is not None:
            import torch
            device = next(self.feature_extractor_.parameters()).device
            X_tensor = torch.FloatTensor(X).to(device)
            with torch.no_grad():
                features = self.feature_extractor_(X_tensor)
                prob_1 = self.classifier_head_(features).squeeze().cpu().numpy()
        else:
            prob_1, _, _ = self._forward_numpy(X)
        
        prob_0 = 1 - prob_1
        return np.column_stack([prob_0, prob_1])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)


class ConstrainedClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifier with fairness constraints using projected gradient descent.
    
    Implements approximate satisfaction of fairness constraints during optimization.
    """
    
    def __init__(self,
                 constraint_type: str = 'demographic_parity',
                 constraint_threshold: float = 0.1,
                 max_iter: int = 500,
                 learning_rate: float = 0.01):
        """
        Parameters:
        -----------
        constraint_type : str
            Type of fairness constraint
        constraint_threshold : float
            Maximum allowed fairness violation
        max_iter : int
            Maximum iterations
        learning_rate : float
            Learning rate
        """
        self.constraint_type = constraint_type
        self.constraint_threshold = constraint_threshold
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weights_ = None
        self.bias_ = None
        
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(z >= 0,
                       1 / (1 + np.exp(-z)),
                       np.exp(z) / (1 + np.exp(z)))
    
    def _project_to_constraint(self, 
                                y_prob: np.ndarray, 
                                sensitive: np.ndarray) -> np.ndarray:
        """
        Project predictions to satisfy fairness constraint approximately.
        """
        mask_0 = sensitive == 0
        mask_1 = sensitive == 1
        
        mean_0 = np.mean(y_prob[mask_0])
        mean_1 = np.mean(y_prob[mask_1])
        diff = mean_0 - mean_1
        
        if abs(diff) > self.constraint_threshold:
            # Adjust predictions to reduce violation
            target_diff = np.sign(diff) * self.constraint_threshold
            adjustment = (diff - target_diff) / 2
            
            y_prob_adjusted = y_prob.copy()
            y_prob_adjusted[mask_0] -= adjustment
            y_prob_adjusted[mask_1] += adjustment
            
            # Clip to valid probability range
            y_prob_adjusted = np.clip(y_prob_adjusted, 0.01, 0.99)
            
            return y_prob_adjusted
        
        return y_prob
    
    def fit(self, X: np.ndarray, y: np.ndarray, sensitive: np.ndarray) -> 'ConstrainedClassifier':
        """Fit the constrained classifier."""
        n_samples, n_features = X.shape
        
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0
        
        for iteration in range(self.max_iter):
            # Forward pass
            z = X @ self.weights_ + self.bias_
            y_prob = self._sigmoid(z)
            
            # Project to constraint
            y_prob_projected = self._project_to_constraint(y_prob, sensitive)
            
            # Compute gradient using projected probabilities
            error = y_prob_projected - y
            grad_w = (1 / n_samples) * (X.T @ error)
            grad_b = np.mean(error)
            
            # Update
            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        z = X @ self.weights_ + self.bias_
        prob_1 = self._sigmoid(z)
        prob_0 = 1 - prob_1
        return np.column_stack([prob_0, prob_1])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)


def train_fair_model(X_train: np.ndarray,
                     y_train: np.ndarray,
                     sensitive_train: np.ndarray,
                     method: str = 'regularized',
                     **kwargs) -> BaseEstimator:
    """
    Train a fair classifier using the specified in-processing method.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    sensitive_train : np.ndarray
        Training sensitive attribute values
    method : str
        Method name: 'regularized', 'adversarial', or 'constrained'
    **kwargs : dict
        Additional parameters for the method
    
    Returns:
    --------
    Trained classifier
    """
    if method == 'regularized':
        clf = FairnessRegularizedClassifier(**kwargs)
    elif method == 'adversarial':
        clf = AdversarialDebiasing(**kwargs)
    elif method == 'constrained':
        clf = ConstrainedClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    clf.fit(X_train, y_train, sensitive_train)
    return clf
