"""
model_utils.py
--------------
Comprehensive utility functions for training, evaluating, saving, loading ML models.
Includes MLflow integration, cross-validation, hyperparameter tuning, and prediction utilities.
Designed for classification and regression tasks.
"""

import os
import joblib
from typing import Any, Dict, Optional, Tuple
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, mean_squared_error, r2_score
)
import numpy as np

# --------------------------------------
# 1. MODEL TRAINING
# --------------------------------------
def train_model(model, X_train, y_train):
    """
    Train a given ML model.
    
    Args:
        model: Scikit-learn compatible model
        X_train: Training features
        y_train: Training labels
    
    Returns:
        Trained model
    """
    model.fit(X_train, y_train)
    return model

# --------------------------------------
# 2. MODEL EVALUATION
# --------------------------------------
def evaluate_classification(model, X_test, y_test, average='weighted') -> Dict[str, Any]:
    """
    Evaluate classification model using multiple metrics.
    """
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average=average, zero_division=0),
        "classification_report": classification_report(y_test, y_pred)
    }
    return metrics

def evaluate_regression(model, X_test, y_test) -> Dict[str, float]:
    """
    Evaluate regression model.
    """
    y_pred = model.predict(X_test)
    metrics = {
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred)
    }
    return metrics

# --------------------------------------
# 3. CROSS-VALIDATION
# --------------------------------------
def cross_validate_model(model, X, y, cv=5, scoring='accuracy') -> Dict[str, float]:
    """
    Perform cross-validation and return mean and std scores.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return {"mean_score": scores.mean(), "std_score": scores.std()}

# --------------------------------------
# 4. GRID SEARCH / HYPERPARAMETER TUNING
# --------------------------------------
def grid_search(model, param_grid: Dict, X, y, cv=5, scoring='accuracy') -> Tuple[Any, Dict[str, Any]]:
    """
    Perform GridSearchCV for hyperparameter tuning.
    Returns best model and best parameters.
    """
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_

# --------------------------------------
# 5. MODEL SAVING & LOADING
# --------------------------------------
def save_model(model, path: str):
    """
    Save model to disk using joblib.
    """
    folder = os.path.dirname(path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    joblib.dump(model, path)

def load_model(path: str):
    """
    Load saved model from disk.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)

# --------------------------------------
# 6. MLflow INTEGRATION
# --------------------------------------
def log_model_mlflow(model, model_name: str, metrics: Optional[Dict[str, float]] = None, params: Optional[Dict[str, Any]] = None):
    """
    Log model, metrics, and hyperparameters to MLflow.
    """
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, model_name)
        if metrics:
            mlflow.log_metrics(metrics)
        if params:
            mlflow.log_params(params)

# --------------------------------------
# 7. PREDICTION UTILITIES
# --------------------------------------
def predict_single(model, X_sample):
    """
    Predict a single sample.
    """
    return model.predict([X_sample])[0]

def predict_batch(model, X):
    """
    Predict a batch of samples.
    """
    return model.predict(X)

# --------------------------------------
# 8. PIPELINE SUPPORT
# --------------------------------------
def train_evaluate_log(model, X_train, y_train, X_test, y_test, model_name: str, task='classification', params=None):
    """
    Train, evaluate, and log a model in one step (for MLflow pipelines).
    Supports classification and regression.
    """
    model.fit(X_train, y_train)
    
    if task == 'classification':
        metrics = evaluate_classification(model, X_test, y_test)
    elif task == 'regression':
        metrics = evaluate_regression(model, X_test, y_test)
    else:
        raise ValueError("task must be 'classification' or 'regression'")
    
    log_model_mlflow(model, model_name=model_name, metrics=metrics, params=params)
    return model, metrics

# --------------------------------------
# 9. UTILITY FUNCTIONS
# --------------------------------------
def get_feature_importances(model, feature_names: list):
    """
    Extract feature importances if model supports it.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        return dict(zip(feature_names, importances))
    else:
        raise AttributeError("Model does not have feature_importances_ attribute")

def check_model_type(model):
    """
    Return model type (classification/regression) based on attributes.
    """
    from sklearn.base import ClassifierMixin, RegressorMixin
    if isinstance(model, ClassifierMixin):
        return 'classification'
    elif isinstance(model, RegressorMixin):
        return 'regression'
    else:
        return 'unknown'