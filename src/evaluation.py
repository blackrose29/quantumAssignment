"""Metrics shared by the classical and the hybrid models.

"Forged" (class 1) is the positive class:

    precision = of the notes flagged as forged, how many really are forged
    recall    = of the forged notes, how many were caught
    F1        = harmonic mean of precision and recall
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)


def model_outputs(model, X) -> tuple[np.ndarray, np.ndarray]:
    """Predicted labels and a continuous score for class 1 from a scikit-learn model.

    The score is the predicted probability when the model offers one and the
    signed distance to the decision boundary otherwise (support vector machine).
    It is only used for the ROC-AUC, which needs a ranking, not a probability.
    """
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        return y_pred, model.predict_proba(X)[:, 1]
    return y_pred, model.decision_function(X)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    """Return all evaluation metrics from predicted labels and class-1 scores."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def mean_std(values) -> tuple[float, float]:
    """Mean and sample standard deviation of a list of numbers."""
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0
