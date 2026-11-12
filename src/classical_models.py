"""Conventional machine-learning baselines (scikit-learn).

Every baseline is a pipeline that first standardises the four features and
then fits the model, so the scaling is learned from the training split only.
Library default hyper-parameters are used unless stated; neither the classical
models nor the hybrid model were tuned on the test data.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_baselines(seed: int = 0) -> dict:
    """Return a dictionary {name: unfitted scikit-learn pipeline}."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)),
        "SVM (RBF kernel)": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", random_state=seed)),
        "k-Nearest Neighbours": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Random Forest": make_pipeline(
            StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=seed)),
        # One hidden layer of 4 neurons: 4*4 + 4 + 4 + 1 = 25 trainable values,
        # the closest small network to the 17 values of the hybrid model.
        "Neural Network (4-4-1)": make_pipeline(
            StandardScaler(), MLPClassifier(hidden_layer_sizes=(4,), activation="tanh",
                                            max_iter=2000, random_state=seed)),
    }


def count_parameters(name: str, model) -> str:
    """Number of trainable values of a fitted baseline, for the results table."""
    estimator = model.steps[-1][1]
    if name == "Logistic Regression":
        return str(estimator.coef_.size + estimator.intercept_.size)
    if name.startswith("Neural Network"):
        return str(sum(c.size for c in estimator.coefs_) + sum(i.size for i in estimator.intercepts_))
    if name.startswith("SVM"):
        return f"{int(estimator.n_support_.sum())} support vectors"
    if name.startswith("Random Forest"):
        return f"{sum(t.tree_.node_count for t in estimator.estimators_)} tree nodes"
    return "stores training set"
