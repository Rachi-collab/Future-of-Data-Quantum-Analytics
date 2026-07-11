"""
Classical machine-learning helpers.
"""

import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)


CLASSIFIERS = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
}


def train_and_evaluate(clf, X_train, y_train, X_test, y_test):
    """Train a classifier and return a dict of metrics."""
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = clf.predict(X_test)
    infer_time = time.perf_counter() - t0

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "train_time_ms": train_time * 1000,
        "infer_time_ms": infer_time * 1000,
        "y_pred": y_pred,
    }


def cross_validate(clf, X, y, cv=5):
    """Return mean ± std CV accuracy."""
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    return scores.mean(), scores.std()
