"""
Data utilities for the Quantum Analytics project.
Generates synthetic datasets for demonstration purposes.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.preprocessing import StandardScaler


def generate_dataset(dataset_name: str, n_samples: int = 300, noise: float = 0.2, random_state: int = 42):
    """
    Generate a classification dataset.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
    y : ndarray of shape (n_samples,)
    feature_names : list[str]
    """
    rng = np.random.RandomState(random_state)

    if dataset_name == "Moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        feature_names = ["Feature 1 (x)", "Feature 2 (y)"]

    elif dataset_name == "Circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
        feature_names = ["Feature 1 (x)", "Feature 2 (y)"]

    elif dataset_name == "Linear":
        X, y = make_classification(
            n_samples=n_samples, n_features=2, n_redundant=0,
            n_informative=2, n_clusters_per_class=1,
            class_sep=1.5, random_state=random_state
        )
        feature_names = ["Feature 1", "Feature 2"]

    elif dataset_name == "Quantum Finance":
        # Simulated financial data (price momentum vs volatility)
        n = n_samples
        momentum = rng.randn(n)
        volatility = np.abs(rng.randn(n)) + 0.5
        noise_vec = rng.randn(n) * noise
        signal = momentum / volatility + noise_vec
        y = (signal > 0).astype(int)
        X = np.column_stack([momentum, volatility])
        feature_names = ["Price Momentum", "Volatility"]

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, feature_names


def make_dataframe(X: np.ndarray, y: np.ndarray, feature_names: list) -> pd.DataFrame:
    """Wrap arrays into a labelled DataFrame."""
    df = pd.DataFrame(X, columns=feature_names)
    df["Label"] = y
    return df


AVAILABLE_DATASETS = ["Moons", "Circles", "Linear", "Quantum Finance"]
