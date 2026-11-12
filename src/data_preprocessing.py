"""Loading, splitting and scaling of the UCI Banknote Authentication dataset.

Two different scalings are needed because the two model families read the
data differently:

* classical models  -> zero mean, unit variance (StandardScaler)
* quantum circuit   -> rotation angles in [0, pi]  (AngleScaler)

Both scalers are fitted on the training split only, so no information from
the test notes leaks into training.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from . import config


def download_dataset(target: Path = config.DATA_FILE) -> Path:
    """Fetch the dataset from the UCI repository if it is not on disk yet."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(config.DATA_URL, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    target.write_bytes(archive.read("data_banknote_authentication.txt"))
    return target


def load_banknote(path: Path = config.DATA_FILE) -> pd.DataFrame:
    """Return the dataset as a DataFrame with named columns.

    The raw file has 1,372 rows and no header:
    variance, skewness, curtosis, entropy, class
    """
    path = Path(path)
    if not path.exists():
        download_dataset(path)
    return pd.read_csv(path, header=None, names=config.FEATURE_NAMES + ["class"])


def get_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Split the DataFrame into a feature matrix X and a label vector y."""
    X = df[config.FEATURE_NAMES].to_numpy(dtype=float)
    y = df["class"].to_numpy(dtype=int)
    return X, y


def make_split(X: np.ndarray, y: np.ndarray, seed: int, test_size: float = config.TEST_SIZE):
    """Stratified train/test split (class proportions are kept in both parts)."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


class AngleScaler:
    """Map every feature linearly onto a rotation angle between 0 and pi.

    RY(angle) applied to |0> gives <Z> = cos(angle). On the interval [0, pi]
    the cosine is strictly decreasing, so two different feature values can
    never be encoded as the same qubit state. Test values that fall outside
    the range seen in training are clipped to the nearest end of the interval.
    """

    def __init__(self) -> None:
        self._scaler = MinMaxScaler(feature_range=(0.0, np.pi))

    def fit(self, X: np.ndarray) -> "AngleScaler":
        self._scaler.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.clip(self._scaler.transform(X), 0.0, np.pi)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics per feature, used for the dataset table in the report."""
    stats = df[config.FEATURE_NAMES].describe().T[["mean", "std", "min", "max"]]
    stats["missing"] = df[config.FEATURE_NAMES].isna().sum()
    return stats.round(3)


if __name__ == "__main__":
    data = load_banknote()
    print(f"Rows: {len(data)}  Columns: {list(data.columns)}")
    print("Class counts:", data["class"].value_counts().sort_index().to_dict())
    print("Duplicate rows:", int(data.duplicated().sum()))
    print(describe(data))
