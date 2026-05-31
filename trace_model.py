"""
TRACE: reproducible NSL-KDD architecture.

TRACE combines flow features with lightweight surrogate-graph statistics for
topology-poor flow datasets. The final classifier uses:

  raw flow features
    -> train-fitted preprocessing
    -> surrogate graph statistics
    -> Borderline-SMOTE on the training fold only
    -> class-balanced XGBoost
    -> validation-calibrated multiclass thresholds

This file intentionally contains model and evaluation code only. It does not
generate figures or tables.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


SEED = 42
CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]
NSL_MAP = {
    "normal": 0,
    "neptune": 1, "back": 1, "land": 1, "pod": 1, "smurf": 1, "teardrop": 1,
    "apache2": 1, "udpstorm": 1, "processtable": 1, "worm": 1, "mailbomb": 1,
    "satan": 2, "ipsweep": 2, "nmap": 2, "portsweep": 2, "mscan": 2, "saint": 2,
    "guess_passwd": 3, "ftp_write": 3, "imap": 3, "phf": 3, "multihop": 3,
    "warezmaster": 3, "warezclient": 3, "spy": 3, "snmpguess": 3,
    "snmpgetattack": 3, "httptunnel": 3, "sendmail": 3, "named": 3,
    "buffer_overflow": 4, "loadmodule": 4, "perl": 4, "rootkit": 4,
    "ps": 4, "xterm": 4, "sqlattack": 4,
}


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


class SurrogateGraphFeatures:
    """Build train-fitted row-level graph statistics from two flow dimensions."""

    def __init__(self, n_nodes: int = 16):
        if n_nodes % 2:
            raise ValueError("n_nodes must be even")
        self.n_nodes = n_nodes
        self.half = n_nodes // 2
        self.src_bins: np.ndarray | None = None
        self.dst_bins: np.ndarray | None = None
        self.pair_count: Counter[int] = Counter()
        self.src_count: Counter[int] = Counter()
        self.dst_count: Counter[int] = Counter()

    @staticmethod
    def _safe_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
        bins = np.percentile(values, np.linspace(0, 100, n_bins + 1)).astype(float)
        bins[0] -= 1e-9
        bins[-1] += 1e-9
        return bins

    def _assign(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.src_bins is None or self.dst_bins is None:
            raise RuntimeError("SurrogateGraphFeatures must be fitted before transform")
        src = np.clip(np.digitize(X[:, 0], self.src_bins) - 1, 0, self.half - 1)
        dst = np.clip(np.digitize(X[:, 1], self.dst_bins) - 1, 0, self.half - 1) + self.half
        return src.astype(int), dst.astype(int)

    def fit(self, X: np.ndarray) -> "SurrogateGraphFeatures":
        self.src_bins = self._safe_bins(X[:, 0], self.half)
        self.dst_bins = self._safe_bins(X[:, 1], self.half)
        src, dst = self._assign(X)
        pairs = src * self.n_nodes + dst
        self.pair_count = Counter(pairs.tolist())
        self.src_count = Counter(src.tolist())
        self.dst_count = Counter(dst.tolist())
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        src, dst = self._assign(X)
        pairs = src * self.n_nodes + dst
        pair_freq = np.array([self.pair_count[int(p)] for p in pairs], dtype=np.float32)
        src_freq = np.array([self.src_count[int(s)] for s in src], dtype=np.float32)
        dst_freq = np.array([self.dst_count[int(d)] for d in dst], dtype=np.float32)
        one_hot_src = np.eye(self.n_nodes, dtype=np.float32)[src]
        one_hot_dst = np.eye(self.n_nodes, dtype=np.float32)[dst]
        local = np.c_[src, dst, np.log1p(pair_freq), np.log1p(src_freq), np.log1p(dst_freq)]
        return np.c_[local, one_hot_src, one_hot_dst]


def augment_rare_classes(
    X: np.ndarray,
    y: np.ndarray,
    target_min: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    counts = Counter(y.tolist())
    strategy = {cls: max(count, target_min) for cls, count in counts.items()}
    try:
        return BorderlineSMOTE(
            sampling_strategy=strategy,
            random_state=random_state,
            k_neighbors=3,
            m_neighbors=10,
        ).fit_resample(X, y)
    except ValueError:
        return SMOTE(
            sampling_strategy=strategy,
            random_state=random_state,
            k_neighbors=3,
        ).fit_resample(X, y)


def calibrate_thresholds(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    thresholds = np.ones(probs.shape[1], dtype=float)
    tuned = probs.argmax(axis=1)
    best = f1_score(y_true, tuned, average="macro", zero_division=0)
    for cls in [4, 3, 2, 1, 0]:
        cls_best = (best, thresholds[cls], tuned.copy())
        for threshold in np.linspace(0.02, 0.85, 84):
            candidate = tuned.copy()
            candidate[probs[:, cls] >= threshold] = cls
            score = f1_score(y_true, candidate, average="macro", zero_division=0)
            if score > cls_best[0]:
                cls_best = (score, threshold, candidate)
        best, thresholds[cls], tuned = cls_best
    return thresholds


def apply_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    pred = probs.argmax(axis=1)
    for cls in [4, 3, 2, 1, 0]:
        pred[probs[:, cls] >= thresholds[cls]] = cls
    return pred


@dataclass
class TraceConfig:
    seed: int = SEED
    validation_size: float = 0.20
    surrogate_nodes: int = 16
    rare_class_target: int = 12_000
    estimators: int = 1_100
    max_depth: int = 5
    learning_rate: float = 0.03
    subsample: float = 0.90
    colsample_bytree: float = 0.86


class TRACE:
    """Final efficient TRACE classifier for topology-poor flow datasets."""

    def __init__(self, config: TraceConfig | None = None):
        self.config = config or TraceConfig()
        self.graph = SurrogateGraphFeatures(self.config.surrogate_nodes)
        self.model: XGBClassifier | None = None
        self.thresholds: np.ndarray | None = None

    def _combine(self, X: np.ndarray) -> np.ndarray:
        return np.c_[X, self.graph.transform(X)]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TRACE":
        X_fit, X_val, y_fit, y_val = train_test_split(
            X,
            y,
            test_size=self.config.validation_size,
            stratify=y,
            random_state=self.config.seed,
        )
        self.graph.fit(X_fit)
        X_fit = self._combine(X_fit)
        X_val = self._combine(X_val)
        X_aug, y_aug = augment_rare_classes(
            X_fit,
            y_fit,
            target_min=self.config.rare_class_target,
            random_state=self.config.seed,
        )
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_aug)
        self.model = XGBClassifier(
            n_estimators=self.config.estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            min_child_weight=1,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=self.config.seed,
            n_jobs=1,
            tree_method="hist",
        )
        self.model.fit(X_aug, y_aug, sample_weight=sample_weight)
        self.thresholds = calibrate_thresholds(self.model.predict_proba(X_val), y_val)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("TRACE must be fitted before prediction")
        return self.model.predict_proba(self._combine(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.thresholds is None:
            raise RuntimeError("TRACE must be fitted before prediction")
        return apply_thresholds(self.predict_proba(X), self.thresholds)


def load_nsl_kdd(
    train_csv: str | Path,
    test_csv: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    for frame in (train, test):
        frame["labels"] = (
            frame["labels"].astype(str).str.strip(".").str.lower().map(NSL_MAP).fillna(0).astype(int)
        )
    y_train = train.pop("labels").to_numpy()
    y_test = test.pop("labels").to_numpy()
    categorical = train.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [col for col in train.columns if col not in categorical]
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("scale", StandardScaler())]), numeric),
            (
                "categorical",
                Pipeline([
                    ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ("scale", StandardScaler()),
                ]),
                categorical,
            ),
        ],
        remainder="drop",
    )
    X_train = np.asarray(preprocess.fit_transform(train), dtype=np.float32)
    X_test = np.asarray(preprocess.transform(test), dtype=np.float32)
    return X_train, X_test, y_train, y_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate TRACE on NSL-KDD")
    parser.add_argument("--train", required=True, help="Path to kdd_train.csv")
    parser.add_argument("--test", required=True, help="Path to kdd_test.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    X_train, X_test, y_train, y_test = load_nsl_kdd(args.train, args.test)
    model = TRACE().fit(X_train, y_train)
    pred = model.predict(X_test)
    for name, value in metric_dict(y_test, pred).items():
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()
