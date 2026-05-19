"""Hybrid input representations A / A_v2 / A_v3 / B / C / D for PCGC.

Ported from ``ECPG/core/pcgc_repr.py``. The only edit relative to the
original is the import path — behaviour is identical, including the
StandardScaler train-only fit inside ``build_fold_hybrid_features`` that
preserves the no-leakage protocol.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from g2c.classification.pcgc import build_train_prototypes, compute_pcgc_scores


REPRESENTATION_CHOICES = {"A", "A_v2", "A_v3", "B", "C", "D"}
MODEL_CHOICES = {"lr", "linsvm", "xgb"}


def load_participant_feature_csv(
    csv_path: str,
    pid_col: str = "p_id",
    label_col: str = "expertise",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load participant-level feature CSV.

    Expected format:
    - one row per participant
    - contains pid_col and label_col
    - all other numeric columns are treated as participant representation features
    """
    df = pd.read_csv(csv_path)

    if pid_col not in df.columns:
        raise ValueError(f"Missing pid column: {pid_col}")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    feature_cols = [c for c in df.columns if c not in {pid_col, label_col}]
    if not feature_cols:
        raise ValueError("No feature columns found.")

    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[pid_col, label_col] + feature_cols).copy()
    df[label_col] = df[label_col].astype(int)

    return df, feature_cols


def build_representation_matrix(
    df_repr: pd.DataFrame,
    pcgc_df: pd.DataFrame,
    feature_cols: List[str],
    representation: str,
) -> np.ndarray:
    """
    Build hybrid input matrix.

    Representation A:
      [d_high, d_low, v1_score, ratio]

    Representation B:
      original participant representation (e.g. 10-d sim_high + sim_low)

    Representation C:
      original representation + [v1_score, v2_score, v3_score]

    Representation D:
      [v3_score]
    """
    if representation not in REPRESENTATION_CHOICES:
        raise ValueError(f"Unsupported representation: {representation}")

    X_repr = df_repr[feature_cols].to_numpy(dtype=float)

    if representation == "A":
        Z = pcgc_df[["d_high", "d_low", "v1_score", "ratio"]].to_numpy(dtype=float)
    elif representation == "A_v2":
        Z = pcgc_df[["d_high", "d_low", "v1_score", "v2_score", "ratio"]].to_numpy(dtype=float)
    elif representation == "A_v3":
        Z = pcgc_df[["d_high", "d_low", "v1_score", "v2_score", "v3_score", "ratio"]].to_numpy(dtype=float)
    elif representation == "B":
        Z = X_repr
    elif representation == "C":
        extra = pcgc_df[["v1_score", "v2_score", "v3_score"]].to_numpy(dtype=float)
        Z = np.concatenate([X_repr, extra], axis=1)
    elif representation == "D":
        Z = pcgc_df[["v3_score"]].to_numpy(dtype=float)
    else:
        raise ValueError(representation)

    return Z


def make_model(model_name: str, seed: int, y_train: np.ndarray, lr_C: float = 1.0):
    if model_name not in MODEL_CHOICES:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_name == "lr":
        return LogisticRegression(
            class_weight="balanced",
            penalty="l2",
            C=lr_C,
            solver="liblinear",
            random_state=seed,
            max_iter=1000,
        )

    if model_name == "linsvm":
        return SVC(
            kernel="linear",
            probability=True,
            class_weight="balanced",
            random_state=seed,
        )

    if model_name == "xgb":
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        scale_pos_weight = n_neg / max(n_pos, 1)

        return XGBClassifier(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            eval_metric="logloss",
            verbosity=0,
        )

    raise ValueError(model_name)


def build_fold_hybrid_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    representation: str,
    agg: str = "mean",
    reg: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Honest fold-wise pipeline:
    1. build prototypes on train only
    2. compute PCGC scores for train/test
    3. build requested representation
    4. scale train only
    """
    X_train_repr = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["expertise"].to_numpy(dtype=int)
    X_test_repr = test_df[feature_cols].to_numpy(dtype=float)

    bundle = build_train_prototypes(X_train_repr, y_train, agg=agg, reg=reg)

    pcgc_train = compute_pcgc_scores(X_train_repr, bundle, score_fn="v1")
    pcgc_test = compute_pcgc_scores(X_test_repr, bundle, score_fn="v1")

    Z_train = build_representation_matrix(train_df, pcgc_train, feature_cols, representation)
    Z_test = build_representation_matrix(test_df, pcgc_test, feature_cols, representation)

    scaler = StandardScaler()
    Z_train = scaler.fit_transform(Z_train)
    Z_test = scaler.transform(Z_test)

    return Z_train, y_train, Z_test
