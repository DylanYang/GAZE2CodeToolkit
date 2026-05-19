"""PCGC: Mahalanobis-prototype scorer for participant expertise.

Verbatim port of ``ECPG/core/pcgc_only.py`` — no behavioural changes.
The locked-baseline reproducibility relies on identical numerics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


EPS = 1e-6


@dataclass
class PrototypeBundle:
    high_proto: np.ndarray
    low_proto: np.ndarray
    inv_cov: np.ndarray


def compute_inv_cov(X: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """Regularized inverse covariance matrix for Mahalanobis distance."""
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    cov = np.cov(X, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=float)
    cov = np.atleast_2d(cov)
    return np.linalg.inv(cov + reg * np.eye(cov.shape[0]))


def mahalanobis_distance(x: np.ndarray, proto: np.ndarray, inv_cov: np.ndarray) -> float:
    diff = x - proto
    val = float(diff @ inv_cov @ diff)
    return float(np.sqrt(max(val, 0.0)))


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def build_train_prototypes(
    X_train_repr: np.ndarray,
    y_train: np.ndarray,
    agg: str = "mean",
    reg: float = 1e-6,
) -> PrototypeBundle:
    """Build class prototypes from training representations only (no leakage)."""
    X_high = X_train_repr[y_train == 1]
    X_low = X_train_repr[y_train == 0]

    if len(X_high) < 2 or len(X_low) < 2:
        raise ValueError(
            f"Need at least 2 samples per class. Got high={len(X_high)}, low={len(X_low)}"
        )

    if agg == "mean":
        high_proto = X_high.mean(axis=0)
        low_proto = X_low.mean(axis=0)
    elif agg == "median":
        high_proto = np.median(X_high, axis=0)
        low_proto = np.median(X_low, axis=0)
    else:
        raise ValueError(f"Unsupported agg: {agg}")

    inv_cov = compute_inv_cov(X_train_repr, reg=reg)
    return PrototypeBundle(high_proto=high_proto, low_proto=low_proto, inv_cov=inv_cov)


def compute_pcgc_scores(
    X_repr: np.ndarray,
    bundle: PrototypeBundle,
    score_fn: str = "v1",
) -> pd.DataFrame:
    """
    Compute Mahalanobis distances and PCGC score variants.

    Returns a DataFrame with columns:
      d_high, d_low, v1_score, v2_score, v3_score, ratio,
      pcgc_score (alias for v1_score), expertise_prob (sigmoid of pcgc_score)
    """
    direction = bundle.high_proto - bundle.low_proto
    direction_norm = np.linalg.norm(direction) + EPS
    direction_unit = direction / direction_norm

    rows = []
    for x in X_repr:
        d_high = mahalanobis_distance(x, bundle.high_proto, bundle.inv_cov)
        d_low = mahalanobis_distance(x, bundle.low_proto, bundle.inv_cov)

        v1 = d_low - d_high
        v2 = (d_low - d_high) / (d_low + d_high + EPS)
        v3 = float(np.dot(x - bundle.low_proto, direction_unit))
        ratio = d_high / (d_low + EPS)

        rows.append(
            {
                "d_high": d_high,
                "d_low": d_low,
                "v1_score": v1,
                "v2_score": v2,
                "v3_score": v3,
                "ratio": ratio,
                "pcgc_score": v1,          # canonical alias
                "expertise_prob": sigmoid(v1),
            }
        )

    return pd.DataFrame(rows)


def loo_balanced_pcgc_scores(
    X: np.ndarray,
    y: np.ndarray,
    agg: str = "mean",
    score_fn: str = "v1",
    seed: int = 42,
) -> np.ndarray:
    """
    Leave-one-out PCGC scoring with balanced class sampling.

    For each participant i:
      - Exclude i from its own class.
      - Down-sample majority class to match minority class size.
      - Compute prototype from balanced groups using `agg`.
      - Score using `score_fn` (v1 | v2 | v3).
      - Return sigmoid probability.
    """
    rng = np.random.RandomState(seed)
    probs = np.empty(len(y), dtype=float)

    for i in range(len(y)):
        high_idx = [j for j in range(len(y)) if y[j] == 1 and j != i]
        low_idx  = [j for j in range(len(y)) if y[j] == 0 and j != i]

        # Balance majority class down to minority size
        n_high = len(high_idx)
        if len(low_idx) > n_high:
            low_idx_bal = rng.choice(low_idx, size=n_high, replace=False).tolist()
        else:
            low_idx_bal = low_idx

        X_high = X[high_idx]
        X_low  = X[low_idx_bal]

        if agg == "mean":
            proto_high = X_high.mean(axis=0)
            proto_low  = X_low.mean(axis=0)
        elif agg == "median":
            proto_high = np.median(X_high, axis=0)
            proto_low  = np.median(X_low,  axis=0)
        else:
            raise ValueError(f"Unsupported agg: {agg}")

        all_train_idx = high_idx + low_idx_bal
        inv_cov_loo = compute_inv_cov(X[all_train_idx])

        x_i = X[i]
        d_h = mahalanobis_distance(x_i, proto_high, inv_cov_loo)
        d_l = mahalanobis_distance(x_i, proto_low,  inv_cov_loo)

        if score_fn == "v1":
            raw = d_l - d_h
        elif score_fn == "v2":
            raw = (d_l - d_h) / (d_l + d_h + EPS)
        elif score_fn == "v3":
            direction = proto_high - proto_low
            norm = np.linalg.norm(direction) + EPS
            raw = float(np.dot(x_i - proto_low, direction / norm))
        else:
            raise ValueError(f"Unknown score_fn: {score_fn}")

        probs[i] = sigmoid(raw)

    return probs
