"""Merge PCGC + token-level participant features into one scaled matrix.

Verbatim port of ``ECPG/feature_integration.py``. No behavioural change.

- Inner-join on ``p_id`` (participants missing from either table are dropped).
- Token features → RobustScaler (median/IQR; robust to outliers).
- Concatenated matrix → StandardScaler (z-scoring).
- For cross-validation, use :func:`build_fold_matrices` so the scalers are
  fit on the *training fold only*, preserving the no-leakage protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PID_COL = "p_id"
_LABEL_COL = "expertise"
_SKIP_COLS = {_PID_COL, _LABEL_COL}


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class FeatureBundle:
    """Result of build_feature_matrix."""
    X: np.ndarray                    # scaled feature matrix  (n_participants, n_features)
    y: np.ndarray                    # integer labels          (n_participants,)
    pids: List[str]                  # aligned participant IDs (n_participants,)
    feature_names: List[str]         # column names for X
    token_scaler: RobustScaler       # fitted on token columns
    final_scaler: StandardScaler     # fitted on concatenated matrix
    n_pcgc_features: int             # number of pcgc feature columns
    n_token_features: int            # number of token feature columns
    dropped_pids: List[str] = field(default_factory=list)  # participants excluded


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _normalise_pid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[_PID_COL] = df[_PID_COL].astype(str).str.strip().str.lower()
    return df


def _feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in _SKIP_COLS]


def _to_float(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _align(
    pcgc: pd.DataFrame,
    token: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Inner-join on p_id.  Returns (pcgc_aligned, token_aligned, dropped_pids).
    """
    pcgc_pids = set(pcgc[_PID_COL])
    token_pids = set(token[_PID_COL])
    common = pcgc_pids & token_pids
    dropped = sorted((pcgc_pids | token_pids) - common)

    pcgc_al = pcgc[pcgc[_PID_COL].isin(common)].copy()
    token_al = token[token[_PID_COL].isin(common)].copy()

    # Align row order
    pcgc_al = pcgc_al.sort_values(_PID_COL).reset_index(drop=True)
    token_al = token_al.sort_values(_PID_COL).reset_index(drop=True)

    assert (pcgc_al[_PID_COL].values == token_al[_PID_COL].values).all(), (
        "p_id order mismatch after sort — check for duplicate participants"
    )
    return pcgc_al, token_al, dropped


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_feature_matrix(
    pcgc_features: pd.DataFrame,
    token_features: pd.DataFrame,
    token_scaler: Optional[RobustScaler] = None,
    final_scaler: Optional[StandardScaler] = None,
) -> FeatureBundle:
    """
    Merge and scale PCGC and token-level features into one matrix.

    Parameters
    ----------
    pcgc_features : pd.DataFrame
        Participant-level PCGC / representation features.
        Must contain ``p_id``, ``expertise``, and at least one numeric column.
    token_features : pd.DataFrame
        Participant-level token features (e.g. from token_features.py).
        Must contain ``p_id`` and at least one numeric column.
        ``expertise`` is used for cross-check if present, otherwise taken
        from pcgc_features.
    token_scaler : RobustScaler, optional
        Pre-fitted scaler for token columns.  If None, a new one is fit on
        the supplied data (use for training; pass the returned scaler for test).
    final_scaler : StandardScaler, optional
        Pre-fitted scaler for the concatenated matrix.  If None, a new one
        is fit on the supplied data.

    Returns
    -------
    FeatureBundle
        Contains X, y, pids, feature_names, fitted scalers, and diagnostics.
    """
    # --- Validate inputs ---
    for name, df in [("pcgc_features", pcgc_features), ("token_features", token_features)]:
        if _PID_COL not in df.columns:
            raise ValueError(f"'{name}' is missing required column '{_PID_COL}'")
    if _LABEL_COL not in pcgc_features.columns:
        raise ValueError(f"'pcgc_features' is missing required column '{_LABEL_COL}'")

    # --- Normalise p_id ---
    pcgc = _normalise_pid(pcgc_features)
    tok = _normalise_pid(token_features)

    # --- Coerce feature columns to float, drop NaN rows ---
    pcgc_cols = _feature_cols(pcgc)
    tok_cols = _feature_cols(tok)

    if not pcgc_cols:
        raise ValueError("'pcgc_features' has no feature columns (only p_id / expertise).")
    if not tok_cols:
        raise ValueError("'token_features' has no feature columns (only p_id / expertise).")

    pcgc = _to_float(pcgc, pcgc_cols).dropna(subset=[_PID_COL, _LABEL_COL] + pcgc_cols)
    tok = _to_float(tok, tok_cols).dropna(subset=[_PID_COL] + tok_cols)

    # --- Participant alignment (inner join) ---
    pcgc_al, tok_al, dropped = _align(pcgc, tok)

    if len(pcgc_al) == 0:
        raise ValueError(
            "No participants remain after alignment — "
            f"pcgc had {len(pcgc)}, token had {len(tok)}, intersection is empty."
        )

    # --- Extract arrays ---
    pids = pcgc_al[_PID_COL].tolist()
    y = pcgc_al[_LABEL_COL].to_numpy(dtype=int)

    X_pcgc = pcgc_al[pcgc_cols].to_numpy(dtype=float)
    X_tok = tok_al[tok_cols].to_numpy(dtype=float)

    # --- Scale token features with RobustScaler ---
    if token_scaler is None:
        token_scaler = RobustScaler()
        X_tok_scaled = token_scaler.fit_transform(X_tok)
    else:
        X_tok_scaled = token_scaler.transform(X_tok)

    # --- Concatenate ---
    X_concat = np.concatenate([X_pcgc, X_tok_scaled], axis=1)
    feature_names = pcgc_cols + tok_cols

    # --- Scale final matrix with StandardScaler ---
    if final_scaler is None:
        final_scaler = StandardScaler()
        X_scaled = final_scaler.fit_transform(X_concat)
    else:
        X_scaled = final_scaler.transform(X_concat)

    return FeatureBundle(
        X=X_scaled,
        y=y,
        pids=pids,
        feature_names=feature_names,
        token_scaler=token_scaler,
        final_scaler=final_scaler,
        n_pcgc_features=len(pcgc_cols),
        n_token_features=len(tok_cols),
        dropped_pids=dropped,
    )


def build_fold_matrices(
    train_pcgc: pd.DataFrame,
    train_token: pd.DataFrame,
    test_pcgc: pd.DataFrame,
    test_token: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fold-honest integration: scalers are fit only on train, then applied to test.

    Parameters
    ----------
    train_pcgc, train_token : pd.DataFrame
        Training participants for PCGC and token features.
    test_pcgc, test_token : pd.DataFrame
        Test participants for PCGC and token features.

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
        Scaled feature matrices and labels for train and test.
    """
    train_bundle = build_feature_matrix(train_pcgc, train_token)
    test_bundle = build_feature_matrix(
        test_pcgc,
        test_token,
        token_scaler=train_bundle.token_scaler,
        final_scaler=train_bundle.final_scaler,
    )
    return train_bundle.X, train_bundle.y, test_bundle.X, test_bundle.y
