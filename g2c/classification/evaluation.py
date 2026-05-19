"""Participant-level CV for PCGC / hybrid expertise classification.

Ported from ``ECPG/core/evaluation.py``. Only the two upstream imports
were rewritten to point at ``g2c.classification.*``; every CV split,
metric computation, and weight aggregation remains bit-identical to the
original so the locked XGBoost baseline reproduces.

The no-leakage protocol — *prototypes built on train only, scaler fit
on train only, OOF AUC estimated on inner-train folds only* — is the
fragile invariant of this file. Do not touch the order of
fit/transform/predict in any of the loops below.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from g2c.classification.pcgc import build_train_prototypes, compute_pcgc_scores
from g2c.classification.representations import build_fold_hybrid_features, make_model


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict:
    """
    Compute AUC, accuracy, macro-F1, and confusion matrix from predicted probabilities.
    """
    y_pred = (y_prob >= threshold).astype(int)

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return {
        "auc": auc,
        "accuracy": acc,
        "macro_f1": f1,
        "confusion_matrix": cm,
    }


def run_single_seed_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    representation: str,
    model_name: str,
    agg: str,
    n_splits: int,
    seed: int,
    reg: float = 1e-6,
    lr_C: float = 1.0,
) -> Dict:
    """
    Honest participant-level CV for hybrid experiments:
    1. split participants by StratifiedKFold
    2. build fold-wise train/test PCGC-derived features
    3. train downstream model on training fold only
    4. predict probabilities on test fold
    """
    y = df["expertise"].to_numpy(dtype=int)
    dummy_X = np.zeros((len(df), 1), dtype=float)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_probs = np.zeros(len(df), dtype=float)

    for train_idx, test_idx in skf.split(dummy_X, y):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        Z_train, y_train, Z_test = build_fold_hybrid_features(
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            representation=representation,
            agg=agg,
            reg=reg,
        )

        model = make_model(model_name, seed=seed, y_train=y_train, lr_C=lr_C)
        model.fit(Z_train, y_train)
        all_probs[test_idx] = model.predict_proba(Z_test)[:, 1]

    metrics = compute_metrics(y, all_probs)
    return {
        "seed": seed,
        "representation": representation,
        "model_name": model_name,
        "aggregation": agg,
        "reg": reg,
        "lr_C": lr_C,
        **metrics,
    }


def run_multi_seed_experiment(
    df: pd.DataFrame,
    feature_cols: List[str],
    representation: str,
    model_name: str,
    agg: str,
    seeds: List[int],
    n_splits: int,
    reg: float = 1e-6,
    lr_C: float = 1.0,
) -> pd.DataFrame:
    """
    Run honest hybrid CV across multiple random seeds.
    """
    rows = []

    for seed in seeds:
        row = run_single_seed_cv(
            df=df,
            feature_cols=feature_cols,
            representation=representation,
            model_name=model_name,
            agg=agg,
            n_splits=n_splits,
            seed=seed,
            reg=reg,
            lr_C=lr_C,
        )
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize mean ± std across seeds.
    """
    group_cols = ["representation", "model_name", "aggregation"]
    for col in ["reg", "lr_C"]:
        if col in results_df.columns:
            group_cols.append(col)
    grouped = (
        results_df
        .groupby(group_cols, as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            acc_mean=("accuracy", "mean"),
            acc_std=("accuracy", "std"),
            f1_mean=("macro_f1", "mean"),
            f1_std=("macro_f1", "std"),
        )
        .sort_values(["auc_mean", "f1_mean"], ascending=False)
        .reset_index(drop=True)
    )
    return grouped


# ---------------------------------------------------------------------------
# Weighted multi-task aggregation helpers
# ---------------------------------------------------------------------------

def _compute_q_weights(train_aucs: List[float], weighting: str) -> np.ndarray:
    """
    Convert per-question training-fold AUCs into normalised nonneg weights.

    Strategies:
      uniform   — equal weight for every question
      w1        — weight_q = max(auc_q - 0.5, 0)            (ReLU)
      w2_<a>    — weight_q = exp(a * (auc_q - 0.5))         (softmax temperature a)
    """
    aucs = np.array(train_aucs, dtype=float)

    if weighting == "uniform":
        w = np.ones(len(aucs))
    elif weighting == "w1":
        w = np.maximum(aucs - 0.5, 0.0)
    elif weighting.startswith("w2_"):
        alpha = float(weighting.split("_")[1])
        w = np.exp(alpha * (aucs - 0.5))
    else:
        raise ValueError(f"Unknown weighting: {weighting}")

    total = w.sum()
    if total < 1e-12:
        w = np.ones(len(aucs)) / len(aucs)   # uniform fallback
    else:
        w = w / total
    return w


def run_weighted_multitask_single_seed_cv(
    q_dfs: Dict[str, pd.DataFrame],
    feat_cols: List[str],
    feature_type: str,
    weighting: str,
    model_name: str,
    agg: str,
    n_splits: int,
    seed: int,
    reg: float = 1e-6,
    lr_C: float = 1.0,
) -> Dict:
    """
    Honest participant-level CV with per-fold, training-only question weighting.

    feature_type:
      "repr_A"  — per-question PCGC [d_high, d_low, v1_score, ratio], dim=4
      "direct"  — per-question raw statistical features, dim=len(feat_cols)

    For BOTH types the per-question discriminability estimate uses PCGC v1_score
    on the training fold only → no test leakage.
    """
    q_names = sorted(q_dfs.keys())

    # Align all questions to a common sorted participant list
    common_pids = sorted(
        set.intersection(*[set(df["p_id"].unique()) for df in q_dfs.values()])
    )

    q_aligned: Dict[str, pd.DataFrame] = {}
    for q in q_names:
        df = q_dfs[q]
        df_q = (
            df[df["p_id"].isin(common_pids)]
            .set_index("p_id")
            .loc[common_pids]
            .reset_index()
        )
        q_aligned[q] = df_q

    y = q_aligned[q_names[0]]["expertise"].to_numpy(dtype=int)
    dummy_X = np.zeros((len(common_pids), 1))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_probs = np.zeros(len(common_pids), dtype=float)

    for train_idx, test_idx in skf.split(dummy_X, y):
        y_train = y[train_idx]

        q_train_feats: List[np.ndarray] = []
        q_test_feats: List[np.ndarray] = []
        train_aucs: List[float] = []

        for q in q_names:
            X_q = q_aligned[q][feat_cols].to_numpy(dtype=float)
            X_q_tr = X_q[train_idx]
            X_q_te = X_q[test_idx]

            # Build PCGC on this question's training fold (no leakage)
            try:
                bundle = build_train_prototypes(X_q_tr, y_train, agg=agg, reg=reg)
            except ValueError:
                # Fallback if a class has <2 samples
                bundle = None

            if bundle is not None:
                pcgc_tr = compute_pcgc_scores(X_q_tr, bundle, score_fn="v1")
                pcgc_te = compute_pcgc_scores(X_q_te, bundle, score_fn="v1")
                try:
                    q_auc = float(roc_auc_score(y_train, pcgc_tr["v1_score"].to_numpy()))
                except Exception:
                    q_auc = 0.5
            else:
                q_auc = 0.5

            if feature_type == "repr_A":
                if bundle is not None:
                    Z_tr = pcgc_tr[["d_high", "d_low", "v1_score", "ratio"]].to_numpy()
                    Z_te = pcgc_te[["d_high", "d_low", "v1_score", "ratio"]].to_numpy()
                else:
                    Z_tr = np.zeros((len(train_idx), 4))
                    Z_te = np.zeros((len(test_idx), 4))
            else:  # "direct"
                Z_tr = X_q_tr
                Z_te = X_q_te

            q_train_feats.append(Z_tr)
            q_test_feats.append(Z_te)
            train_aucs.append(q_auc)

        # Fold-specific weights derived from training data only
        weights = _compute_q_weights(train_aucs, weighting)

        # Weighted mean across questions → (n_participants, d)
        Z_train = sum(w * Z for w, Z in zip(weights, q_train_feats))
        Z_test  = sum(w * Z for w, Z in zip(weights, q_test_feats))

        scaler = StandardScaler()
        Z_train = scaler.fit_transform(Z_train)
        Z_test  = scaler.transform(Z_test)

        model = make_model(model_name, seed=seed, y_train=y_train, lr_C=lr_C)
        model.fit(Z_train, y_train)
        all_probs[test_idx] = model.predict_proba(Z_test)[:, 1]

    metrics = compute_metrics(y, all_probs)
    return {
        "seed": seed,
        "feature_type": feature_type,
        "weighting": weighting,
        "model_name": model_name,
        "aggregation": agg,
        "reg": reg,
        "lr_C": lr_C,
        **metrics,
    }


def run_weighted_multitask_experiment(
    q_dfs: Dict[str, pd.DataFrame],
    feat_cols: List[str],
    feature_type: str,
    weighting: str,
    model_name: str,
    agg: str,
    seeds: List[int],
    n_splits: int,
    reg: float = 1e-6,
    lr_C: float = 1.0,
) -> pd.DataFrame:
    """Run weighted multi-task CV across multiple random seeds."""
    rows = []
    for seed in seeds:
        row = run_weighted_multitask_single_seed_cv(
            q_dfs=q_dfs,
            feat_cols=feat_cols,
            feature_type=feature_type,
            weighting=weighting,
            model_name=model_name,
            agg=agg,
            n_splits=n_splits,
            seed=seed,
            reg=reg,
            lr_C=lr_C,
        )
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# OOF-based task-quality estimation
# ---------------------------------------------------------------------------

def _estimate_oof_q_aucs(
    q_train_arrays: Dict[str, np.ndarray],
    y_train: np.ndarray,
    inner_splits: int = 3,
    inner_seed: int = 0,
    agg: str = "mean",
    reg: float = 1e-6,
) -> Dict[str, float]:
    """
    Estimate per-question unbiased AUC using inner OOF CV on the outer
    training fold only.

    For each question:
      - split outer-training participants into inner_splits inner folds
      - build PCGC on each inner training fold
      - collect OOF v1_score predictions on inner test participants
      - compute AUC of OOF predictions vs y_train  →  unbiased quality estimate

    All computation uses outer-training data only: no test leakage.
    """
    n_train = len(y_train)
    inner_skf = StratifiedKFold(
        n_splits=inner_splits, shuffle=True, random_state=inner_seed
    )
    dummy = np.zeros((n_train, 1))

    q_oof_aucs: Dict[str, float] = {}

    for q_name, X_q in q_train_arrays.items():
        oof_scores = np.zeros(n_train, dtype=float)

        for i_tr_idx, i_te_idx in inner_skf.split(dummy, y_train):
            y_i_tr = y_train[i_tr_idx]
            X_i_tr = X_q[i_tr_idx]
            X_i_te = X_q[i_te_idx]

            try:
                bundle = build_train_prototypes(X_i_tr, y_i_tr, agg=agg, reg=reg)
                pcgc_te = compute_pcgc_scores(X_i_te, bundle, score_fn="v1")
                oof_scores[i_te_idx] = pcgc_te["v1_score"].to_numpy()
            except Exception:
                oof_scores[i_te_idx] = 0.0

        try:
            q_oof_aucs[q_name] = float(roc_auc_score(y_train, oof_scores))
        except Exception:
            q_oof_aucs[q_name] = 0.5

    return q_oof_aucs


def run_oof_weighted_multitask_single_seed_cv(
    q_dfs: Dict[str, pd.DataFrame],
    feat_cols: List[str],
    feature_type: str,
    weighting: str,
    model_name: str,
    agg: str,
    n_splits: int,
    seed: int,
    reg: float = 1e-6,
    lr_C: float = 1.0,
    inner_splits: int = 3,
) -> Dict:
    """
    Honest participant-level CV with OOF-estimated per-question task weights.

    feature_type:
      "repr_A"      — weighted mean of [d_high, d_low, v1_score, ratio]  → 4-dim
      "direct"      — weighted mean of raw direct stats                   → 8-dim
      "score_stack" — stack of per-question v1_scores, scaled by weights  → k-dim

    weighting:
      "uniform"     — equal weights (no OOF estimation needed)
      "oof_relu"    — weight_q = max(OOF_AUC_q - 0.5, 0)
      "oof_soft_<a>"— weight_q = exp(a * (OOF_AUC_q - 0.5))

    OOF AUCs are estimated by inner_splits-fold CV on the outer training fold.
    The same inner_seed = seed is used for reproducibility.
    """
    q_names = sorted(q_dfs.keys())

    # Align all questions to the same sorted participant order
    common_pids = sorted(
        set.intersection(*[set(df["p_id"].unique()) for df in q_dfs.values()])
    )
    q_aligned: Dict[str, pd.DataFrame] = {}
    for q in q_names:
        df = q_dfs[q]
        q_aligned[q] = (
            df[df["p_id"].isin(common_pids)]
            .set_index("p_id")
            .loc[common_pids]
            .reset_index()
        )

    y = q_aligned[q_names[0]]["expertise"].to_numpy(dtype=int)
    dummy_X = np.zeros((len(common_pids), 1))

    outer_skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_probs = np.zeros(len(common_pids), dtype=float)

    for train_idx, test_idx in outer_skf.split(dummy_X, y):
        y_train = y[train_idx]

        # ---- Step 1: build per-question feature matrices for this outer fold ----
        q_X_train: Dict[str, np.ndarray] = {}
        q_X_test:  Dict[str, np.ndarray] = {}
        for q in q_names:
            X_q = q_aligned[q][feat_cols].to_numpy(dtype=float)
            q_X_train[q] = X_q[train_idx]
            q_X_test[q]  = X_q[test_idx]

        # ---- Step 2: estimate unbiased OOF AUC per question ----
        if weighting == "uniform":
            q_oof_aucs = {q: 0.5 for q in q_names}   # unused but keeps code uniform
        else:
            q_oof_aucs = _estimate_oof_q_aucs(
                q_train_arrays=q_X_train,
                y_train=y_train,
                inner_splits=inner_splits,
                inner_seed=seed,
                agg=agg,
                reg=reg,
            )

        # ---- Step 3: convert OOF AUCs to normalized weights ----
        oof_auc_list = [q_oof_aucs[q] for q in q_names]

        if weighting == "uniform":
            raw_w = np.ones(len(q_names))
        elif weighting == "oof_relu":
            raw_w = np.maximum(np.array(oof_auc_list) - 0.5, 0.0)
        elif weighting.startswith("oof_soft_"):
            alpha = float(weighting.split("_")[2])
            raw_w = np.exp(alpha * (np.array(oof_auc_list) - 0.5))
        else:
            raise ValueError(f"Unknown OOF weighting: {weighting}")

        total = raw_w.sum()
        weights = raw_w / total if total > 1e-12 else np.ones(len(q_names)) / len(q_names)

        # ---- Step 4: build per-question representations (train + test) ----
        q_train_feats: List[np.ndarray] = []
        q_test_feats:  List[np.ndarray] = []

        for q in q_names:
            X_q_tr = q_X_train[q]
            X_q_te = q_X_test[q]

            try:
                bundle = build_train_prototypes(X_q_tr, y_train, agg=agg, reg=reg)
            except ValueError:
                bundle = None

            if feature_type in ("repr_A", "repr_A_concat"):
                if bundle is not None:
                    ptr = compute_pcgc_scores(X_q_tr, bundle, score_fn="v1")
                    pte = compute_pcgc_scores(X_q_te, bundle, score_fn="v1")
                    Z_tr = ptr[["d_high", "d_low", "v1_score", "ratio"]].to_numpy()
                    Z_te = pte[["d_high", "d_low", "v1_score", "ratio"]].to_numpy()
                else:
                    Z_tr = np.zeros((len(train_idx), 4))
                    Z_te = np.zeros((len(test_idx),  4))

            elif feature_type == "direct":
                Z_tr = X_q_tr
                Z_te = X_q_te

            elif feature_type in ("score_stack", "score_concat"):
                # Each question contributes one scalar: v1_score (PCGC discriminability)
                if bundle is not None:
                    ptr = compute_pcgc_scores(X_q_tr, bundle, score_fn="v1")
                    pte = compute_pcgc_scores(X_q_te, bundle, score_fn="v1")
                    Z_tr = ptr[["v1_score"]].to_numpy()
                    Z_te = pte[["v1_score"]].to_numpy()
                else:
                    Z_tr = np.zeros((len(train_idx), 1))
                    Z_te = np.zeros((len(test_idx),  1))

            else:
                raise ValueError(f"Unknown feature_type: {feature_type}")

            q_train_feats.append(Z_tr)
            q_test_feats.append(Z_te)

        # ---- Step 5: aggregate ----
        if feature_type in ("score_stack", "repr_A_concat", "score_concat"):
            # Horizontal concatenation: shape (n, k*d); no weighting applied
            Z_train = np.hstack(q_train_feats)
            Z_test  = np.hstack(q_test_feats)
        else:
            # Weighted mean: shape (n, d)
            Z_train = sum(w * Z for w, Z in zip(weights, q_train_feats))
            Z_test  = sum(w * Z for w, Z in zip(weights, q_test_feats))

        # ---- Step 6: scale, train, predict ----
        scaler = StandardScaler()
        Z_train = scaler.fit_transform(Z_train)
        Z_test  = scaler.transform(Z_test)

        model = make_model(model_name, seed=seed, y_train=y_train, lr_C=lr_C)
        model.fit(Z_train, y_train)
        all_probs[test_idx] = model.predict_proba(Z_test)[:, 1]

    metrics = compute_metrics(y, all_probs)
    return {
        "seed": seed,
        "feature_type": feature_type,
        "weighting": weighting,
        "model_name": model_name,
        "aggregation": agg,
        "reg": reg,
        "lr_C": lr_C,
        **metrics,
    }


def run_oof_weighted_multitask_experiment(
    q_dfs: Dict[str, pd.DataFrame],
    feat_cols: List[str],
    feature_type: str,
    weighting: str,
    model_name: str,
    agg: str,
    seeds: List[int],
    n_splits: int,
    reg: float = 1e-6,
    lr_C: float = 1.0,
    inner_splits: int = 3,
) -> pd.DataFrame:
    """Run OOF-weighted multi-task CV across multiple seeds."""
    rows = []
    for seed in seeds:
        row = run_oof_weighted_multitask_single_seed_cv(
            q_dfs=q_dfs,
            feat_cols=feat_cols,
            feature_type=feature_type,
            weighting=weighting,
            model_name=model_name,
            agg=agg,
            n_splits=n_splits,
            seed=seed,
            reg=reg,
            lr_C=lr_C,
            inner_splits=inner_splits,
        )
        rows.append(row)
    return pd.DataFrame(rows)
