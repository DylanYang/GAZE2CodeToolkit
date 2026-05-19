"""Gaze-based expertise classification (Stage II / ECPG).

This subpackage hosts the **interpretable, prototype-based** classification
line of the GAZE2CodeToolkit pipeline. It consumes the fixation × token
AOI table produced by :mod:`g2c.aoi` and `cli.extract_aoi`, and turns it
into participant-level expertise predictions with a strict no-leakage
participant-level CV protocol.

Public modules
--------------
- :mod:`g2c.classification.token_features`
    8-D vocabulary-agnostic token-level features per participant per question.
- :mod:`g2c.classification.pcgc`
    PCGC Mahalanobis prototype scorer (interpretable core).
- :mod:`g2c.classification.representations`
    Hybrid input representations A / A_v2 / A_v3 / B / C / D and
    fold-honest train/test feature builders.
- :mod:`g2c.classification.evaluation`
    StratifiedKFold participant-level CV with AUC / macro-F1 metrics,
    and weighted multi-task variants.
- :mod:`g2c.classification.feature_integration`
    Merge PCGC + token feature tables into a single scaled matrix.

Sourced from the ECPG research codebase (see Obsidian: *ECPG Index*).
"""
from __future__ import annotations

# Re-export the most commonly used names. Heavy submodules (sklearn /
# xgboost) are only imported lazily inside the submodules themselves —
# users who never touch classification do not pay the import cost.

__all__ = [
    "extract_token_features_from_csv",
    "TOKEN_FEAT_COLS",
    "PrototypeBundle",
    "build_train_prototypes",
    "compute_pcgc_scores",
    "load_participant_feature_csv",
    "make_model",
    "build_fold_hybrid_features",
    "run_multi_seed_experiment",
    "run_weighted_multitask_experiment",
    "run_oof_weighted_multitask_experiment",
    "summarize_results",
    "build_feature_matrix",
    "build_fold_matrices",
    "FeatureBundle",
]


def __getattr__(name: str):
    # Lazy attribute access — only imports the submodule once.
    if name in {"extract_token_features_from_csv", "TOKEN_FEAT_COLS"}:
        from .token_features import extract_token_features_from_csv, TOKEN_FEAT_COLS
        return {"extract_token_features_from_csv": extract_token_features_from_csv,
                "TOKEN_FEAT_COLS": TOKEN_FEAT_COLS}[name]
    if name in {"PrototypeBundle", "build_train_prototypes", "compute_pcgc_scores"}:
        from .pcgc import PrototypeBundle, build_train_prototypes, compute_pcgc_scores
        return {"PrototypeBundle": PrototypeBundle,
                "build_train_prototypes": build_train_prototypes,
                "compute_pcgc_scores": compute_pcgc_scores}[name]
    if name in {"load_participant_feature_csv", "make_model", "build_fold_hybrid_features"}:
        from .representations import (
            load_participant_feature_csv, make_model, build_fold_hybrid_features,
        )
        return {"load_participant_feature_csv": load_participant_feature_csv,
                "make_model": make_model,
                "build_fold_hybrid_features": build_fold_hybrid_features}[name]
    if name in {"run_multi_seed_experiment", "run_weighted_multitask_experiment",
                "run_oof_weighted_multitask_experiment", "summarize_results"}:
        from .evaluation import (
            run_multi_seed_experiment, run_weighted_multitask_experiment,
            run_oof_weighted_multitask_experiment, summarize_results,
        )
        return {"run_multi_seed_experiment": run_multi_seed_experiment,
                "run_weighted_multitask_experiment": run_weighted_multitask_experiment,
                "run_oof_weighted_multitask_experiment": run_oof_weighted_multitask_experiment,
                "summarize_results": summarize_results}[name]
    if name in {"build_feature_matrix", "build_fold_matrices", "FeatureBundle"}:
        from .feature_integration import build_feature_matrix, build_fold_matrices, FeatureBundle
        return {"build_feature_matrix": build_feature_matrix,
                "build_fold_matrices": build_fold_matrices,
                "FeatureBundle": FeatureBundle}[name]
    raise AttributeError(f"module 'g2c.classification' has no attribute {name!r}")
