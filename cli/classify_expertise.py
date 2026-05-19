"""Train and evaluate gaze-based expertise classifiers (PCGC / hybrid).

This CLI is the toolkit's entry point for Stage II — classification on
participant-level features built from the fixation × token AOI tables
emitted by :mod:`cli.extract_aoi`.

Two subcommands
---------------

``build-features``
    Roll up a directory of fixation × token CSVs (one per question, e.g.
    ``Q1.csv … Q5.csv``) into per-question ``participant_features_<Q>_token.csv``
    files containing 8 vocabulary-agnostic token statistics per
    participant (Shannon entropy, Gini, top-3 fraction, …).

``train``
    Run multi-seed, participant-level Stratified-K-Fold CV with the
    PCGC prototype scorer plus a downstream classifier
    (``lr | linsvm | xgb``). Supports both single-task mode
    (``--input <per-question.csv>``) and weighted multi-task mode
    (``--multitask-dir <dir>``).

Examples
--------
    # 1) Token-level features for every question
    python -m cli.classify_expertise build-features \\
        --raw-dir  output/unl_um/group/aoi/labelled \\
        --out-dir  output/unl_um/classification

    # 2) Single-task classification (representation A, XGBoost, 9 seeds)
    python -m cli.classify_expertise train \\
        --input output/unl_um/classification/participant_features_Q1_token.csv \\
        --representation A --model xgb \\
        --output-dir output/unl_um/classification/results

    # 3) Weighted multi-task across all 7 questions
    python -m cli.classify_expertise train \\
        --multitask-dir output/unl_um/classification \\
        --feature-type repr_A --weighting w1 --model xgb \\
        --output-dir output/unl_um/classification/results
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from g2c.classification.evaluation import (
    run_multi_seed_experiment,
    run_oof_weighted_multitask_experiment,
    run_weighted_multitask_experiment,
    summarize_results,
)
from g2c.classification.representations import load_participant_feature_csv
from g2c.classification.token_features import (
    QUESTIONS,
    extract_token_features_from_csv,
)


# ---------------------------------------------------------------------------
# build-features
# ---------------------------------------------------------------------------

def _add_build_features_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "build-features",
        help="Aggregate per-question fixation×token CSVs into participant-level "
             "token-feature CSVs.",
    )
    p.add_argument("--raw-dir", required=True, type=Path,
                   help="Directory containing one CSV per question (Q*.csv). "
                        "Each CSV must contain columns p_id, expertise, duration, aoi_token.")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Directory to write participant_features_<Q>_token.csv files.")
    p.add_argument("--questions", nargs="+", default=QUESTIONS,
                   help=f"Question IDs to process. Default: {QUESTIONS}.")
    p.set_defaults(func=_cmd_build_features)


def _cmd_build_features(args: argparse.Namespace) -> int:
    raw_dir = args.raw_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n_done = 0
    for q in args.questions:
        raw_csv = raw_dir / f"{q}.csv"
        if not raw_csv.exists():
            print(f"  SKIP {q} — file not found: {raw_csv}", file=sys.stderr)
            continue
        feat_df = extract_token_features_from_csv(str(raw_csv))
        out_path = out_dir / f"participant_features_{q}_token.csv"
        feat_df.to_csv(out_path, index=False)
        print(f"  {q}: {len(feat_df):>3d} participants → {out_path}", file=sys.stderr)
        n_done += 1

    if n_done == 0:
        print("[classify_expertise] no question CSVs found — nothing written.",
              file=sys.stderr)
        return 2
    return 0


# ---------------------------------------------------------------------------
# train (single-task and multi-task)
# ---------------------------------------------------------------------------

def _add_train_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "train",
        help="Run participant-level CV with PCGC + downstream classifier.",
    )

    # --- mode selection ---
    p.add_argument("--input", type=Path, default=None,
                   help="Path to participant-level CSV (single-task mode).")
    p.add_argument("--multitask-dir", type=Path, default=None,
                   help="Directory with per-question feature CSVs "
                        "(participant_features_<Q>_<suffix>.csv). "
                        "When set, runs the weighted multi-task pipeline instead.")

    # --- single-task knobs ---
    p.add_argument("--representation", type=str, default="A",
                   choices=["A", "A_v2", "A_v3", "B", "C", "D"],
                   help="Hybrid representation (single-task mode).")

    # --- multi-task knobs ---
    p.add_argument("--multitask-questions", nargs="+",
                   default=["Q1", "Q2A", "Q2B", "Q3", "Q4A", "Q4B", "Q5"],
                   help="Question IDs to include in multi-task aggregation.")
    p.add_argument("--multitask-suffix", type=str, default="token",
                   help="Filename suffix that selects which per-question CSVs to "
                        "load (e.g. 'token' → participant_features_Q1_token.csv).")
    p.add_argument("--feature-type", type=str, default="repr_A",
                   choices=["repr_A", "direct", "score_stack",
                            "repr_A_concat", "score_concat"],
                   help="Per-question feature type for multi-task mode.")
    p.add_argument("--weighting", type=str, default="w1",
                   help="Multi-task weighting: uniform | w1 | w2_<a> | "
                        "oof_relu | oof_soft_<a>.")
    p.add_argument("--inner-splits", type=int, default=3,
                   help="Inner folds for OOF task-quality estimation.")

    # --- shared knobs ---
    p.add_argument("--model", type=str, default="lr",
                   choices=["lr", "linsvm", "xgb"],
                   help="Downstream classifier.")
    p.add_argument("--agg", type=str, default="mean",
                   choices=["mean", "median"],
                   help="Prototype aggregation.")
    p.add_argument("--seeds", nargs="+", type=int,
                   default=[42, 43, 44, 45, 46, 47, 48, 49, 50],
                   help="Random seeds for multi-seed evaluation.")
    p.add_argument("--n-splits", type=int, default=5,
                   help="Number of StratifiedKFold splits.")
    p.add_argument("--reg", type=float, default=1e-6,
                   help="Covariance regularisation for Mahalanobis distance.")
    p.add_argument("--lr-c", type=float, default=1.0,
                   help="Logistic Regression C parameter.")
    p.add_argument("--output-dir", type=Path, default=Path("results"),
                   help="Directory for per-seed and summary CSVs.")

    p.set_defaults(func=_cmd_train)


def _cmd_train(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.multitask_dir is not None:
        return _train_multitask(args)
    if args.input is None:
        print("[classify_expertise] either --input or --multitask-dir is required.",
              file=sys.stderr)
        return 2
    return _train_single(args)


def _train_single(args: argparse.Namespace) -> int:
    df, feature_cols = load_participant_feature_csv(
        csv_path=str(args.input), pid_col="p_id", label_col="expertise",
    )

    print("=" * 79, file=sys.stderr)
    print("SINGLE-TASK CLASSIFICATION", file=sys.stderr)
    print("=" * 79, file=sys.stderr)
    print(f"Input file          : {args.input}", file=sys.stderr)
    print(f"Participants        : {len(df)}", file=sys.stderr)
    print(f"High (1) / Low (0)  : "
          f"{(df['expertise'] == 1).sum()} / {(df['expertise'] == 0).sum()}",
          file=sys.stderr)
    print(f"Feature dims        : {len(feature_cols)}", file=sys.stderr)
    print(f"Representation      : {args.representation}", file=sys.stderr)
    print(f"Model               : {args.model}", file=sys.stderr)
    print(f"Seeds × splits      : {len(args.seeds)} × {args.n_splits}",
          file=sys.stderr)

    results_df = run_multi_seed_experiment(
        df=df, feature_cols=feature_cols,
        representation=args.representation,
        model_name=args.model, agg=args.agg,
        seeds=args.seeds, n_splits=args.n_splits,
        reg=args.reg, lr_C=args.lr_c,
    )
    summary_df = summarize_results(results_df)

    stem = (
        f"repr_{args.representation}_model_{args.model}"
        f"_agg_{args.agg}_reg{args.reg}_lrc{args.lr_c}"
    )
    _emit_results(results_df, summary_df, stem, args.output_dir)
    return 0


def _train_multitask(args: argparse.Namespace) -> int:
    q_dfs: dict[str, pd.DataFrame] = {}
    feat_cols: list[str] | None = None
    suffix = args.multitask_suffix

    for q in args.multitask_questions:
        csv_path = args.multitask_dir / f"participant_features_{q}_{suffix}.csv"
        if not csv_path.exists():
            print(f"[classify_expertise] missing per-question file: {csv_path}",
                  file=sys.stderr)
            return 2
        df_q, fc = load_participant_feature_csv(str(csv_path))
        q_dfs[q] = df_q
        if feat_cols is None:
            feat_cols = fc

    common = set.intersection(*[set(df["p_id"].unique()) for df in q_dfs.values()])

    print("=" * 79, file=sys.stderr)
    print("WEIGHTED MULTI-TASK CLASSIFICATION", file=sys.stderr)
    print("=" * 79, file=sys.stderr)
    print(f"Questions           : {args.multitask_questions}", file=sys.stderr)
    print(f"Common participants : {len(common)}", file=sys.stderr)
    print(f"Feature type        : {args.feature_type}", file=sys.stderr)
    print(f"Weighting           : {args.weighting}", file=sys.stderr)
    print(f"Model               : {args.model}", file=sys.stderr)
    print(f"Seeds × splits      : {len(args.seeds)} × {args.n_splits}",
          file=sys.stderr)

    is_oof = (
        args.weighting.startswith("oof_")
        or args.feature_type in ("score_stack", "repr_A_concat", "score_concat")
    )

    if is_oof:
        print(f"Inner splits (OOF)  : {args.inner_splits}", file=sys.stderr)
        results_df = run_oof_weighted_multitask_experiment(
            q_dfs=q_dfs, feat_cols=feat_cols,
            feature_type=args.feature_type, weighting=args.weighting,
            model_name=args.model, agg=args.agg,
            seeds=args.seeds, n_splits=args.n_splits,
            reg=args.reg, lr_C=args.lr_c, inner_splits=args.inner_splits,
        )
    else:
        results_df = run_weighted_multitask_experiment(
            q_dfs=q_dfs, feat_cols=feat_cols,
            feature_type=args.feature_type, weighting=args.weighting,
            model_name=args.model, agg=args.agg,
            seeds=args.seeds, n_splits=args.n_splits,
            reg=args.reg, lr_C=args.lr_c,
        )

    grp_cols = ["feature_type", "weighting", "model_name",
                "aggregation", "reg", "lr_C"]
    summary_df = (
        results_df
        .groupby(grp_cols, as_index=False)
        .agg(
            auc_mean=("auc", "mean"), auc_std=("auc", "std"),
            acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
            f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
        )
        .sort_values(["auc_mean", "f1_mean"], ascending=False)
        .reset_index(drop=True)
    )

    stem = (
        f"mt_{args.feature_type}_{args.weighting}"
        f"_model_{args.model}_agg_{args.agg}_reg{args.reg}_lrc{args.lr_c}"
    )
    _emit_results(results_df, summary_df, stem, args.output_dir)
    return 0


def _emit_results(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    stem: str,
    output_dir: Path,
) -> None:
    print("\n" + "=" * 79, file=sys.stderr)
    print("PER-SEED RESULTS", file=sys.stderr)
    print("=" * 79, file=sys.stderr)
    print(results_df.to_string(index=False), file=sys.stderr)

    print("\n" + "=" * 79, file=sys.stderr)
    print("SUMMARY", file=sys.stderr)
    print("=" * 79, file=sys.stderr)
    print(summary_df.to_string(index=False), file=sys.stderr)

    per_seed_path = output_dir / f"{stem}_per_seed.csv"
    summary_path = output_dir / f"{stem}_summary.csv"
    results_df.to_csv(per_seed_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nSaved:", file=sys.stderr)
    print(f"  Per-seed: {per_seed_path}", file=sys.stderr)
    print(f"  Summary : {summary_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True,
                                metavar="{build-features,train}")
    _add_build_features_parser(sub)
    _add_train_parser(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
