"""Tab 5 — gaze-based expertise classification (PCGC / hybrid).

Wraps `cli.classify_expertise` (subcommands `build-features` and
`train`). Two sections in one tab: build the participant-level token
features, then run multi-seed participant-level CV.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

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

from ..components import download_dataframe_button, metric_card


def render() -> None:
    st.header("Classify expertise (PCGC / hybrid)")
    st.caption(
        "Stage II — turns the fixation × token AOI table into "
        "participant-level expertise predictions. Strict participant-level "
        "CV; prototypes and scalers are fit on the training fold only. "
        "Equivalent to `python -m cli.classify_expertise`."
    )

    _build_features_section()
    st.divider()
    _train_section()


# ---------------------------------------------------------------------------
# Build-features section
# ---------------------------------------------------------------------------

def _build_features_section() -> None:
    st.subheader("Step 1 — Build token features")
    st.caption(
        "Input: a directory of per-question CSVs (`Q1.csv … Q5.csv`), each "
        "containing the columns `p_id, expertise, duration, aoi_token`. "
        "Output: one `participant_features_<Q>_token.csv` per question with "
        "8 token-level statistics per participant."
    )

    c1, c2 = st.columns(2)
    with c1:
        raw_dir = st.text_input(
            "Labelled fixation × token directory",
            value="output/unl_um/aoi/labelled",
            key="cls_raw_dir",
            help="Each CSV must already contain p_id and expertise columns "
                 "(join cli.score_expertise output onto the AOI hit-test).",
        )
    with c2:
        out_dir = st.text_input(
            "Output directory",
            value="output/unl_um/classification",
            key="cls_out_dir",
        )

    questions = st.multiselect(
        "Questions to process",
        options=QUESTIONS,
        default=QUESTIONS,
        key="cls_bf_questions",
    )

    if not st.button("Build features", type="primary", key="cls_bf_run"):
        return

    raw = Path(raw_dir)
    out = Path(out_dir)
    if not raw.is_dir():
        st.error(f"Input directory not found: `{raw}`")
        return
    out.mkdir(parents=True, exist_ok=True)

    progress = st.progress(0.0, text="Extracting token features …")
    results: list[dict] = []
    skipped: list[str] = []
    for i, q in enumerate(questions):
        raw_csv = raw / f"{q}.csv"
        if not raw_csv.exists():
            skipped.append(q)
            progress.progress((i + 1) / len(questions),
                              text=f"Skipped {q} (no CSV)")
            continue
        try:
            feat_df = extract_token_features_from_csv(str(raw_csv))
        except ValueError as exc:
            st.error(f"{q}: {exc}")
            progress.empty()
            return
        out_path = out / f"participant_features_{q}_token.csv"
        feat_df.to_csv(out_path, index=False)
        results.append({
            "question": q,
            "participants": len(feat_df),
            "high (1)": int((feat_df["expertise"] == 1).sum()),
            "low (0)": int((feat_df["expertise"] == 0).sum()),
            "output": str(out_path),
        })
        progress.progress((i + 1) / len(questions),
                          text=f"Wrote {out_path.name}")

    progress.empty()

    if skipped:
        st.warning(f"Skipped questions (no CSV found): {', '.join(skipped)}")
    if not results:
        st.info("No CSVs were written.")
        return

    st.success(f"Wrote {len(results)} feature CSVs to `{out}`.")
    st.dataframe(pd.DataFrame(results), use_container_width=True)


# ---------------------------------------------------------------------------
# Train section
# ---------------------------------------------------------------------------

def _train_section() -> None:
    st.subheader("Step 2 — Train + evaluate (participant-level CV)")
    st.caption(
        "Multi-seed StratifiedKFold CV. Single-task = one per-question CSV. "
        "Multi-task = aggregate across questions with weighted-mean or "
        "OOF-AUC-weighted strategies."
    )

    mode = st.radio(
        "Mode",
        options=["Single-task", "Multi-task"],
        index=1,
        horizontal=True,
        key="cls_train_mode",
    )

    if mode == "Single-task":
        _single_task_section()
    else:
        _multi_task_section()


def _single_task_section() -> None:
    c1, c2 = st.columns([3, 1])
    with c1:
        csv_path = st.text_input(
            "Participant feature CSV",
            value="output/unl_um/classification/participant_features_Q1_token.csv",
            key="cls_st_input",
        )
    with c2:
        representation = st.selectbox(
            "Representation",
            options=["A", "A_v2", "A_v3", "B", "C", "D"],
            index=0,
            key="cls_st_repr",
        )

    model, agg, seeds, n_splits, reg, lr_c, output_dir = _shared_train_controls("st")

    if not st.button("Run single-task CV", type="primary", key="cls_st_run"):
        return

    path = Path(csv_path)
    if not path.exists():
        st.error(f"CSV not found: `{path}`")
        return

    try:
        df, feature_cols = load_participant_feature_csv(
            csv_path=str(path), pid_col="p_id", label_col="expertise",
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    n_high = int((df["expertise"] == 1).sum())
    n_low = int((df["expertise"] == 0).sum())
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Participants", len(df))
    with c2: metric_card("High (1)", n_high)
    with c3: metric_card("Low (0)", n_low)
    with c4: metric_card("Feature dims", len(feature_cols))

    with st.spinner(f"Running {len(seeds)} seeds × {n_splits} splits …"):
        results_df = run_multi_seed_experiment(
            df=df, feature_cols=feature_cols,
            representation=representation, model_name=model, agg=agg,
            seeds=seeds, n_splits=n_splits, reg=reg, lr_C=lr_c,
        )
        summary_df = summarize_results(results_df)

    stem = (f"repr_{representation}_model_{model}"
            f"_agg_{agg}_reg{reg}_lrc{lr_c}")
    _show_results(results_df, summary_df, stem, output_dir, key_prefix="st")


def _multi_task_section() -> None:
    c1, c2 = st.columns([3, 1])
    with c1:
        mt_dir = st.text_input(
            "Multi-task directory",
            value="output/unl_um/classification",
            key="cls_mt_dir",
            help="Directory of `participant_features_<Q>_<suffix>.csv` files.",
        )
    with c2:
        suffix = st.text_input("Suffix", value="token", key="cls_mt_suffix")

    questions = st.multiselect(
        "Questions",
        options=QUESTIONS,
        default=QUESTIONS,
        key="cls_mt_questions",
    )

    c3, c4 = st.columns(2)
    with c3:
        feature_type = st.selectbox(
            "Feature type",
            options=["repr_A", "direct", "score_stack",
                     "repr_A_concat", "score_concat"],
            index=0,
            key="cls_mt_ftype",
        )
    with c4:
        weighting = st.text_input(
            "Weighting",
            value="w1",
            key="cls_mt_weighting",
            help="uniform | w1 | w2_<alpha> | oof_relu | oof_soft_<alpha>",
        )

    inner_splits = st.number_input(
        "OOF inner splits (only used for `oof_*` weighting or `*_concat` features)",
        min_value=2, max_value=10, value=3, step=1,
        key="cls_mt_inner",
    )

    model, agg, seeds, n_splits, reg, lr_c, output_dir = _shared_train_controls("mt")

    if not st.button("Run multi-task CV", type="primary", key="cls_mt_run"):
        return

    mt = Path(mt_dir)
    if not mt.is_dir():
        st.error(f"Directory not found: `{mt}`")
        return
    if not questions:
        st.error("Select at least one question.")
        return

    q_dfs: dict[str, pd.DataFrame] = {}
    feat_cols: list[str] | None = None
    for q in questions:
        csv_path = mt / f"participant_features_{q}_{suffix}.csv"
        if not csv_path.exists():
            st.error(f"Missing CSV: `{csv_path}`")
            return
        try:
            df_q, fc = load_participant_feature_csv(str(csv_path))
        except ValueError as exc:
            st.error(f"{q}: {exc}")
            return
        q_dfs[q] = df_q
        if feat_cols is None:
            feat_cols = fc

    common = set.intersection(*[set(df["p_id"].unique()) for df in q_dfs.values()])
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Questions", len(q_dfs))
    with c2: metric_card("Common participants", len(common))
    with c3: metric_card("Feature dims (per Q)", len(feat_cols) if feat_cols else 0)

    is_oof = (
        weighting.startswith("oof_")
        or feature_type in ("score_stack", "repr_A_concat", "score_concat")
    )

    with st.spinner(
        f"Running {len(seeds)} seeds × {n_splits} splits "
        f"({'OOF' if is_oof else 'biased'} weighting) …"
    ):
        if is_oof:
            results_df = run_oof_weighted_multitask_experiment(
                q_dfs=q_dfs, feat_cols=feat_cols,
                feature_type=feature_type, weighting=weighting,
                model_name=model, agg=agg,
                seeds=seeds, n_splits=n_splits,
                reg=reg, lr_C=lr_c, inner_splits=int(inner_splits),
            )
        else:
            results_df = run_weighted_multitask_experiment(
                q_dfs=q_dfs, feat_cols=feat_cols,
                feature_type=feature_type, weighting=weighting,
                model_name=model, agg=agg,
                seeds=seeds, n_splits=n_splits,
                reg=reg, lr_C=lr_c,
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

    stem = (f"mt_{feature_type}_{weighting}"
            f"_model_{model}_agg_{agg}_reg{reg}_lrc{lr_c}")
    _show_results(results_df, summary_df, stem, output_dir, key_prefix="mt")


# ---------------------------------------------------------------------------
# Shared widgets
# ---------------------------------------------------------------------------

def _shared_train_controls(key_prefix: str):
    c1, c2, c3 = st.columns(3)
    with c1:
        model = st.selectbox(
            "Classifier",
            options=["lr", "linsvm", "xgb"],
            index=2,
            key=f"cls_{key_prefix}_model",
        )
    with c2:
        agg = st.selectbox(
            "Prototype aggregation",
            options=["mean", "median"],
            index=0,
            key=f"cls_{key_prefix}_agg",
        )
    with c3:
        seeds_raw = st.text_input(
            "Seeds (comma- or space-separated)",
            value="42 43 44 45 46 47 48 49 50",
            key=f"cls_{key_prefix}_seeds",
        )

    c4, c5, c6 = st.columns(3)
    with c4:
        n_splits = st.number_input(
            "K-fold splits", min_value=2, max_value=10, value=5, step=1,
            key=f"cls_{key_prefix}_splits",
        )
    with c5:
        reg = st.number_input(
            "Mahalanobis reg",
            min_value=1e-9, max_value=1.0, value=1e-6,
            format="%.1e",
            key=f"cls_{key_prefix}_reg",
        )
    with c6:
        lr_c = st.number_input(
            "LR C", min_value=1e-3, max_value=100.0, value=1.0,
            format="%.3f",
            key=f"cls_{key_prefix}_lrc",
        )

    output_dir = st.text_input(
        "Output directory (CSV reports)",
        value=f"output/classification/{key_prefix}_results",
        key=f"cls_{key_prefix}_outdir",
    )

    seeds = _parse_seeds(seeds_raw)
    return model, agg, seeds, int(n_splits), float(reg), float(lr_c), Path(output_dir)


def _parse_seeds(raw: str) -> list[int]:
    parts = [p for p in raw.replace(",", " ").split() if p]
    out: list[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            continue
    return out or [42]


def _show_results(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    stem: str,
    output_dir: Path,
    key_prefix: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_path = output_dir / f"{stem}_per_seed.csv"
    summary_path = output_dir / f"{stem}_summary.csv"
    results_df.to_csv(per_seed_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    st.subheader("Summary")
    if not summary_df.empty:
        row = summary_df.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("AUC (mean)", row["auc_mean"], "{:.3f}")
        with c2: metric_card("AUC (std)", row["auc_std"], "{:.3f}")
        with c3: metric_card("Macro-F1 (mean)", row["f1_mean"], "{:.3f}")
        with c4: metric_card("Accuracy (mean)", row["acc_mean"], "{:.3f}")
    st.dataframe(summary_df, use_container_width=True)

    with st.expander("Per-seed results", expanded=False):
        st.dataframe(results_df, use_container_width=True)

    st.caption(f"Wrote `{per_seed_path}` and `{summary_path}`.")

    c1, c2 = st.columns(2)
    with c1:
        download_dataframe_button(
            results_df,
            label="Download per-seed CSV",
            filename=f"{stem}_per_seed.csv",
            key=f"cls_{key_prefix}_dl_seed",
        )
    with c2:
        download_dataframe_button(
            summary_df,
            label="Download summary CSV",
            filename=f"{stem}_summary.csv",
            key=f"cls_{key_prefix}_dl_summary",
        )
