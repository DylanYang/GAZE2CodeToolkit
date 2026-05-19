"""Tab 4 — evaluate OCR token detection against a ground-truth CSV."""
from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from cli.evaluate_ocr import (OCREvaluator, REQUIRED_COLUMNS,
                                full_text_cer_wer, plot_roc)

from ..components import download_dataframe_button, metric_card


def render() -> None:
    st.header("Evaluate OCR detection")
    st.caption(
        "Compare an OCR detection CSV against ground truth using IoU "
        "matching + text similarity. Equivalent to "
        "`python -m cli.evaluate_ocr`."
    )

    c1, c2 = st.columns(2)
    with c1:
        gt_file = st.file_uploader("Ground-truth CSV", type=["csv"], key="ev_gt")
        st.caption(f"Required columns: {', '.join(REQUIRED_COLUMNS)}")
    with c2:
        det_file = st.file_uploader("Detected-tokens CSV", type=["csv"],
                                    key="ev_det")
        st.caption("Same columns; optional `confidence`.")

    c3, c4, c5 = st.columns(3)
    with c3:
        iou_threshold = st.slider("IoU threshold", 0.1, 0.95, 0.5, 0.05,
                                  key="ev_iou")
    with c4:
        text_sim = st.slider("Text similarity threshold", 0.1, 1.0, 0.8, 0.05,
                             key="ev_sim")
    with c5:
        skip_roc = st.checkbox("Skip ROC curve", value=False, key="ev_skip_roc")

    if not (gt_file and det_file):
        st.info("Upload both CSVs to enable evaluation.")
        return

    if not st.button("Evaluate", type="primary", key="ev_run"):
        return

    gt_df = pd.read_csv(gt_file)
    det_df = pd.read_csv(det_file)

    # Two upstream schemas need auto-translating into the evaluator's
    # `line_num, x, y, width, height, text` columns:
    #
    # (1) AOI Step 1 token-structure CSV (`cli.extract_aoi`):
    #     kind, name, trial_id, x, y, width, height, token, image
    #         → rename `name`→`line_num`, `token`→`text`
    #
    # (2) AOI Step 2 hit-test CSV (one row per fixation × AOI):
    #     ..., aoi_name, aoi_token, aoi_x, aoi_y, aoi_width, aoi_height, ...
    #         → rename aoi_*  → x/y/width/height/line_num/text
    #         → deduplicate to one row per unique AOI box.
    _SIMPLE_RENAMES = {"name": "line_num", "token": "text"}
    _AOI_HIT_RENAMES = {
        "aoi_name":   "line_num",
        "aoi_token":  "text",
        "aoi_x":      "x",
        "aoi_y":      "y",
        "aoi_width":  "width",
        "aoi_height": "height",
    }

    def _normalise(label: str, df: pd.DataFrame) -> pd.DataFrame:
        # Hit-test schema first (more specific — has both aoi_x/aoi_y).
        if {"aoi_x", "aoi_y", "aoi_name"}.issubset(df.columns):
            applicable = {o: n for o, n in _AOI_HIT_RENAMES.items()
                          if o in df.columns}
            df = df.rename(columns=applicable)
            before = len(df)
            df = df.drop_duplicates(
                subset=["line_num", "x", "y", "width", "height", "text"]
            ).reset_index(drop=True)
            st.caption(
                f"**{label}**: detected hit-test schema — auto-renamed "
                f"`aoi_*` columns and deduplicated "
                f"{before:,} → {len(df):,} unique AOI rows."
            )
            return df
        # Token-structure schema.
        applicable = {o: n for o, n in _SIMPLE_RENAMES.items()
                      if o in df.columns and n not in df.columns}
        if applicable:
            df = df.rename(columns=applicable)
            st.caption(
                f"**{label}**: auto-renamed "
                + ", ".join(f"`{o}`→`{n}`" for o, n in applicable.items())
            )
        return df

    gt_df = _normalise("ground_truth", gt_df)
    det_df = _normalise("detected", det_df)

    for df, name in [(gt_df, "ground_truth"), (det_df, "detected")]:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            st.error(
                f"Missing columns in {name}: {missing}.  "
                f"Got columns: {list(df.columns)}"
            )
            return

    st.write(f"GT rows: **{len(gt_df)}**  ·  Detected rows: **{len(det_df)}**")

    eval_bar = st.progress(0.0, text="Preparing OCR evaluation …")

    def _on_eval(fraction: float, message: str) -> None:
        eval_bar.progress(min(max(fraction, 0.0), 1.0), text=message)

    evaluator = OCREvaluator(iou_threshold=iou_threshold,
                             text_sim_threshold=text_sim)
    metrics, matches = evaluator.evaluate(gt_df, det_df,
                                           progress_callback=_on_eval)
    ft = full_text_cer_wer(gt_df, det_df)
    metrics["full_text"] = ft

    import time as _t
    _t.sleep(0.3)
    eval_bar.empty()

    st.subheader("Detection")
    m1, m2, m3, m4 = st.columns(4)
    with m1: metric_card("True positives", metrics["true_positives"])
    with m2: metric_card("False positives", metrics["false_positives"])
    with m3: metric_card("False negatives", metrics["false_negatives"])
    with m4: metric_card("Match rate", metrics["match_rate"], "{:.3f}")

    m5, m6, m7, m8 = st.columns(4)
    with m5: metric_card("Precision", metrics["precision"], "{:.3f}")
    with m6: metric_card("Recall",    metrics["recall"],    "{:.3f}")
    with m7: metric_card("F1 score",  metrics["f1_score"],  "{:.3f}")
    with m8: metric_card("Avg IoU",   metrics["average_iou"],"{:.3f}")

    st.subheader("Text quality")
    t1, t2, t3, t4 = st.columns(4)
    with t1: metric_card("Char accuracy", metrics["character_accuracy"], "{:.3f}")
    with t2: metric_card("CER (avg)",     metrics["character_error_rate"], "{:.3f}")
    with t3: metric_card("Word accuracy", metrics["word_accuracy"], "{:.3f}")
    with t4: metric_card("WER (avg)",     metrics["word_error_rate"], "{:.3f}")

    st.subheader("Full-text comparison")
    f1, f2, f3 = st.columns(3)
    with f1: metric_card("Full-text CER", ft["cer"], "{:.4f}")
    with f2: metric_card("Full-text WER", ft["wer"], "{:.4f}")
    with f3: metric_card("Substitutions / Insertions / Deletions",
                          f"{ft['substitutions']} / {ft['insertions']} / {ft['deletions']}")

    if not skip_roc:
        st.subheader("ROC curve")
        with st.spinner("Computing ROC …"):
            tmp_path = Path("/tmp/_gaze2code_streamlit_roc.png")
            roc_auc = plot_roc(matches, det_df, gt_df, tmp_path)
        if roc_auc is None:
            st.info("ROC undefined (all rows matched or no positives).")
        else:
            st.image(str(tmp_path), caption=f"AUC = {roc_auc:.3f}")

    st.subheader("Matched pairs")
    matches_df = pd.DataFrame(matches)
    if matches_df.empty:
        st.write("_No matches found._")
    else:
        st.dataframe(matches_df, use_container_width=True)
        download_dataframe_button(matches_df, label="Download matches CSV",
                                   filename="matches.csv", key="ev_dl_matches")

    # Single-row metrics export
    metrics_flat = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
    metrics_flat.update({f"full_text_{k}": v for k, v in ft.items()})
    metrics_df = pd.DataFrame([metrics_flat])
    download_dataframe_button(metrics_df, label="Download metrics CSV",
                               filename="metrics.csv", key="ev_dl_metrics")
