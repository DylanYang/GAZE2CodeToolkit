"""Tab 1 — Extract fixations from a dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from g2c import util

from ..components import (dataset_path_overrides, dataset_picker,
                           output_dir_input, sample_size_slider)
from ..state import load_dataset_with_bar


def render() -> None:
    st.header("Extract fixations")
    st.caption(
        "Parse a Tobii dataset with the unified parser and write fixation "
        "CSVs. Equivalent to `python -m cli.extract_fixations`."
    )

    c1, c2 = st.columns(2)
    with c1:
        dataset = dataset_picker(key="extract_dataset")
        sample_size = sample_size_slider(default=10, max_value=100,
                                         key="extract_n")
    with c2:
        mode = st.radio(
            "Export mode",
            options=["all", "by-task", "per-participant"],
            captions=[
                "One combined CSV.",
                "One CSV per trial.",
                "One CSV per (trial, participant).",
            ],
            key="extract_mode",
        )
        # Key includes `dataset` + `mode` so each combination keeps its own
        # text-input state — switching mode therefore shows the matching
        # default path instead of the value the user typed for the previous
        # mode. All three modes now live under one `extracted-fixations/`
        # parent so the dataset folder is grouped consistently with
        # `aoi/`. Path layout:
        #   all              → output/{dataset}/extracted-fixations/all/
        #   by-task          → output/{dataset}/extracted-fixations/by-task/
        #   per-participant  → output/{dataset}/extracted-fixations/per-participant/
        out_dir = output_dir_input(
            default=f"output/{dataset.lower()}/extracted-fixations/{mode}",
            key=f"extract_outdir_{dataset}_{mode}",
        )

    # Dataset paths — let the user point the parser at a different
    # location without editing datasets_config.py.
    raw_dir_override, stimuli_dir_override, default_raw_dir, _ = (
        dataset_path_overrides(dataset, key_prefix="extract")
    )

    with st.expander("Optional filters"):
        st.caption("Leave empty to include everything the parser returned.")
        c3, c4 = st.columns(2)
        with c3:
            trial_filter_raw = st.text_input(
                "Restrict to trial IDs (comma-separated)", key="extract_trials",
            )
        with c4:
            exp_filter_raw = st.text_input(
                "Restrict to experiment IDs (comma-separated)", key="extract_exps",
            )

    if not st.button("Run extraction", type="primary", key="extract_run"):
        return

    # --- Stage 1: parse the dataset (progress bar driven by parser callback) ---
    eye_events, samples = load_dataset_with_bar(
        dataset, sample_size,
        bar_text_prefix=f"Parsing {dataset} (sample_size={sample_size})",
        raw_dir=raw_dir_override,
        stimuli_dir=stimuli_dir_override,
    )

    if eye_events.empty:
        active_raw = raw_dir_override or default_raw_dir
        st.error(
            "Parser returned no eye events. "
            f"Check that TSV files exist under `{active_raw}` "
            "or override the path in the **Dataset paths** expander."
        )
        return

    trial_range = pd.Series(_split_csv(trial_filter_raw)
                            or samples["trial_id"].unique())
    exp_range = pd.Series(_split_csv(exp_filter_raw)
                          or eye_events["experiment_id"].unique())

    st.success(
        f"Parsed {len(eye_events):,} fixation rows over "
        f"{eye_events['experiment_id'].nunique()} participant(s) "
        f"× {eye_events['trial_id'].nunique()} trial(s)."
    )

    # --- Stage 2: export CSVs (progress bar driven by export_fixations callback) ---
    export_bar = st.progress(0.0, text=f"Preparing export ({mode}) …")

    def _on_export(fraction: float, message: str) -> None:
        export_bar.progress(min(max(fraction, 0.0), 1.0), text=message)

    if mode == "all":
        util.export_fixations(eye_events, samples, exp_range, trial_range,
                              str(out_dir), byall=True,
                              progress_callback=_on_export)
    elif mode == "by-task":
        util.export_fixations(eye_events, samples, exp_range, trial_range,
                              str(out_dir), bytask=True,
                              progress_callback=_on_export)
    else:
        util.export_fixations(eye_events, samples, exp_range, trial_range,
                              str(out_dir),
                              progress_callback=_on_export)

    import time as _t
    _t.sleep(0.3)
    export_bar.empty()

    n_files = sum(1 for _ in Path(out_dir).rglob("*.csv"))
    st.success(f"Wrote {n_files} CSV file(s) under `{out_dir}`.")

    st.subheader("eye_events preview")
    st.dataframe(eye_events.head(50), use_container_width=True)

    st.subheader("samples preview")
    st.dataframe(samples.head(20), use_container_width=True)


def _split_csv(raw: str) -> list[str]:
    if not raw or not raw.strip():
        return []
    return [s.strip() for s in raw.split(",") if s.strip()]
