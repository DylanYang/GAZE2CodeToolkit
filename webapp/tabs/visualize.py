"""Tab 3 — render trial overlay / heatmap / duration / timeline."""
from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from g2c import visualization
from g2c.util import sample_offset

from ..components import (dataset_path_overrides, dataset_picker,
                           participant_picker, sample_size_slider, trial_picker)
from ..state import load_dataset_with_bar, save_uploaded_file

KIND_LABELS = {
    "trial":     "Trial overlay (fixations / saccades / AOIs on stimulus)",
    "heatmap":   "Heatmap (duration-weighted Gaussian density)",
    "duration":  "Line-level fixation duration bars",
    "timeline":  "Line-vs-time fixation timeline",
}


def render() -> None:
    st.header("Visualize gaze and AOIs")
    st.caption(
        "Render trial overlay, heatmap, fixation-duration bars and timeline "
        "as PNG. Equivalent to `python -m cli.visualize`."
    )

    source = st.radio("Source", ["Dataset trial", "AOI CSV upload"],
                       horizontal=True, key="viz_source")

    trial_data: pd.DataFrame | None = None
    samples_data: pd.DataFrame | None = None
    tag = "viz"

    if source == "Dataset trial":
        c1, c2 = st.columns(2)
        with c1:
            dataset = dataset_picker(key="viz_dataset")
            trial_id = trial_picker(dataset, key="viz_trial")
            sample_size = sample_size_slider(default=10, max_value=100,
                                             key="viz_n")
        with c2:
            st.write(" ")  # spacing
            load_btn = st.button("Load dataset", key="viz_load",
                                 type="primary")

        raw_dir_override, stimuli_dir_override, _, _ = dataset_path_overrides(
            dataset, key_prefix="viz"
        )

        if not load_btn and "viz_eye_events" not in st.session_state:
            st.info("Pick a dataset and trial, then click **Load dataset**.")
            return
        if load_btn:
            eye_events, samples = load_dataset_with_bar(
                dataset, sample_size,
                bar_text_prefix=(f"Parsing {dataset} for visualization "
                                  f"(sample_size={sample_size})"),
                raw_dir=raw_dir_override,
                stimuli_dir=stimuli_dir_override,
            )
            st.session_state["viz_eye_events"] = eye_events
            st.session_state["viz_samples"] = samples

        eye_events = st.session_state["viz_eye_events"]
        samples = st.session_state["viz_samples"]

        sel = participant_picker(eye_events, key="viz_participant")
        if sel is None:
            trial_data = eye_events[eye_events["trial_id"] == trial_id].copy()
            samples_data = samples[samples["trial_id"] == trial_id].copy()
            tag = f"all_{trial_id}"
        else:
            mask = ((eye_events["experiment_id"] == sel)
                    & (eye_events["trial_id"] == trial_id))
            smask = ((samples["experiment_id"] == sel)
                     & (samples["trial_id"] == trial_id))
            trial_data = eye_events.loc[mask].copy()
            samples_data = samples.loc[smask].copy()
            tag = f"{sel}_{trial_id}"

    else:  # AOI CSV upload
        uploaded = st.file_uploader("AOI-fixation CSV", type=["csv"],
                                    key="viz_aoi_upload")
        if uploaded is None:
            st.info("Upload a previously hit-tested fixation × AOI CSV to "
                    "continue.")
            return
        csv_path = save_uploaded_file(uploaded, suffix=".csv")
        trial_data = pd.read_csv(csv_path)
        samples_data = trial_data  # draw_trial expects same schema fallback
        tag = Path(uploaded.name).stem

    if trial_data is None or trial_data.empty:
        st.warning("No fixations to visualize.")
        return

    kinds = st.multiselect(
        "Visualizations to render",
        list(KIND_LABELS),
        default=["trial", "heatmap"],
        format_func=lambda k: KIND_LABELS[k],
        key="viz_kinds",
    )

    # Auto-detect which "gaze point" columns the samples DataFrame actually
    # contains. Tobii Pro Nano (UNL/YMU) exports use names like
    # "Gaze point X"; Tobii X3-120 (McChesney legacy) exports use
    # "Gaze point X [DACS px]". Offer a dropdown so the user can override.
    x_candidates, y_candidates = _detect_gaze_columns(
        samples_data if samples_data is not None else trial_data
    )

    with st.expander("Trial-overlay parameters"):
        c3, c4 = st.columns(2)
        with c3:
            r3 = st.slider("r3 (base circle radius)", 0.5, 10.0, 3.0, 0.5,
                           key="viz_r3")
            r5 = st.slider("r5 (per-100ms growth)", 0.05, 5.0, 0.8, 0.05,
                           key="viz_r5")
        with c4:
            draw_saccade = st.checkbox("draw saccades", False, key="viz_sac")
            draw_aoi = st.checkbox("draw AOIs", False, key="viz_aoi")
            draw_raw = st.checkbox(
                "draw raw samples", False, key="viz_raw",
                disabled=not x_candidates,
                help=("Disabled because no gaze-point columns were found "
                      "in the samples DataFrame.") if not x_candidates else None,
            )
        if x_candidates:
            c_x, c_y = st.columns(2)
            with c_x:
                sample_x_col = st.selectbox(
                    "Sample X column", x_candidates,
                    index=0, key="viz_sample_x",
                    help="Auto-detected from the samples DataFrame.",
                )
            with c_y:
                sample_y_col = st.selectbox(
                    "Sample Y column", y_candidates,
                    index=0, key="viz_sample_y",
                )
        else:
            sample_x_col = "Gaze point X"
            sample_y_col = "Gaze point Y"

    with st.expander("Coordinate offsets (calibration correction)"):
        st.caption(
            "Shift every fixation by a constant (x, y) offset before rendering. "
            "Useful when an eye-tracker calibration drift biases all fixations "
            "in one direction. All four visualizations now respect this offset "
            "exclusively — the previously silent internal heatmap shifts of "
            "(0, −40) and (+30, +80) have been removed."
        )
        c_ox, c_oy = st.columns(2)
        with c_ox:
            x_offset = st.slider("X offset (px)", -300, 300, 0, 1,
                                  key="viz_x_offset")
        with c_oy:
            y_offset = st.slider("Y offset (px)", -300, 300, 0, 1,
                                  key="viz_y_offset")

    with st.expander("Heatmap parameters"):
        c5, c6 = st.columns(2)
        with c5:
            sigma = st.slider("sigma", 5.0, 80.0, 17.0, 1.0, key="viz_sigma")
            alpha = st.slider("alpha", 0.1, 1.0, 0.6, 0.05, key="viz_alpha")
        with c6:
            vmin = st.number_input("vmin", value=0.0, step=10.0, key="viz_vmin")
            vmax = st.number_input("vmax", value=100.0, step=10.0,
                                   key="viz_vmax")
            contours = st.checkbox("contours", False, key="viz_contour")

    if not st.button("Render", type="primary", key="viz_render"):
        return

    if not kinds:
        st.warning("Select at least one visualization kind.")
        return

    # Apply the user-supplied calibration offset to fixations (x0, y0 / x1, y1).
    if x_offset != 0 or y_offset != 0:
        trial_data = sample_offset(trial_data, x_offset, y_offset)
        # Also shift the raw-sample columns when present, so the trial
        # overlay's raw-sample dots line up with the shifted fixations.
        if (samples_data is not None
                and sample_x_col in samples_data.columns
                and sample_y_col in samples_data.columns):
            samples_data = sample_offset(
                samples_data, x_offset, y_offset,
                x0_col=sample_x_col, y0_col=sample_y_col,
                x1_col="__unused_x1__", y1_col="__unused_y1__",
            )
        st.caption(
            f"Applied calibration offset Δx = {x_offset:+d} px, "
            f"Δy = {y_offset:+d} px before rendering."
        )

    # One progress bar that advances as each requested visualization
    # completes. Individual matplotlib renderings are atomic, so the bar
    # steps per kind rather than within each kind.
    n_kinds = len(kinds)
    render_bar = st.progress(0.0, text="Preparing rendering …")
    step_idx = 0

    def _step(label: str) -> None:
        nonlocal step_idx
        step_idx += 1
        render_bar.progress(
            step_idx / n_kinds,
            text=f"Rendering {label} ({step_idx}/{n_kinds}) …",
        )

    if "trial" in kinds:
        _step("trial overlay")
        st.subheader("Trial overlay")
        img = visualization.draw_trial(
            trial_data,
            samples_data if samples_data is not None else trial_data,
            draw_raw_data=draw_raw,
            draw_fixation=True,
            draw_saccade=draw_saccade,
            draw_aoi=draw_aoi,
            sample_x_col=sample_x_col,
            sample_y_col=sample_y_col,
            r3=r3,
            r5=r5,
        )
        if img is not None and hasattr(img, "save"):
            st.image(img, caption=f"Trial overlay — {tag}")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            st.download_button("Download PNG", buf.getvalue(),
                               file_name=f"trial_{tag}.png",
                               mime="image/png", key="viz_dl_trial")
        else:
            _render_current_figure(f"trial_{tag}.png", "viz_dl_trial")

    if "heatmap" in kinds:
        _step("heatmap")
        st.subheader("Heatmap")
        visualization.draw_heatmap(
            trial_data,
            contours=contours,
            figsize=(18, 10),
            alpha=alpha,
            sigma_value=sigma,
            vmin=vmin,
            vmax=vmax,
        )
        _render_current_figure(f"heatmap_{tag}.png", "viz_dl_heatmap")

    if "duration" in kinds:
        _step("fixation duration")
        st.subheader("Fixation duration (per line)")
        img = visualization.fixation_duration(trial_data)
        if img is not None and hasattr(img, "save"):
            st.image(img, caption=f"Fixation duration — {tag}")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            st.download_button("Download PNG", buf.getvalue(),
                               file_name=f"duration_{tag}.png",
                               mime="image/png", key="viz_dl_dur")
        else:
            _render_current_figure(f"duration_{tag}.png", "viz_dl_dur")

    if "timeline" in kinds:
        _step("timeline")
        st.subheader("Line-vs-time timeline")
        visualization.fixation_timeline(trial_data, figsize=(15, 10))
        _render_current_figure(f"timeline_{tag}.png", "viz_dl_tl")

    render_bar.progress(1.0, text=f"Done — rendered {n_kinds} visualization(s).")
    import time as _t
    _t.sleep(0.3)
    render_bar.empty()


def _render_current_figure(filename: str, dl_key: str) -> None:
    fig = plt.gcf()
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    st.download_button("Download PNG", buf.getvalue(),
                       file_name=filename, mime="image/png", key=dl_key)


# Patterns we recognise as raw-gaze-point columns. The order biases the
# auto-selected default: prefer the cyclopean "Gaze point X" used by Tobii
# Pro Nano (UNL/YMU) over the older Tobii X3-120 names or per-eye columns.
_X_PATTERNS = (
    "Gaze point X",            # Tobii Pro Nano (UNL/YMU)
    "Gaze point X [DACS px]",  # Tobii X3-120 (McChesney legacy)
    "Gaze point left X",
    "Gaze point right X",
)
_Y_PATTERNS = (
    "Gaze point Y",
    "Gaze point Y [DACS px]",
    "Gaze point left Y",
    "Gaze point right Y",
)


def _detect_gaze_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return ([candidate X column names], [candidate Y column names])
    actually present in `df`, ordered by preference. Empty lists if none."""
    if df is None or df.empty:
        return [], []
    cols = set(df.columns.astype(str))
    xs = [c for c in _X_PATTERNS if c in cols]
    ys = [c for c in _Y_PATTERNS if c in cols]
    return xs, ys
