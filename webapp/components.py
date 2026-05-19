"""Reusable Streamlit widgets shared across tabs."""
from __future__ import annotations

import io
from pathlib import Path

import streamlit as st


def dataset_picker(label: str = "Dataset", key: str | None = None) -> str:
    """Dropdown listing all datasets known to the unified parser."""
    from g2c.parsers import available_datasets
    options = available_datasets()
    if not options:
        st.error("No datasets are registered in `datasets_config.py`.")
        st.stop()
    return st.selectbox(label, options, key=key)


def sample_size_slider(
    *, default: int = 10, max_value: int = 100, key: str | None = None
) -> int:
    return st.slider(
        "Participants to parse (sample_size)",
        min_value=1,
        max_value=max_value,
        value=default,
        key=key,
    )


def trial_picker(dataset_name: str, key: str | None = None) -> str:
    from .state import trial_ids_for_dataset
    options = trial_ids_for_dataset(dataset_name)
    return st.selectbox("Trial ID", options, key=key)


def participant_picker(
    eye_events, label: str = "Participant", *, all_option: bool = True,
    key: str | None = None,
) -> str | None:
    """Returns the selected participant id, or None if 'All' was chosen."""
    parts = sorted(eye_events["experiment_id"].astype(str).unique())
    if all_option:
        parts = ["All participants"] + parts
    sel = st.selectbox(label, parts, key=key)
    if all_option and sel == "All participants":
        return None
    return sel


def metric_card(label: str, value, fmt: str = "{}") -> None:
    """Compact one-line stat — Streamlit's `st.metric` with formatting."""
    if value is None:
        value_str = "N/A"
    elif isinstance(value, float):
        value_str = fmt.format(value)
    else:
        value_str = str(value)
    st.metric(label, value_str)


def download_dataframe_button(
    df, *, label: str, filename: str, key: str | None = None,
) -> None:
    """Render a Streamlit button that downloads `df` as CSV."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button(
        label,
        data=buf.getvalue().encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def download_image_button(
    image_bytes: bytes, *, label: str, filename: str, mime: str = "image/png",
    key: str | None = None,
) -> None:
    st.download_button(label, data=image_bytes, file_name=filename, mime=mime, key=key)


def output_dir_input(default: str, key: str | None = None) -> Path:
    """Text input for an output directory; ensures it exists on return."""
    raw = st.text_input("Output directory", value=default, key=key)
    from .state import ensure_output_dir
    return ensure_output_dir(raw)


def dataset_path_overrides(
    dataset_name: str, key_prefix: str,
) -> tuple[str | None, str | None, str, str]:
    """Render the standard "Dataset paths" expander.

    Returns
    -------
    (raw_dir_override, stimuli_dir_override, effective_raw_dir, effective_stimuli_dir)
        The two overrides are `None` if the user left the field at its
        default (signalling the parser should fall back to the static
        config). The two `effective_*` values are the actually-active
        paths the caller should display in error messages.
    """
    from g2c.parsers.datasets_config import DATASETS

    cfg = DATASETS[dataset_name]
    default_raw = cfg["raw_dir"]
    default_stim = cfg["stimuli_dir"]

    with st.expander("Dataset paths (override `datasets_config.py`)",
                       expanded=False):
        st.caption(
            f"Defaults come from `datasets_config.DATASETS['{dataset_name}']`. "
            "Settings here are per-tab — if you have moved the dataset on disk, "
            "update this expander on each tab that needs it."
        )
        raw_input = st.text_input(
            "Raw TSV directory (raw_dir)",
            value=default_raw,
            key=f"{key_prefix}_raw_dir_{dataset_name}",
        )
        stim_input = st.text_input(
            "Stimuli image directory (stimuli_dir)",
            value=default_stim,
            key=f"{key_prefix}_stimuli_dir_{dataset_name}",
        )

    raw_override = (raw_input
                    if raw_input and raw_input != default_raw
                    else None)
    stim_override = (stim_input
                     if stim_input and stim_input != default_stim
                     else None)
    effective_raw = raw_override or default_raw
    effective_stim = stim_override or default_stim
    return raw_override, stim_override, effective_raw, effective_stim
