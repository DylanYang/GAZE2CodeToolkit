"""Cached loaders and session-state helpers shared by the Streamlit tabs.

The dataset loader uses session-state caching (not `@st.cache_data`) so
that the underlying parser's `progress_callback` fires on the first
call (cache miss) but subsequent calls with the same arguments return
instantly without re-running the parser.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import streamlit as st


def _dataset_cache_key(dataset_name: str, sample_size: int,
                        raw_dir: Optional[str], stimuli_dir: Optional[str]) -> str:
    # Path overrides MUST be part of the cache key — otherwise switching
    # raw_dir would silently return the previously cached dataset.
    return (f"_dataset_cache::{dataset_name}::{sample_size}::"
            f"{raw_dir or ''}::{stimuli_dir or ''}")


def load_dataset(
    dataset_name: str,
    sample_size: int,
    progress_callback=None,
    raw_dir: Optional[str] = None,
    stimuli_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parse a dataset and cache the result in session state.

    On cache miss the underlying `parsers.load` is called with
    `progress_callback`, so a Streamlit progress bar advances as files
    are parsed. On cache hit the function returns immediately and the
    callback is not invoked.

    `raw_dir` and `stimuli_dir` override the static paths from
    `datasets_config.py`. Different overrides produce distinct cache
    entries so switching paths re-parses correctly.
    """
    cache_key = _dataset_cache_key(dataset_name, sample_size,
                                    raw_dir, stimuli_dir)
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    from g2c.parsers import load
    result = load(dataset_name, sample_size=sample_size,
                  progress_callback=progress_callback,
                  raw_dir=raw_dir, stimuli_dir=stimuli_dir)
    st.session_state[cache_key] = result
    return result


def load_dataset_with_bar(
    dataset_name: str,
    sample_size: int,
    *,
    bar_text_prefix: str = "Loading dataset",
    raw_dir: Optional[str] = None,
    stimuli_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper: render an `st.progress` bar driven by the
    parser callback, then auto-clear the bar.
    """
    cache_key = _dataset_cache_key(dataset_name, sample_size,
                                    raw_dir, stimuli_dir)
    if cache_key in st.session_state:
        # Cache hit — no progress bar needed.
        return st.session_state[cache_key]

    progress_bar = st.progress(0.0, text=f"{bar_text_prefix} …")

    def _on_progress(fraction: float, message: str) -> None:
        progress_bar.progress(min(max(fraction, 0.0), 1.0), text=message)

    result = load_dataset(dataset_name, sample_size,
                           progress_callback=_on_progress,
                           raw_dir=raw_dir, stimuli_dir=stimuli_dir)

    import time
    time.sleep(0.3)
    progress_bar.empty()
    return result


@st.cache_data(show_spinner=False)
def trial_ids_for_dataset(dataset_name: str) -> list[str]:
    """Trial identifiers come straight from the dataset config; no need to
    actually parse any TSVs to enumerate them.
    """
    from g2c.parsers.datasets_config import DATASETS

    cfg = DATASETS[dataset_name]
    stimuli = cfg["stimuli_names"]
    strategy = cfg["trial_id_strategy"]
    out: list[str] = []
    for i, ev in enumerate(stimuli):
        words = ev.split()
        if strategy == "first_word":
            out.append(words[0] if words else str(i + 1))
        elif strategy == "third_word":
            out.append(words[2] if len(words) >= 3 else str(i + 1))
        elif strategy == "trial_index":
            out.append(str(i + 1))
        else:
            out.append(str(i + 1))
    return out


def save_uploaded_file(uploaded, suffix: Optional[str] = None) -> Path:
    """Persist an `st.file_uploader` upload to a temp path and return it."""
    if suffix is None:
        suffix = Path(uploaded.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.close()
    return Path(tmp.name)


def ensure_output_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
