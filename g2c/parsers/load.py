"""Unified parser entry point.

Use this in new code:

    from g2c.parsers import load
    eye_events, samples = load("YMU_UM", sample_size=100)

Legacy `UNL_UM()` / `YMU_UM()` functions are still exported by
`g2c.parsers` for backward compatibility and dispatch through this same
module.
"""
from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from .datasets_config import DATASETS, TOBII_PRO_COLUMNS
from .tobii import load_tobii


def _is_tobii_config(cfg: dict) -> bool:
    """Identify Tobii-family configs by their ``columns`` schema or by
    an ``eye_tracker`` string that starts with "Tobii".

    Datasets onboarded through the Streamlit "Onboard Tobii" tab will
    always satisfy both checks; the hand-written UNL_UM / YMU_UM entries
    in :mod:`datasets_config` satisfy the first.
    """
    if cfg.get("columns") is TOBII_PRO_COLUMNS:
        return True
    return str(cfg.get("eye_tracker", "")).startswith("Tobii")


def load(dataset_name: str,
         sample_size: Optional[int] = None,
         progress_callback=None,
         raw_dir: Optional[str] = None,
         stimuli_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load a dataset by name.

    Parameters
    ----------
    dataset_name : str
        Identifier defined in `datasets_config.DATASETS`.
    sample_size : int, optional
        Cap on the number of participant files. Defaults to the dataset's
        `default_sample_size` in the config.
    progress_callback : callable, optional
        Forwarded to the underlying parser so UI wrappers can render a
        progress bar (see `tobii.load_tobii`).

    Returns
    -------
    (eye_events, samples) : tuple of pd.DataFrame
        Canonical schema; `eye_events` filtered to fixations.

    Raises
    ------
    KeyError
        If `dataset_name` has no registered config.
    NotImplementedError
        If the config exists but the loader cannot infer how to parse
        the dataset from it (no recognised eye-tracker family).
    """
    if dataset_name not in DATASETS:
        raise KeyError(
            f"Unknown dataset {dataset_name!r}. "
            f"Available: {sorted(DATASETS)}"
        )
    cfg = DATASETS[dataset_name]
    if _is_tobii_config(cfg):
        return load_tobii(dataset_name, sample_size=sample_size,
                           progress_callback=progress_callback,
                           raw_dir=raw_dir, stimuli_dir=stimuli_dir)
    raise NotImplementedError(
        f"Dataset {dataset_name!r} is registered but its config does not "
        f"match any known eye-tracker family (expected "
        f"`columns=TOBII_PRO_COLUMNS` or `eye_tracker` starting with "
        f"'Tobii'). Cannot dispatch to a parser."
    )


def available_datasets() -> list[str]:
    """Return the sorted list of dataset names known to the unified loader."""
    return sorted(DATASETS)
