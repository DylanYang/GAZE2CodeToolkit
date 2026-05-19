"""Unified Tobii TSV parser (UNL_UM, YMU_UM).

Replaces the two near-duplicate dataset-specific parsers with a single
config-driven implementation. Behaviour for the existing two datasets is
preserved exactly; the legacy `UNL_UM()` and `YMU_UM()` wrappers in this
package call into `load_tobii()` so existing notebooks keep working.

Adding a third Tobii dataset (same hardware, different stimuli markers)
is a configuration change only — extend `datasets_config.DATASETS`, no
code edit required.
"""
from __future__ import annotations

import os
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .datasets_config import DATASETS
from .eye_events import get_eye_event_columns


class TobiiParseError(Exception):
    """Raised when a TSV file cannot be split into the expected trials."""


# ---------------------------------------------------------------------------
# Trial-split strategies
# ---------------------------------------------------------------------------

def _trial_spans_paired_markers(
    raw: pd.DataFrame,
    event_value_col: str,
    stimuli_names: tuple,
    per_trial: int,
    n_stimuli: int,
) -> list[tuple[int, int]]:
    """Strategy used by UNL_UM.

    Each stimulus appears `per_trial` times in the marker stream (start /
    intermediate / end). The trial span runs from the first to the last
    marker for that stimulus.
    """
    idxs = raw.loc[raw[event_value_col].isin(stimuli_names)].index.tolist()
    expected = per_trial * n_stimuli
    if len(idxs) < expected:
        raise TobiiParseError(
            f"Expected at least {expected} marker rows, found {len(idxs)}"
        )
    spans: list[tuple[int, int]] = []
    for i in range(n_stimuli):
        start = idxs[i * per_trial]
        end = idxs[i * per_trial + (per_trial - 1)]
        spans.append((start, end))
    return spans


def _trial_spans_dedup_sequential(
    raw: pd.DataFrame,
    event_value_col: str,
    stimuli_names: tuple,
    n_stimuli: int,
) -> list[tuple[int, int]]:
    """Strategy used by YMU_UM.

    Each unique stimulus marker is kept once (first occurrence). The trial
    runs from that marker to the next stimulus marker (or end of file).
    """
    rows = raw[raw[event_value_col].isin(stimuli_names)].drop_duplicates(
        subset=[event_value_col], keep="first"
    )
    idxs = rows.index.tolist()
    if len(idxs) != n_stimuli:
        raise TobiiParseError(
            f"Expected {n_stimuli} unique stimuli, found {len(idxs)}"
        )
    spans: list[tuple[int, int]] = []
    for i in range(n_stimuli):
        start = idxs[i]
        end = idxs[i + 1] if i + 1 < n_stimuli else len(raw)
        spans.append((start, end))
    return spans


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------

def _split_words(s: str) -> list[str]:
    return s.split() if isinstance(s, str) else []


def _derive_trial_id(event_value: str, strategy: str, idx: int) -> str:
    if strategy == "first_word":
        words = _split_words(event_value)
        return words[0] if words else str(idx + 1)
    if strategy == "third_word":
        words = _split_words(event_value)
        return words[2] if len(words) >= 3 else str(idx + 1)
    if strategy == "trial_index":
        return str(idx + 1)
    raise ValueError(f"Unknown trial_id strategy: {strategy!r}")


def _render_stimuli_name(event_value: str, idx: int, template: str) -> str:
    words = _split_words(event_value)
    return template.format(
        event_value=event_value if isinstance(event_value, str) else "",
        idx_plus_one=idx + 1,
        first_word=words[0] if words else "",
        third_word=words[2] if len(words) >= 3 else "",
    )


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def load_tobii(
    dataset_name: str,
    sample_size: Optional[int] = None,
    progress_callback=None,
    raw_dir: Optional[str] = None,
    stimuli_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parse a Tobii-based dataset by config name.

    Parameters
    ----------
    dataset_name : str
        Key in `datasets_config.DATASETS`.
    sample_size : int, optional
        Cap on the number of participant TSV files to parse. If None, the
        dataset's `default_sample_size` is used.
    progress_callback : callable, optional
        If given, called as ``callback(fraction, message)`` once per file
        plus a final 100% completion message. Lets UI wrappers render a
        progress bar.
    raw_dir : str, optional
        Override for the dataset's raw-TSV directory. When None, the path
        baked into `datasets_config.DATASETS[name]["raw_dir"]` is used.
    stimuli_dir : str, optional
        Override for the dataset's stimulus-image directory. When None,
        the path from the config is used.

    Returns
    -------
    eye_events, samples : tuple of pd.DataFrame
        `eye_events` follows the canonical schema from `get_eye_event_columns()`
        and is filtered to fixations only. `samples` is the raw stream tagged
        with experiment/trial metadata and pruned of MCS-norm columns.
    """
    def _report(frac, msg):
        if progress_callback is not None:
            try:
                progress_callback(frac, msg)
            except Exception:
                pass
    if dataset_name not in DATASETS:
        raise KeyError(
            f"Unknown dataset {dataset_name!r}. Available: {sorted(DATASETS)}"
        )
    cfg = DATASETS[dataset_name]

    if sample_size is None:
        sample_size = cfg.get("default_sample_size")

    cm = cfg["columns"]
    ev_col = cm["event_value"]
    stimuli_names = tuple(cfg["stimuli_names"])
    n_stimuli = cfg["n_stimuli"]
    fix_label = cfg["fixation_label"]
    split_cfg = cfg["trial_split"]
    template = cfg["stimuli_name_template"]
    tid_strategy = cfg["trial_id_strategy"]
    participant_col = cfg.get("participant_col") or "Participant name"
    drop_from = cfg.get("drop_cols_from")

    # Honour caller-supplied path overrides (per-call), falling back to
    # the static config when not provided.
    effective_raw_dir = raw_dir if raw_dir else cfg["raw_dir"]
    effective_stimuli_dir = stimuli_dir if stimuli_dir else cfg["stimuli_dir"]

    _report(0.0, f"Discovering {dataset_name} TSV files in "
                  f"{effective_raw_dir} …")
    # Discover .tsv files (sorted, capped to sample_size).
    all_files: list[tuple[str, str]] = []
    for root, _, files in os.walk(effective_raw_dir):
        for f in sorted(files):
            if f.endswith(".tsv"):
                all_files.append((root, f))
    if sample_size is not None:
        all_files = all_files[:sample_size]
    n_total = len(all_files)

    eye_events = pd.DataFrame()
    samples = pd.DataFrame()

    bar = tqdm(all_files, desc=f"Parsing {dataset_name}", unit="file", colour="green")
    for i, (root, fname) in enumerate(bar):
        _report(
            (i + 1) / max(n_total, 1),
            f"Parsing {dataset_name}: {fname} ({i + 1}/{n_total}) …",
        )
        path = os.path.join(root, fname)
        try:
            raw = pd.read_csv(path, sep="\t")
            if drop_from is not None and drop_from < raw.shape[1]:
                raw.drop(raw.columns[drop_from:], axis=1, inplace=True)

            strategy = split_cfg["strategy"]
            if strategy == "paired_markers":
                spans = _trial_spans_paired_markers(
                    raw, ev_col, stimuli_names,
                    per_trial=split_cfg["per_trial"], n_stimuli=n_stimuli,
                )
            elif strategy == "dedup_sequential":
                spans = _trial_spans_dedup_sequential(
                    raw, ev_col, stimuli_names, n_stimuli=n_stimuli,
                )
            else:
                raise ValueError(f"Unknown trial-split strategy: {strategy!r}")

            for idx, (start, end) in enumerate(spans):
                trial_samples = raw.iloc[start:end].copy()

                if participant_col in raw.columns:
                    participant_id = raw[participant_col].iloc[0]
                else:
                    participant_id = fname.split(".")[0]
                experiment_id = participant_id

                ev_val = raw[ev_col].iloc[start]
                stimuli_name = _render_stimuli_name(ev_val, idx, template)
                trial_id = _derive_trial_id(ev_val, tid_strategy, idx)

                _tag_metadata(trial_samples, cfg, fname,
                              experiment_id, participant_id,
                              stimuli_name, trial_id,
                              stimuli_dir=effective_stimuli_dir)
                samples = pd.concat([samples, trial_samples])

                trial_ee = _build_eye_events(trial_samples, cm, fix_label)
                _tag_metadata(trial_ee, cfg, fname,
                              experiment_id, participant_id,
                              stimuli_name, trial_id,
                              stimuli_dir=effective_stimuli_dir)
                eye_events = pd.concat([eye_events, trial_ee])

        except Exception as exc:
            warnings.warn(f"[{dataset_name}] skip {fname}: {exc}", stacklevel=2)
            continue

    # Drop sample columns that downstream code does not consume.
    drop_cols = cfg.get("samples_drop_cols", [])
    if not samples.empty:
        samples.drop(columns=[c for c in drop_cols if c in samples.columns],
                     inplace=True)

    if not eye_events.empty:
        eye_events = eye_events[get_eye_event_columns()]

    # Preserve the original column-reorder trick used by both legacy parsers.
    if not samples.empty and "Eye movement type index" in samples.columns:
        samples = pd.concat(
            [samples.loc[:, "eye_tracker":],
             samples.loc[:, :"Eye movement type index"]],
            axis=1,
        )

    eye_events.reset_index(drop=True, inplace=True)
    samples.reset_index(drop=True, inplace=True)
    _report(
        1.0,
        f"Parsed {dataset_name}: {len(eye_events):,} fixation rows, "
        f"{eye_events['experiment_id'].nunique() if not eye_events.empty else 0} "
        f"participant(s).",
    )
    return eye_events, samples


def _tag_metadata(df: pd.DataFrame, cfg: dict, fname: str,
                  experiment_id: str, participant_id: str,
                  stimuli_name: str, trial_id: str,
                  stimuli_dir: Optional[str] = None) -> None:
    df["eye_tracker"] = cfg["eye_tracker"]
    df["experiment_id"] = experiment_id
    df["participant_id"] = participant_id
    df["filename"] = fname
    df["trial_id"] = str(trial_id)
    df["stimuli_module"] = stimuli_dir if stimuli_dir else cfg["stimuli_dir"]
    df["stimuli_name"] = stimuli_name


def _build_eye_events(trial_samples: pd.DataFrame, cm: dict,
                      fix_label: str) -> pd.DataFrame:
    """Project per-trial samples down to canonical eye-event columns."""
    base_cols = [cm["timestamp"], cm["duration"], cm["x"], cm["y"], cm["event_type"]]
    has_pupil = cm.get("pupil_l") and cm["pupil_l"] in trial_samples.columns
    if has_pupil:
        base_cols.extend([cm["pupil_l"], cm["pupil_r"]])
    ee = trial_samples[base_cols].copy()

    rename_map = {
        cm["timestamp"]: "timestamp",
        cm["duration"]: "duration",
        cm["x"]: "x0",
        cm["y"]: "y0",
        cm["event_type"]: "eye_event_type",
    }
    if has_pupil:
        rename_map[cm["pupil_l"]] = "pupil_l"
        rename_map[cm["pupil_r"]] = "pupil_r"
    ee.rename(columns=rename_map, inplace=True)

    ee["x1"] = np.nan
    ee["y1"] = np.nan
    ee["token"] = None
    if not has_pupil:
        ee["pupil_l"] = np.nan
        ee["pupil_r"] = np.nan
    ee["amplitude"] = np.nan
    ee["peak_velocity"] = np.nan

    ee = ee[ee["eye_event_type"] == fix_label].copy()
    ee["eye_event_type"] = "fixation"
    return ee
