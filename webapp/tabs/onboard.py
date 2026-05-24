"""Tab 5 — onboard a Tobii dataset (merged TSV + stimuli) into the toolkit.

Two-step flow for adding a brand-new dataset:

  Step 1 — Split a single merged Tobii Pro Lab TSV export by
           "Participant name" into per-participant TSVs
           (one file per participant, matching the layout the rest
           of the toolkit expects under ``datasets/<name>/rawdata/``).

  Step 2 — Upload the stimulus PNGs that go with the dataset, saved
           under ``datasets/<name>/stimuli/``.

At the end, the tab prints a ready-to-paste ``datasets_config.DATASETS``
snippet so the user can register the new dataset name and make it
selectable in the other tabs.
"""
from __future__ import annotations

import importlib
import re
import shutil
from collections import Counter
from pathlib import Path

import streamlit as st


def _reload_datasets_config() -> None:
    """Reload ``g2c.parsers.datasets_config`` and the modules that bind
    ``DATASETS`` from it, so subsequent ``available_datasets()`` calls in
    the *same* Streamlit run see the edits we just wrote to disk.

    Without this, Python's import cache would keep the stale
    ``DATASETS`` dict in memory and the user would have to press *R* to
    rerun for the dropdowns in other tabs to refresh.

    We pull the modules from ``sys.modules`` rather than via attribute
    access because ``g2c/parsers/__init__.py`` does
    ``from .load import load``, which shadows the submodule attribute
    ``g2c.parsers.load`` with the function of the same name. ``sys.modules``
    is unaffected by that shadowing.
    """
    import sys
    try:
        for name in (
            "g2c.parsers.datasets_config",
            "g2c.parsers.load",
            "g2c.parsers",
        ):
            mod = sys.modules.get(name)
            if mod is not None:
                importlib.reload(mod)
        # Also drop any @st.cache_data results that may have read from the
        # old DATASETS — e.g. legacy code that wrapped trial_ids_for_dataset.
        try:
            st.cache_data.clear()
        except Exception:
            pass
    except Exception:
        # Reload is best-effort. If it fails for any reason, the user can
        # always fall back to pressing R / restarting Streamlit.
        pass

_DEFAULT_PARTICIPANT_COL = "Participant name"
_DEFAULT_EVENT_VALUE_COL = "Event value"

# Match a "Q-prefix" like Q1, Q2A, Q5b at the start of a string. Used to
# pair an uploaded stimulus PNG filename with the corresponding
# Event-value marker scanned out of the recording.
_Q_PREFIX_RE = re.compile(r"^Q\d+[A-Za-z]?", re.IGNORECASE)
_PID_PATTERN = re.compile(r"P\d+")


def render() -> None:
    st.header("Onboard a Tobii dataset")
    st.caption(
        "Turn a single merged Tobii Pro Lab TSV export plus a folder of "
        "stimulus PNGs into the per-participant layout the rest of the "
        "toolkit expects (`datasets/<name>/rawdata/`, `datasets/<name>/stimuli/`). "
        "Also lets you tear down a previously added dataset."
    )

    datasets_root = st.text_input(
        "Datasets root directory",
        value="datasets",
        key="ob_datasets_root",
        help="Path where new dataset folders are created and where "
             "existing folders are looked up for removal. Absolute or "
             "relative to the GAZE2CodeToolkit working directory. "
             "Default `datasets/` matches the other tabs and "
             "`datasets_config.py`.",
    )
    if not datasets_root.strip():
        st.error("Datasets root directory cannot be empty.")
        return
    root = Path(datasets_root)
    if not root.is_absolute():
        root = Path.cwd() / root

    # ----------------------------------------------------------------- #
    # Add flow (Steps 1–3) — requires a confirmed new dataset name      #
    # ----------------------------------------------------------------- #
    st.subheader("Add a new dataset")

    c_name, c_btn = st.columns([3, 1])
    with c_name:
        name = st.text_input(
            "New dataset name",
            value="",
            key="ob_name",
            help="Folder name under the datasets root. Allowed characters: "
                 "letters, digits, underscore.",
        )
    name_valid = bool(name) and bool(re.fullmatch(r"[A-Za-z0-9_]+", name))
    with c_btn:
        st.write(" ")  # vertical alignment with the text input
        clicked_confirm = st.button(
            "Confirm name",
            type="primary",
            disabled=not name_valid,
            key="ob_name_confirm",
            help="Lock in the dataset name and reveal Steps 1 – 3 below. "
                 "Changing the name afterwards requires confirming again.",
        )

    if name and not name_valid:
        st.error(
            "Dataset name may only contain letters, digits, and underscores."
        )

    # Track the confirmed name across reruns. A button is True only on the
    # rerun that fires it, so we persist the latest confirmation in
    # session_state and gate Steps 1 – 3 on (name == confirmed_name).
    if clicked_confirm and name_valid:
        st.session_state["ob_add_confirmed"] = name
    confirmed = st.session_state.get("ob_add_confirmed")

    add_active = (
        name_valid and confirmed == name
    )

    if add_active:
        out_base = root / name
        raw_dir = out_base / "rawdata"
        stim_dir = out_base / "stimuli"

        st.code(
            f"{out_base}/\n├── rawdata/   # per-participant TSVs\n"
            f"└── stimuli/   # PNGs",
            language="text",
        )

        st.divider()
        _step1_split_tsv(name=name, raw_dir=raw_dir)
        st.divider()
        _step2_upload_stimuli(stim_dir=stim_dir)
        st.divider()
        _step3_config_snippet(name=name, raw_dir=raw_dir, stim_dir=stim_dir)
    elif name_valid:
        st.info(
            f"Click **Confirm name** to begin Steps 1 – 3 for `{name}`."
        )
    else:
        st.info(
            "Fill in a new dataset name and click **Confirm name** to "
            "begin Steps 1 – 3. Removing an existing dataset (Step 4 below) "
            "does not require a name."
        )

    # ----------------------------------------------------------------- #
    # Remove flow (Step 4) — always available                            #
    # ----------------------------------------------------------------- #
    st.divider()
    _step4_remove_dataset(datasets_root=root)


# ---------------------------------------------------------------------------
# Step 1 — split TSV
# ---------------------------------------------------------------------------

def _step1_split_tsv(*, name: str, raw_dir: Path) -> None:
    st.subheader("Step 1 — Provide per-participant TSVs")
    st.caption(
        "Populate `datasets/<name>/rawdata/` with one TSV per participant. "
        "Either split a single merged Tobii Pro Lab export, or copy a "
        "directory that is already split. In both cases the Event-value "
        "markers in the recording are also scanned so Step 2 can "
        "auto-rename your uploaded stimuli."
    )

    input_mode = st.radio(
        "Input format",
        [
            "Single merged TSV (to be split)",
            "Per-participant TSVs (already split)",
        ],
        horizontal=True,
        key="ob_input_mode",
        help="`Merged` runs a streaming split by Participant name. "
             "`Already split` copies each .tsv from a directory you "
             "provide and tallies markers along the way.",
    )

    if input_mode.startswith("Per-participant"):
        _step1_copy_split_dir(name=name, raw_dir=raw_dir)
        return

    source = st.radio(
        "TSV source",
        ["Server path (recommended for large files)", "Upload"],
        horizontal=True,
        key="ob_tsv_source",
    )

    tsv_path: Path | None = None

    if source.startswith("Server"):
        tsv_path_str = st.text_input(
            "Path to merged Tobii TSV",
            value="../_archive/UNL02/UNL02.tsv",
            key="ob_tsv_path",
            help="Absolute path, or relative to the GAZE2CodeToolkit "
                 "working directory. The archived UNL02 lives one level "
                 "above the toolkit at `autoresearch/_archive/`, so the "
                 "default is `../_archive/UNL02/UNL02.tsv` from cwd.",
        )
        if tsv_path_str:
            tsv_path = Path(tsv_path_str)
            if not tsv_path.is_absolute():
                tsv_path = Path.cwd() / tsv_path
    else:
        st.caption(
            "Server-side upload cap is set to **50 GB** by "
            "`.streamlit/config.toml`. Two progress phases:\n\n"
            "1. **Browser → server** — shown by Streamlit's own bar "
            "inside the upload widget below (the widget cancels its own "
            "byte transfer if you remove the file before it's done).\n"
            "2. **Server → disk** — shown as a bar that appears here "
            "*after* phase 1, spooling the in-memory upload to "
            "`/tmp/` in 64 MB chunks.\n\n"
            "Browser tabs typically OOM around 2–4 GB before phase 1 "
            "finishes, so for >2 GB files the **Server path** option "
            "above is still the safer route."
        )
        up = st.file_uploader("Upload .tsv", type=["tsv"], key="ob_tsv_upload")
        spool_slot = st.empty()    # reserved spot, populated as soon as upload completes
        if up is not None:
            tmp = Path(f"/tmp/_ob_upload_{name}.tsv")
            with spool_slot.container():
                tsv_path = _buffer_upload_to_disk(up, tmp)

    pid_col = st.text_input(
        "Participant column name",
        value=_DEFAULT_PARTICIPANT_COL,
        key="ob_pid_col",
        help="Tobii Pro Lab exports use 'Participant name' by default.",
    )

    overwrite = st.checkbox(
        "Overwrite existing rawdata/ folder if present",
        value=False, key="ob_overwrite_raw",
    )

    if not st.button("Split", type="primary", key="ob_split_run"):
        return

    if tsv_path is None:
        st.error("Provide a TSV source first.")
        return
    if not tsv_path.exists():
        st.error(f"TSV not found: `{tsv_path}`")
        return

    if raw_dir.exists() and any(raw_dir.iterdir()):
        if not overwrite:
            st.error(
                f"`{raw_dir}` is non-empty. Tick **Overwrite** to proceed, "
                "or pick a different dataset name."
            )
            return
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = tsv_path.stat().st_size
    progress = st.progress(0.0, text=f"Reading {tsv_path.name} ({total_bytes / 1e9:.2f} GB) …")

    try:
        results, ev_counts = _split_tsv_streaming(
            tsv_path=tsv_path,
            out_dir=raw_dir,
            prefix=name,
            participant_col=pid_col,
            progress_cb=lambda f, msg: progress.progress(
                min(max(f, 0.0), 1.0), text=msg
            ),
        )
    except _SplitError as exc:
        progress.empty()
        st.error(str(exc))
        return

    progress.empty()
    st.success(
        f"Wrote **{len(results)}** per-participant TSVs to `{raw_dir}`."
    )

    # Stash detected markers for Step 2's auto-rename feature.
    st.session_state["ob_detected_markers"] = dict(ev_counts)
    st.session_state["ob_n_participants"] = len(results)

    import pandas as pd
    summary = pd.DataFrame(
        [(pid, n_rows, str(path.relative_to(Path.cwd())))
         for pid, n_rows, path in results],
        columns=["participant", "rows", "output_path"],
    )
    st.dataframe(summary, use_container_width=True)

    n_markers = sum(1 for v in ev_counts.values() if v >= len(results))
    st.caption(
        f"Scanned **{sum(ev_counts.values()):,}** marker rows in the "
        f"Event-value column. {n_markers} marker(s) appear at least once "
        f"per participant — these are candidate stimulus markers for "
        f"Step 2's auto-rename."
    )


class _SplitError(Exception):
    """Raised when the merged TSV cannot be parsed."""


# ---------------------------------------------------------------------------
# Upload helper
# ---------------------------------------------------------------------------

# 64 MB matches typical SSD-friendly sequential write block sizes and keeps
# the progress-bar update rate around once every few hundred ms even for
# multi-GB files.
_UPLOAD_CHUNK_BYTES = 64 * 1024 * 1024


def _buffer_upload_to_disk(uploaded_file, target: Path) -> Path:
    """Stream a Streamlit ``UploadedFile`` to disk in 64 MB chunks while
    showing a progress bar. Streamlit's own progress UI covers the
    browser → server transfer; this bar covers the subsequent server-side
    spool to a real path on disk, which can take noticeable time for
    multi-GB files because ``uploaded_file.getbuffer()`` would otherwise
    materialise the entire payload in one shot with no UI feedback.
    """
    total = getattr(uploaded_file, "size", None) or 0
    bar = st.progress(
        0.0,
        text=(
            f"Spooling upload to `{target}` "
            f"({total / 1e9:.2f} GB) …" if total else
            f"Spooling upload to `{target}` …"
        ),
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    with open(target, "wb") as fout:
        while True:
            chunk = uploaded_file.read(_UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            fout.write(chunk)
            written += len(chunk)
            if total:
                bar.progress(
                    min(written / total, 1.0),
                    text=(f"Spooling … {written / 1e9:.2f} / "
                          f"{total / 1e9:.2f} GB"),
                )
            else:
                # Unknown total — keep the bar at indeterminate.
                bar.progress(
                    0.0,
                    text=f"Spooling … {written / 1e9:.2f} GB written",
                )
    bar.progress(1.0, text=f"Spooled {written / 1e9:.2f} GB → `{target}`")
    return target


# ---------------------------------------------------------------------------
# Step 1 — copy an already-split directory
# ---------------------------------------------------------------------------

def _step1_copy_split_dir(*, name: str, raw_dir: Path) -> None:
    """Branch of Step 1 that bypasses the split step.

    The user supplies a directory whose ``*.tsv`` files are already
    one-per-participant (e.g. the existing ``UNL_UM24_30July/rawdata/``
    layout). The branch copies each file into ``datasets/<name>/rawdata/``
    while streaming through it once to tally the Event-value markers
    — Step 2 needs those for auto-renaming uploaded stimuli.
    """
    src_dir_str = st.text_input(
        "Directory containing per-participant .tsv files",
        value="datasets/UNL_UM24_30July/rawdata",
        key="ob_split_src_dir",
        help="Absolute path, or relative to the GAZE2CodeToolkit working "
             "directory. All `.tsv` files in the top level of this "
             "directory are copied into `datasets/<name>/rawdata/` "
             "verbatim (filenames preserved).",
    )

    pid_col = st.text_input(
        "Participant column name",
        value=_DEFAULT_PARTICIPANT_COL,
        key="ob_pid_col_split",
        help="Used only to report participant IDs in the summary table. "
             "The parser reads this column from inside each TSV, so the "
             "physical filename does not have to encode the participant.",
    )

    overwrite = st.checkbox(
        "Overwrite existing rawdata/ folder if present",
        value=False, key="ob_overwrite_raw_split",
    )

    if not st.button("Copy", type="primary", key="ob_copy_run"):
        return

    src_dir = Path(src_dir_str)
    if not src_dir.is_absolute():
        src_dir = Path.cwd() / src_dir
    if not src_dir.is_dir():
        st.error(f"Source directory not found: `{src_dir}`")
        return

    files = sorted(src_dir.glob("*.tsv"))
    if not files:
        st.error(f"No `.tsv` files found in `{src_dir}`.")
        return

    if raw_dir.exists() and any(raw_dir.iterdir()):
        if not overwrite:
            st.error(
                f"`{raw_dir}` is non-empty. Tick **Overwrite** to proceed, "
                "or pick a different dataset name."
            )
            return
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    progress = st.progress(0.0, text=f"Copying {len(files)} TSV(s) …")
    try:
        results, ev_counts = _copy_split_files(
            files=files,
            out_dir=raw_dir,
            participant_col=pid_col,
            progress_cb=lambda f, msg: progress.progress(
                min(max(f, 0.0), 1.0), text=msg
            ),
        )
    except _SplitError as exc:
        progress.empty()
        st.error(str(exc))
        return
    progress.empty()

    st.success(
        f"Copied **{len(results)}** per-participant TSV(s) into `{raw_dir}`."
    )

    st.session_state["ob_detected_markers"] = dict(ev_counts)
    st.session_state["ob_n_participants"] = len(results)

    import pandas as pd
    summary = pd.DataFrame(
        [(pid, n_rows, str(path.relative_to(Path.cwd())))
         for pid, n_rows, path in results],
        columns=["participant", "rows", "output_path"],
    )
    st.dataframe(summary, use_container_width=True)

    n_markers = sum(1 for v in ev_counts.values() if v >= len(results))
    st.caption(
        f"Scanned **{sum(ev_counts.values()):,}** marker rows across all "
        f"files. {n_markers} marker(s) appear at least once per "
        "participant — candidate stimulus markers for Step 2's "
        "auto-rename."
    )


def _copy_split_files(
    *, files: list[Path],
    out_dir: Path,
    participant_col: str,
    progress_cb,
) -> tuple[list[tuple[str, int, Path]], Counter]:
    """Copy each per-participant TSV into ``out_dir`` (preserving the
    original filename) and tally Event-value markers in a single
    streaming pass.

    Returns ``(results, ev_counts)`` with the same shape as
    :func:`_split_tsv_streaming` so the caller can treat both Step 1
    branches uniformly.
    """
    ev_counts: Counter = Counter()
    results: list[tuple[str, int, Path]] = []
    n_files = len(files)

    for i, src in enumerate(files):
        out_path = out_dir / src.name
        with open(src, "rb") as fin, open(out_path, "wb") as fout:
            header_bytes = fin.readline()
            if not header_bytes:
                raise _SplitError(f"TSV is empty: {src}")
            fout.write(header_bytes)

            header = (header_bytes.decode("utf-8", errors="replace")
                      .rstrip("\r\n").split("\t"))
            pid_idx = (header.index(participant_col)
                       if participant_col in header else None)
            ev_idx = (header.index(_DEFAULT_EVENT_VALUE_COL)
                      if _DEFAULT_EVENT_VALUE_COL in header else None)

            row_count = 0
            pid_value: str | None = None
            for line_bytes in fin:
                fout.write(line_bytes)
                stripped = line_bytes.rstrip(b"\r\n")
                if not stripped:
                    continue
                row_count += 1
                parts = stripped.split(b"\t")
                if ev_idx is not None and ev_idx < len(parts) and parts[ev_idx]:
                    ev_counts[parts[ev_idx].decode("utf-8", errors="replace")] += 1
                if pid_value is None and pid_idx is not None \
                        and pid_idx < len(parts) and parts[pid_idx]:
                    pid_value = parts[pid_idx].decode("utf-8", errors="replace")

        if pid_value:
            m = _PID_PATTERN.search(pid_value)
            pid_label = m.group(0) if m else pid_value.replace(" ", "_")
        else:
            pid_label = src.stem

        results.append((pid_label, row_count, out_path))
        progress_cb(
            (i + 1) / n_files,
            f"Copied {src.name}  ({i + 1}/{n_files})",
        )

    return results, ev_counts


def _split_tsv_streaming(
    *, tsv_path: Path,
    out_dir: Path,
    prefix: str,
    participant_col: str,
    progress_cb,
) -> tuple[list[tuple[str, int, Path]], Counter]:
    """Stream the merged TSV row-by-row, append each row to its participant's
    output file. Returns a tuple of:

    * ``[(participant_id, row_count, output_path), …]`` — one entry per
      participant detected;
    * ``Counter[event_value -> count]`` — counts of non-empty values in
      the ``Event value`` column over the whole file. Used by Step 2 to
      auto-rename uploaded stimuli to match the actual markers in the
      recording.

    Implementation note: we read the file in **binary** mode and split each
    line manually on ``\\t``. This avoids the Python text-mode restriction
    that disables ``tell()`` while iterating, lets us report byte-accurate
    progress, and preserves the exact input bytes in each output file
    (no encoding normalisation). Tobii Pro Lab exports do not quote fields,
    so naïve ``split(b"\\t")`` is safe.
    """
    total_bytes = tsv_path.stat().st_size

    with open(tsv_path, "rb") as fin:
        header_bytes = fin.readline()
        if not header_bytes:
            raise _SplitError("TSV is empty.")
        header = (header_bytes.decode("utf-8", errors="replace")
                  .rstrip("\r\n").split("\t"))

        if participant_col not in header:
            preview = ", ".join(header[:10]) + (" …" if len(header) > 10 else "")
            raise _SplitError(
                f"Column `{participant_col}` not found in TSV header.\n"
                f"First columns: {preview}"
            )
        pid_idx = header.index(participant_col)
        ev_idx = (header.index(_DEFAULT_EVENT_VALUE_COL)
                  if _DEFAULT_EVENT_VALUE_COL in header else None)
        ev_counts: Counter = Counter()

        open_files: dict[str, list] = {}
        try:
            row_count = 0
            for line_bytes in fin:
                row_count += 1
                stripped = line_bytes.rstrip(b"\r\n")
                if not stripped:
                    continue
                # Only decode the participant column, not the whole row —
                # we write the original bytes verbatim downstream.
                parts = stripped.split(b"\t")
                if pid_idx >= len(parts) or not parts[pid_idx]:
                    continue
                pid_full = parts[pid_idx].decode("utf-8", errors="replace")
                m = _PID_PATTERN.search(pid_full)
                pid_label = m.group(0) if m else pid_full.replace(" ", "_")

                entry = open_files.get(pid_label)
                if entry is None:
                    out_path = out_dir / f"{prefix}_{pid_label}.tsv"
                    fh = open(out_path, "wb")
                    fh.write(header_bytes)
                    entry = [fh, 0, out_path]
                    open_files[pid_label] = entry

                entry[0].write(line_bytes)
                entry[1] += 1

                # Tally Event-value markers — used by Step 2 for auto-rename.
                if ev_idx is not None and ev_idx < len(parts) and parts[ev_idx]:
                    ev_counts[parts[ev_idx].decode("utf-8", errors="replace")] += 1

                if row_count % 50000 == 0:
                    pos = fin.tell()
                    frac = pos / total_bytes if total_bytes else 1.0
                    progress_cb(
                        frac,
                        f"Splitting … {pos / 1e9:.2f} / {total_bytes / 1e9:.2f} GB "
                        f"· {len(open_files)} participants so far",
                    )
        finally:
            for entry in open_files.values():
                entry[0].close()

    progress_cb(1.0, "Done.")

    results = [
        (pid, entry[1], entry[2])
        for pid, entry in sorted(open_files.items())
    ]
    return results, ev_counts


# ---------------------------------------------------------------------------
# Step 2 — upload stimuli
# ---------------------------------------------------------------------------

def _step2_upload_stimuli(*, stim_dir: Path) -> None:
    st.subheader("Step 2 — Upload stimulus PNGs")
    st.caption(
        "Drag in every stimulus image used in the recording (one PNG per "
        "trial). Markers scanned from the recording's `Event value` column "
        "are matched against your uploaded filenames by **Q-prefix** "
        "(e.g., `Q1-SpecifyOutput.png` → marker `Q1 (localhost)` → saved "
        "as `Q1 (localhost).png`). This makes the saved filename match "
        "what `stimuli_name_template: \"{event_value}.png\"` expects, so "
        "the Tobii parser can find the stimulus images automatically."
    )

    uploaded = st.file_uploader(
        "Stimulus PNGs",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="ob_stim_upload",
    )
    if not uploaded:
        st.info("Drop the stimuli files when ready.")
        return

    # Resolve auto-rename mapping. Prefer the in-session marker cache from
    # Step 1; if missing (page refreshed, or Step 2 used in isolation),
    # fall back to scanning one already-split TSV from rawdata/ to
    # rebuild the marker counts on the fly.
    detected_markers, n_participants, markers_source = _resolve_step2_markers(
        stim_dir.parent / "rawdata"
    )
    st.caption(f"Markers source: **{markers_source}** "
               f"({len(detected_markers)} unique markers, "
               f"{n_participants} participant(s))")

    # Pass every detected marker through to the Q-prefix matcher.
    # Threshold-based filtering used to live here, but it broke when the
    # markers came from a single-file scan (per-participant counts), and
    # `_find_marker_for_filename` already filters by Q-prefix + tie-breaks
    # by shortest marker, which is precise enough on its own.
    candidate_markers = list(detected_markers)

    auto_rename = st.checkbox(
        "Auto-rename uploaded files to match detected markers",
        value=bool(candidate_markers),
        disabled=not candidate_markers,
        key="ob_stim_autorename",
        help="Requires Step 1 to have finished in this session. "
             "When off, files keep their uploaded names.",
    )

    # Build the preview / edit table.
    import pandas as pd
    rows = []
    for f in uploaded:
        suggested = (
            _find_marker_for_filename(f.name, candidate_markers)
            if auto_rename else None
        )
        target_stem = suggested if suggested else Path(f.name).stem
        rows.append({
            "uploaded": f.name,
            "size_kb": round(len(f.getbuffer()) / 1024, 1),
            "matched_marker": suggested or "—",
            "save_as": target_stem,
        })
    preview_df = pd.DataFrame(rows)

    edited_df = st.data_editor(
        preview_df,
        hide_index=True,
        use_container_width=True,
        key="ob_stim_preview",
        disabled=["uploaded", "size_kb", "matched_marker"],
        column_config={
            "uploaded":       st.column_config.TextColumn("Uploaded"),
            "size_kb":        st.column_config.NumberColumn("Size (KB)",
                                                              format="%.1f"),
            "matched_marker": st.column_config.TextColumn(
                "Matched marker",
                help="Marker auto-detected by Q-prefix match against the "
                     "Event-value column scanned in Step 1.",
            ),
            "save_as":        st.column_config.TextColumn(
                "Save as (editable)",
                help="Final filename stem on disk. Override here if the "
                     "auto-match is wrong. The `.png` / `.jpg` extension "
                     "is added automatically based on the upload type.",
            ),
        },
    )

    if candidate_markers:
        with st.expander(
            f"Inspect all {len(candidate_markers)} detected marker(s)",
            expanded=False,
        ):
            st.caption(
                "All non-empty Event-value strings observed in the "
                "recording. Stimulus markers, keyboard events, and "
                "Pro Lab screen markers are all listed; the auto-rename "
                "matcher only fires on the ones whose Q-prefix matches "
                "an uploaded filename."
            )
            mk_df = pd.DataFrame(
                sorted(
                    [(m, detected_markers[m]) for m in candidate_markers],
                    key=lambda x: -x[1],
                ),
                columns=["marker", "count"],
            )
            st.dataframe(mk_df, use_container_width=True, hide_index=True)

    if not st.button("Save stimuli", type="primary", key="ob_stim_save"):
        return

    stim_dir.mkdir(parents=True, exist_ok=True)
    saved_rows = []
    seen_names: set[str] = set()
    for f, save_stem in zip(uploaded, edited_df["save_as"].tolist()):
        save_stem = str(save_stem).strip() or Path(f.name).stem
        ext = Path(f.name).suffix.lower() or ".png"
        target_name = f"{save_stem}{ext}"

        # Avoid silent overwrites within the same upload batch.
        unique_name = target_name
        i = 1
        while unique_name in seen_names:
            unique_name = f"{save_stem}__{i}{ext}"
            i += 1
        seen_names.add(unique_name)
        out_path = stim_dir / unique_name

        with open(out_path, "wb") as h:
            h.write(f.getbuffer())
        saved_rows.append({
            "uploaded": f.name,
            "saved_as": unique_name,
            "renamed":  "yes" if unique_name != f.name else "no",
            "size_kb":  round(out_path.stat().st_size / 1024, 1),
            "path":     str(out_path.relative_to(Path.cwd())),
        })

    st.success(f"Saved **{len(saved_rows)}** stimulus file(s) to `{stim_dir}`.")
    st.dataframe(pd.DataFrame(saved_rows), use_container_width=True)


def _resolve_step2_markers(raw_dir: Path) -> tuple[dict, int, str]:
    """Get the (markers, n_participants, source-label) tuple for Step 2.

    Order of preference:
    1. ``st.session_state`` cache populated by Step 1 in this session.
    2. Scan one TSV from ``raw_dir`` — handy when the page was refreshed
       or when Step 2 is used in isolation against a rawdata folder that
       already exists.
    3. Empty fallback — auto-rename disables, user keeps original names.
    """
    cached = st.session_state.get("ob_detected_markers")
    if cached:
        return (
            dict(cached),
            st.session_state.get("ob_n_participants", 0),
            "Step 1 cache",
        )

    if raw_dir.is_dir():
        tsvs = sorted(raw_dir.glob("*.tsv"))
        if tsvs:
            counts = _scan_markers_one_file(tsvs[0])
            if counts:
                return (
                    dict(counts),
                    len(tsvs),
                    f"scanned `{tsvs[0].relative_to(Path.cwd()) if Path.cwd() in tsvs[0].parents else tsvs[0]}`",
                )

    return ({}, 0, "none — auto-rename disabled")


def _detect_columns_override(raw_dir: Path) -> dict[str, str]:
    """Detect any divergence between the actual TSV header in ``raw_dir``
    and the default ``TOBII_PRO_COLUMNS`` mapping.

    Newer Tobii Pro Lab exports rename ``Gaze event duration`` to
    ``Eye movement event duration`` (one example). Any future renames
    should be added here. Returns the *override* dict (only the keys
    that differ), or an empty dict if no overrides are needed.
    """
    if not raw_dir.is_dir():
        return {}
    tsvs = sorted(raw_dir.glob("*.tsv"))
    if not tsvs:
        return {}
    try:
        with open(tsvs[0], "rb") as f:
            header_bytes = f.readline()
    except Exception:
        return {}
    header = (header_bytes.decode("utf-8", errors="replace")
              .rstrip("\r\n").split("\t"))
    header_set = set(header)

    overrides: dict[str, str] = {}
    if ("Gaze event duration" not in header_set
            and "Eye movement event duration" in header_set):
        overrides["duration"] = "Eye movement event duration"
    return overrides


def _scan_markers_one_file(tsv_path: Path) -> Counter:
    """Count Event-value column entries in one TSV file. Returns an empty
    Counter on any I/O / header error."""
    counts: Counter = Counter()
    try:
        with open(tsv_path, "rb") as fin:
            header_bytes = fin.readline()
            if not header_bytes:
                return counts
            header = (header_bytes.decode("utf-8", errors="replace")
                      .rstrip("\r\n").split("\t"))
            if _DEFAULT_EVENT_VALUE_COL not in header:
                return counts
            ev_idx = header.index(_DEFAULT_EVENT_VALUE_COL)
            for line_bytes in fin:
                stripped = line_bytes.rstrip(b"\r\n")
                if not stripped:
                    continue
                parts = stripped.split(b"\t")
                if ev_idx < len(parts) and parts[ev_idx]:
                    counts[parts[ev_idx].decode("utf-8", errors="replace")] += 1
    except Exception:
        return Counter()
    return counts


def _find_marker_for_filename(
    filename: str, candidate_markers: list[str],
) -> str | None:
    """Match an uploaded stimulus filename to a recording marker by
    Q-prefix (`Q1`, `Q2A`, `Q5b`, …). Case-insensitive.

    When multiple markers share the same Q-prefix (e.g. both
    ``Q1 (localhost)`` and ``Q1 Prompt (localhost)`` exist in the
    recording), the shortest marker wins. This is the right heuristic
    for Tobii Pro Lab recordings where the bare ``Q1 (localhost)`` is
    the stimulus event and the longer ``Q1 Prompt (localhost)`` is the
    instructions-screen event before it.

    Returns ``None`` only when no marker has a matching Q-prefix.
    """
    file_prefix = _extract_q_prefix(Path(filename).stem)
    if not file_prefix:
        return None
    matches = [
        m for m in candidate_markers
        if _extract_q_prefix(m) == file_prefix
    ]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    return min(matches, key=len)


def _extract_q_prefix(s: str) -> str | None:
    """Return the leading Q-prefix (e.g. ``Q1``, ``Q2A``) of ``s``,
    uppercased, or ``None`` if there is no Q-prefix at the start."""
    m = _Q_PREFIX_RE.match(s)
    return m.group(0).upper() if m else None


# ---------------------------------------------------------------------------
# Step 3 — config snippet
# ---------------------------------------------------------------------------

def _step3_config_snippet(*, name: str, raw_dir: Path, stim_dir: Path) -> None:
    st.subheader("Step 3 — Register the dataset (paste into config)")

    raw_exists = raw_dir.exists() and any(raw_dir.iterdir())
    stim_files = sorted(stim_dir.glob("*.png")) if stim_dir.exists() else []

    if not raw_exists or not stim_files:
        missing = []
        if not raw_exists: missing.append("Step 1 (split TSV)")
        if not stim_files: missing.append("Step 2 (upload stimuli)")
        st.info(f"Complete first: {', '.join(missing)}.")
        return

    stim_stems = [f.stem for f in stim_files]
    stim_tuple = ",\n        ".join(repr(s) for s in stim_stems)
    n_participants = len(list(raw_dir.glob("*.tsv")))

    # Render the raw / stimuli dirs as relative paths from cwd when possible,
    # so the snippet stays portable across machines.
    def _render(path: Path) -> str:
        try:
            return path.relative_to(Path.cwd()).as_posix()
        except ValueError:
            return path.as_posix()

    # Scan one TSV from rawdata to detect any column-name overrides the
    # newer Tobii Pro Lab export uses (e.g. "Eye movement event duration"
    # replacing "Gaze event duration"). Injected directly into the snippet
    # so the user does not have to remember to patch it manually.
    overrides = _detect_columns_override(raw_dir)
    if overrides:
        ov_pairs = ", ".join(f"{k!r}: {v!r}" for k, v in overrides.items())
        columns_line = f'"columns": {{**TOBII_PRO_COLUMNS, {ov_pairs}}},'
    else:
        columns_line = '"columns": TOBII_PRO_COLUMNS,'

    snippet = f'''DATASETS["{name}"] = {{
    "eye_tracker": "Tobii I-VT (Fixation)",
    "raw_dir": "{_render(raw_dir)}",
    "stimuli_dir": "{_render(stim_dir)}",
    "stimuli_names": (
        {stim_tuple},
    ),
    "n_stimuli": {len(stim_stems)},
    {columns_line}
    "participant_col": "Participant name",
    "trial_split": {{"strategy": "paired_markers", "per_trial": 3}},
    "stimuli_name_template": "{{event_value}}.png",
    "trial_id_strategy": "first_word",
    "fixation_label": "Fixation",
    "drop_cols_from": 99,             # TODO verify — AOI block boundary
    "samples_drop_cols": TOBII_PRO_SAMPLES_DROP_COLS,
    "default_sample_size": {n_participants},
}}
'''
    st.code(snippet, language="python")

    config_path = Path("g2c/parsers/datasets_config.py")
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    c1, c2 = st.columns([1, 3])
    with c1:
        do_append = st.button(
            "Append to datasets_config.py",
            type="primary",
            key="ob_cfg_append",
            help="Writes the snippet above to the bottom of "
                 "`g2c/parsers/datasets_config.py` as a "
                 f"`DATASETS['{name}'] = ...` assignment. Refuses to "
                 "overwrite if an entry for this name already exists.",
        )
    with c2:
        st.caption(
            f"Or copy the snippet above into `{config_path.relative_to(Path.cwd()) if Path.cwd() in config_path.parents else config_path}` yourself."
        )

    if do_append:
        _append_to_config(config_path, name, snippet)

    st.warning(
        "**Two values you should verify before relying on the new dataset:**\n\n"
        "1. **`drop_cols_from`** — defaults to 99 (correct for UNL_UM). If "
        "your export has extra AOI-definition columns, the right index "
        "differs. Look at the header of one of the split TSVs and find the "
        "column where `AOI hit [...]` columns end and metadata resumes.\n\n"
        "2. **`stimuli_name_template`** — assumes the Event-value marker in "
        "the recording exactly matches the stimulus filename (minus `.png`). "
        "If your stimuli are named differently from the markers (e.g., "
        "`Q1-SpecifyOutput.png` vs marker `Q1 (localhost)`), either rename "
        "the PNGs to match or add a mapping in the parser."
    )


# ---------------------------------------------------------------------------
# Step 4 — remove dataset
# ---------------------------------------------------------------------------

def _step4_remove_dataset(*, datasets_root: Path) -> None:
    st.subheader("Step 4 — Remove an existing dataset")
    st.caption(
        "Tear down a dataset that was added by this tab (or any other "
        "dataset registered in `datasets_config.py`). Lets you remove the "
        "data folder, the config entry, or both. Each operation is "
        "independent."
    )

    # Only offer datasets that are actually registered in
    # `datasets_config.py` — i.e. the ones the rest of the toolkit can
    # see. Orphan on-disk folders (no config entry) are not listed here;
    # remove them manually with the shell if needed.
    try:
        from g2c.parsers import available_datasets
        registered = sorted(available_datasets())
    except Exception as e:
        st.warning(f"Could not read registered datasets: {e}")
        return

    if not registered:
        st.info("No registered datasets to remove.")
        return

    target = st.selectbox(
        "Dataset to remove",
        registered,
        index=0,
        key="ob_rm_target",
        format_func=lambda n: (
            f"{n}  "
            + ("[folder ✓]" if (datasets_root / n).is_dir() else "[folder —]")
        ),
    )

    target_folder = datasets_root / target
    config_path = Path("g2c/parsers/datasets_config.py")
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    folder_exists = target_folder.is_dir()
    config_has_entry = True  # selectbox only lists registered names

    folder_size_str = ""
    if folder_exists:
        try:
            total = sum(p.stat().st_size for p in target_folder.rglob("*") if p.is_file())
            folder_size_str = f" ({total / 1e6:.1f} MB)" if total < 1e9 else f" ({total / 1e9:.2f} GB)"
        except Exception:
            folder_size_str = ""

    c1, c2 = st.columns(2)
    with c1:
        do_folder = st.checkbox(
            f"Delete data folder `{target_folder}`{folder_size_str}",
            value=folder_exists,
            disabled=not folder_exists,
            key="ob_rm_folder",
        )
    with c2:
        do_config = st.checkbox(
            f"Delete config entry `DATASETS[\"{target}\"]`",
            value=config_has_entry,
            disabled=not config_has_entry,
            key="ob_rm_config",
        )

    if not (do_folder or do_config):
        st.info("Tick at least one box to enable removal.")
        return

    confirm = st.checkbox(
        f"I confirm I want to remove **{target}** — this cannot be undone.",
        value=False, key="ob_rm_confirm",
    )

    if not st.button("Delete", type="secondary", key="ob_rm_run",
                     disabled=not confirm):
        return

    if do_folder and folder_exists:
        try:
            shutil.rmtree(target_folder)
            st.success(f"Deleted folder `{target_folder}`.")
        except Exception as e:
            st.error(f"Failed to delete folder: {e}")

    if do_config and config_has_entry:
        removed, msg = _remove_from_config(config_path, target)
        if removed:
            st.success(msg + "  Dropdowns refreshed.")
            # Force the current rerun to re-render the selectbox with the
            # updated registered list so `target` disappears immediately
            # instead of lingering until the next interaction.
            st.rerun()
        else:
            st.error(msg)


def _find_matching_brace(text: str, open_pos: int) -> int:
    """Return the index of the `}` that matches the `{` at ``open_pos``.

    Naïve brace counter — accurate for our config file because none of
    the string values in the DATASETS entries contain literal `{` or `}`
    characters. Robust enough for both the dict-literal style
    (``"NAME": { ... }`` inside ``DATASETS = { ... }``) and the appended
    assignment style (``DATASETS["NAME"] = { ... }``).
    """
    assert text[open_pos] == "{"
    depth = 0
    i = open_pos
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("Unbalanced braces while removing DATASETS entry.")


def _remove_from_config(config_path: Path, name: str) -> tuple[bool, str]:
    """Remove the ``DATASETS["<name>"]`` block from ``datasets_config.py``.

    Handles two styles:
    - **Appended assignment** (added by Step 3 of this tab):
      ``DATASETS["NAME"] = { ... }`` at the top level.
    - **Dict-literal entry** (the original UNL_UM / YMU_UM style):
      ``    "NAME": { ... },`` inside the ``DATASETS = { ... }`` block.
    """
    if not config_path.exists():
        return False, f"Config file not found: `{config_path}`"

    text = config_path.read_text(encoding="utf-8")

    pat_appended = re.compile(
        rf'^DATASETS\["{re.escape(name)}"\]\s*=\s*\{{',
        flags=re.MULTILINE,
    )
    pat_literal = re.compile(
        rf'^([ \t]+)"{re.escape(name)}"\s*:\s*\{{',
        flags=re.MULTILINE,
    )

    m_app = pat_appended.search(text)
    m_lit = pat_literal.search(text)

    if not m_app and not m_lit:
        return False, (
            f'`DATASETS["{name}"]` not found in '
            f'`{config_path.relative_to(Path.cwd()) if Path.cwd() in config_path.parents else config_path}`.'
        )

    if m_app:
        start = m_app.start()
        brace = text.index("{", m_app.end() - 1)
        end = _find_matching_brace(text, brace) + 1
        # Consume one trailing newline so the file stays tidy.
        if end < len(text) and text[end] == "\n":
            end += 1
        # Also pull back any blank line directly above the assignment.
        if start >= 2 and text[start - 2 : start] == "\n\n":
            start -= 1
        new_text = text[:start] + text[end:]
        style = "appended assignment"
    else:
        # Dict-literal entry — preserve trailing comma + newline handling.
        start = m_lit.start()
        brace = text.index("{", m_lit.end() - 1)
        end = _find_matching_brace(text, brace) + 1
        # Consume optional trailing comma and whitespace through the next newline.
        if end < len(text) and text[end] == ",":
            end += 1
        while end < len(text) and text[end] in " \t":
            end += 1
        if end < len(text) and text[end] == "\n":
            end += 1
        new_text = text[:start] + text[end:]
        style = "dict-literal entry"

    config_path.write_text(new_text, encoding="utf-8")
    _reload_datasets_config()
    rel = (config_path.relative_to(Path.cwd())
           if Path.cwd() in config_path.parents else config_path)
    return True, f'Removed `DATASETS["{name}"]` ({style}) from `{rel}`.'


def _append_to_config(config_path: Path, name: str, snippet: str) -> None:
    """Append the generated snippet to ``datasets_config.py``.

    Refuses if a ``DATASETS["<name>"]`` entry already exists, to avoid
    silently shadowing user edits. The append is plain text — no AST —
    because the snippet shape is fixed and we never want to touch
    surrounding code.
    """
    if not config_path.exists():
        st.error(f"Config file not found: `{config_path}`")
        return

    current = config_path.read_text(encoding="utf-8")
    marker = f'DATASETS["{name}"]'
    if marker in current:
        st.error(
            f"`{marker}` already exists in `{config_path.name}`. "
            "Pick a different dataset name, or remove the existing entry "
            "in the file first."
        )
        return

    # Append after a blank line so the new block is clearly separated.
    sep = "" if current.endswith("\n\n") else ("\n" if current.endswith("\n") else "\n\n")
    new_content = current + sep + snippet
    if not new_content.endswith("\n"):
        new_content += "\n"

    config_path.write_text(new_content, encoding="utf-8")
    _reload_datasets_config()

    rel = (config_path.relative_to(Path.cwd())
           if Path.cwd() in config_path.parents else config_path)
    st.success(
        f"Appended `{marker}` to `{rel}`.  "
        f"`{name}` is now visible in every dataset dropdown — no need "
        "to press R."
    )
