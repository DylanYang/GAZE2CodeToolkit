"""Tab 2 — OCR token AOIs + fixation hit-test.

Two-stage UI:
  Step 1 — detect token AOIs on a stimulus image (OCR).
  Step 2 — map fixations to those AOIs in either GROUP mode (all
           participants on the trial) or INDIVIDUAL mode (one
           participant).

OCR output is cached in session_state so the hit-test step can re-run
without re-running Tesseract every time a slider changes.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from g2c import aoi as aoi_pkg

from ..components import (dataset_path_overrides, dataset_picker,
                           output_dir_input, sample_size_slider, trial_picker)
from ..state import (ensure_output_dir, load_dataset, load_dataset_with_bar,
                      save_uploaded_file)


def render() -> None:
    st.header("OCR token AOIs and fixation hit-test")
    st.caption(
        "Step 1: detect token bounding boxes with Tesseract.  "
        "Step 2: map fixations to those boxes — either for the whole "
        "**group** of participants on a trial, or for one **individual** "
        "participant.  Equivalent to `python -m cli.extract_aoi`."
    )

    # =====================================================================
    # Step 1 — OCR detection
    # =====================================================================
    st.subheader("Step 1 · Detect token AOIs (OCR)")

    source = st.radio("Stimulus source", ["Dataset stimulus", "Upload image"],
                       horizontal=True, key="aoi_source")

    image_path: Path | None = None
    dataset: str | None = None
    trial_id: str | None = None
    raw_dir_override: str | None = None
    stimuli_dir_override: str | None = None

    if source == "Dataset stimulus":
        c1, c2 = st.columns(2)
        with c1:
            dataset = dataset_picker(key="aoi_dataset")
            trial_id = trial_picker(dataset, key="aoi_trial")
        with c2:
            image_prefix = st.text_input(
                "Filename prefix",
                value="Quiz - " if dataset == "YMU_UM" else "",
                key="aoi_prefix",
            )
            image_suffix = st.text_input(
                "Filename suffix",
                value=" (localhost).png",
                key="aoi_suffix",
            )

        # Path overrides — picks up both raw_dir (for the hit-test step)
        # and stimuli_dir (used as the default image directory below).
        (raw_dir_override, stimuli_dir_override,
          _eff_raw, eff_stimuli) = dataset_path_overrides(
              dataset, key_prefix="aoi"
        )
        image_dir = st.text_input(
            "Stimulus image directory",
            value=eff_stimuli,
            key=f"aoi_imgdir_{dataset}",
            help="Defaults to the dataset's stimuli_dir; override above.",
        )
        image_path = Path(image_dir) / f"{image_prefix}{trial_id}{image_suffix}"
        st.caption(f"Resolved path: `{image_path}`")
    else:
        uploaded = st.file_uploader("Stimulus image (PNG/JPG)",
                                    type=["png", "jpg", "jpeg"],
                                    key="aoi_upload")
        if uploaded is None:
            st.info("Upload a stimulus image to continue.")
            return
        image_path = save_uploaded_file(uploaded)

    if image_path and not image_path.exists():
        st.error(f"Image not found: {image_path}")
        return

    with st.expander("OCR parameters"):
        c3, c4 = st.columns(2)
        with c3:
            scale = st.slider("scale_factor", 0.5, 4.0, 2.0, 0.1,
                              key="aoi_scale")
            min_conf = st.slider("min_confidence", 0, 100, 60,
                                 key="aoi_minconf")
        with c4:
            psm = st.selectbox("PSM (page-segmentation mode)",
                               ["6", "3", "1", "11"], index=0, key="aoi_psm")
            oem = st.selectbox("OEM (engine mode)",
                               ["3", "1"], index=0, key="aoi_oem")
        preprocess = st.checkbox("Use CLAHE / adaptive-threshold preprocessing",
                                 value=False, key="aoi_preprocess")

    if st.button("Run OCR", type="primary", key="aoi_run_ocr"):
        progress_bar = st.progress(0.0, text="Preparing OCR …")

        def _on_progress(fraction: float, message: str) -> None:
            # Streamlit clamps to [0, 1] anyway, but be defensive.
            progress_bar.progress(min(max(fraction, 0.0), 1.0), text=message)

        aoi_df = aoi_pkg.aoi_detector(
            str(image_path),
            scale_factor=scale,
            debug=False,
            use_preprocessing=preprocess,
            min_confidence=min_conf,
            psm=psm,
            oem=oem,
            progress_callback=_on_progress,
        )
        # Briefly hold the completion message, then clear the bar.
        import time
        time.sleep(0.4)
        progress_bar.empty()
        if aoi_df.empty:
            st.warning("No tokens detected. Try lowering `min_confidence` "
                       "or enabling preprocessing.")
            for k in ("_aoi_cache_df", "_aoi_cache_image",
                      "_aoi_cache_dataset", "_aoi_cache_trial_id"):
                st.session_state.pop(k, None)
            return
        # Cache across reruns so hit-test parameter changes don't re-trigger
        # OCR. The leading underscore avoids any collision with widget keys.
        st.session_state["_aoi_cache_df"] = aoi_df
        st.session_state["_aoi_cache_image"] = str(image_path)
        st.session_state["_aoi_cache_dataset"] = dataset
        st.session_state["_aoi_cache_trial_id"] = trial_id
        st.session_state["_aoi_cache_raw_dir"] = raw_dir_override
        st.session_state["_aoi_cache_stimuli_dir"] = stimuli_dir_override

    if "_aoi_cache_df" not in st.session_state:
        st.info("Click **Run OCR** to detect token AOIs.")
        return

    aoi_df: pd.DataFrame = st.session_state["_aoi_cache_df"]
    cached_image_path = Path(st.session_state["_aoi_cache_image"])
    cached_dataset: str | None = st.session_state.get("_aoi_cache_dataset")
    cached_trial_id: str | None = st.session_state.get("_aoi_cache_trial_id")
    cached_raw_dir: str | None = st.session_state.get("_aoi_cache_raw_dir")
    cached_stim_dir: str | None = st.session_state.get("_aoi_cache_stimuli_dir")

    st.success(f"Detected **{len(aoi_df)}** token AOIs on `{cached_image_path.name}`.")
    overlay = _draw_aoi_overlay(cached_image_path, aoi_df)
    c5, c6 = st.columns(2)
    with c5:
        st.image(str(cached_image_path), caption="Original stimulus")
    with c6:
        st.image(overlay, caption="OCR token overlay")

    with st.expander("Detected tokens table"):
        st.dataframe(aoi_df, use_container_width=True)

    # =====================================================================
    # Step 2 — Fixation hit-test
    # =====================================================================
    if cached_dataset is None or cached_trial_id is None:
        st.info("Step 2 requires a stimulus from a registered dataset. "
                "Uploaded images can only run Step 1.")
        return

    st.divider()
    st.subheader("Step 2 · Map fixations to AOIs")

    mode = st.radio(
        "Mapping mode",
        options=["group", "individual"],
        format_func=lambda m: ("Group  (all participants on this trial)"
                               if m == "group"
                               else "Individual  (one participant)"),
        horizontal=True,
        key="aoi_mode",
    )
    is_individual = (mode == "individual")

    # --- Fixation source: extracted CSV (preferred) vs re-parse from TSVs ---
    fix_source = st.radio(
        "Fixation source",
        options=["extracted_csv", "parse_dataset"],
        format_func=lambda s: (
            "Extracted fixations CSV  (use output from the Extract tab — fast)"
            if s == "extracted_csv"
            else "Re-parse dataset TSVs  (always fresh — slow)"),
        horizontal=False,
        key="aoi_fix_source",
        help=(
            "After running the Extract tab the fixations are already on disk "
            "under `output/{dataset}/...`. Loading those CSVs is much faster "
            "than re-parsing the raw TSVs."
        ),
    )

    c7, c8 = st.columns(2)
    with c7:
        # Sample size only matters when we re-parse the TSVs; when reading
        # an extracted CSV, every participant the CSV contains is used.
        sample_size = sample_size_slider(
            default=20, max_value=100, key="aoi_n",
        ) if fix_source == "parse_dataset" else 0
        if fix_source == "extracted_csv":
            st.caption("`sample_size` ignored — CSV defines its own scope.")
    with c8:
        radius = st.slider(
            "hit_test radius (px)", 5, 80, 35, key="aoi_radius",
            help="Default 35 matches the effective historical behaviour "
                 "(legacy code silently used 35 regardless of the value "
                 "passed in). Use 35 to stay comparable with the locked "
                 "ECPG training tables.",
        )

    # Default output root sits under `aoi/` so all AOI artefacts
    # (token structures + hit-test CSVs) live in one folder:
    #   output/{dataset}/aoi/aoi_tokens_structure/...
    #   output/{dataset}/aoi/group/aoi_fixations_{trial}.csv
    #   output/{dataset}/aoi/individual/aoi_fixations_{trial}_{participant}.csv
    out_root = output_dir_input(
        default=f"output/{cached_dataset.lower()}/aoi",
        key="aoi_outroot",
    )

    # --- Resolve the fixation table the user wants to map ---
    trial_data: pd.DataFrame | None = None
    candidates: list[str] = []

    if fix_source == "extracted_csv":
        # Mirrors the three Extract-tab modes — all now live under
        # `extracted-fixations/`. The legacy `group/fixations` layout
        # is retained for backward compatibility with old notebooks.
        dataset_lower = cached_dataset.lower()
        ext_root = f"output/{dataset_lower}/extracted-fixations"
        per_part_dir = (f"{ext_root}/per-participant/{cached_trial_id}")
        per_part_globs = sorted(Path(per_part_dir).glob(
            f"fixations_{cached_trial_id}_*.csv"
        )) if Path(per_part_dir).exists() else []

        # Layout → representation. For single-file layouts this is the
        # CSV path; for `per-participant` it's the directory that holds
        # one CSV per participant (concat for group / pick one for
        # individual). `None` means user must upload or type.
        layout_paths = {
            "all":            f"{ext_root}/all/fixations_all.csv",
            "by-task":        (f"{ext_root}/by-task/"
                                f"fixations_{cached_trial_id}.csv"),
            "per-participant": per_part_dir,
            "group (legacy)": (f"output/{dataset_lower}/group/fixations/"
                                f"fixations_{cached_trial_id}.csv"),
            "custom path":    None,
            "upload":         None,
        }
        layout_labels = {
            "all":             "all  (single CSV with every trial)",
            "by-task":         "by-task  (one CSV per trial — all participants)",
            "per-participant": ("per-participant  (one CSV per participant — "
                                "concat for group, pick one for individual)"),
            "group (legacy)":  "group (legacy)  (old notebook layout)",
            "custom path":     "custom path  (type one)",
            "upload":          "upload  (browse file)",
        }

        def _layout_exists(name: str) -> bool:
            p = layout_paths.get(name)
            if not p:
                return False
            if name == "per-participant":
                return bool(per_part_globs)
            return Path(p).exists()

        # "Natural" layouts for each mapping mode — used only for the
        # `(unusual for …)` hint in the dropdown. ALL combinations
        # still work; the hint just flags that the chosen layout is
        # not the obvious one for the active mode.
        _natural_for_group = {"all", "by-task", "group (legacy)"}
        _natural_for_individual = {"per-participant"}

        def _unusual_hint(name: str) -> str:
            if name in ("custom path", "upload"):
                return ""
            if is_individual and name not in _natural_for_individual:
                return " (unusual for individual)"
            if (not is_individual) and name not in _natural_for_group:
                return " (unusual for group)"
            return ""

        # Auto-pick a sensible default that respects the chosen
        # mapping mode:
        #   - individual → prefer `per-participant`
        #   - group      → prefer `by-task` (matches the typical
        #                  one-CSV-per-trial workflow; `all` is the
        #                  fallback if no `by-task` CSV exists)
        # State key includes `mode` so switching group↔individual
        # re-evaluates the default instead of inheriting the wrong
        # layout.
        if is_individual:
            preferred = ["per-participant", "by-task", "all",
                         "group (legacy)"]
        else:
            preferred = ["by-task", "all", "per-participant",
                         "group (legacy)"]
        default_layout = next(
            (name for name in preferred if _layout_exists(name)),
            "by-task",
        )

        options_list = list(layout_paths.keys())
        layout_state_key = (f"aoi_fixcsv_layout_"
                             f"{cached_dataset}_{cached_trial_id}_{mode}")

        layout_choice = st.selectbox(
            "Fixations CSV layout",
            options=options_list,
            index=options_list.index(default_layout),
            format_func=lambda k: (
                f"{layout_labels[k]}"
                f"{_unusual_hint(k)}  "
                f"{'[found]' if _layout_exists(k) else ''}"
            ),
            key=layout_state_key,
        )

        # If the user picked an unusual combination, surface a small
        # warning under the selectbox so it's not silently weird.
        _hint = _unusual_hint(layout_choice)
        if _hint:
            st.caption(
                f"ℹ️ `{layout_choice}` is unusual for **{mode}** mode — "
                "it still works (filter / concat handles it), but the "
                "natural source for this mode is "
                + ("`per-participant`."
                   if is_individual
                   else "`all` or `by-task`.")
            )

        # Resolve the path text input default from the chosen layout.
        if layout_choice == "upload":
            csv_path_str = ""
        elif layout_choice == "custom path":
            csv_path_str = st.text_input(
                "Fixations CSV path",
                value=(layout_paths.get("by-task") or ""),
                key=f"aoi_fixcsv_custom_{cached_dataset}_{cached_trial_id}",
            )
        elif layout_choice == "per-participant":
            n_found = len(per_part_globs)
            if n_found:
                st.caption(
                    f"Per-participant layout → directory `{per_part_dir}/` "
                    f"({n_found} CSV(s) found). "
                    + ("All CSVs will be concatenated for group mode."
                       if not is_individual
                       else "The selected participant's CSV will be loaded.")
                )
            else:
                st.caption(
                    f"Per-participant layout → directory `{per_part_dir}/` "
                    f"(no CSVs found — run the Extract tab in "
                    f"`per-participant` mode first)."
                )
            csv_path_str = per_part_dir
        else:
            csv_path_str = layout_paths[layout_choice] or ""
            st.caption(f"Path → `{csv_path_str}`  "
                       f"{'(found)' if Path(csv_path_str).exists() else '(missing)'}")

        uploaded_csv = None
        if layout_choice == "upload":
            uploaded_csv = st.file_uploader(
                "Upload a fixations CSV",
                type=["csv"],
                key=f"aoi_fixcsv_upload_{cached_dataset}_{cached_trial_id}",
            )

        loaded_df: pd.DataFrame | None = None
        if uploaded_csv is not None:
            try:
                loaded_df = pd.read_csv(uploaded_csv)
            except Exception as exc:
                st.error(f"Could not read uploaded CSV: {exc}")
                return
        elif layout_choice == "per-participant":
            if not per_part_globs:
                st.warning(
                    f"No per-participant CSVs found under "
                    f"`{per_part_dir}/`. Run the Extract tab in "
                    f"`per-participant` mode first."
                )
                return
            try:
                loaded_df = pd.concat(
                    [pd.read_csv(p) for p in per_part_globs],
                    ignore_index=True,
                )
            except Exception as exc:
                st.error(f"Could not read per-participant CSVs: {exc}")
                return
        elif csv_path_str and Path(csv_path_str).exists():
            try:
                loaded_df = pd.read_csv(csv_path_str)
            except Exception as exc:
                st.error(f"Could not read `{csv_path_str}`: {exc}")
                return
        else:
            st.warning(
                f"CSV not found at `{csv_path_str}`. "
                f"Run the Extract tab, upload a CSV, or switch to "
                f"**Re-parse dataset TSVs**."
            )
            return

        # Filter to the active trial and (optionally) one participant.
        if "trial_id" in loaded_df.columns:
            loaded_df = loaded_df[
                loaded_df["trial_id"].astype(str) == str(cached_trial_id)
            ]
        if loaded_df.empty:
            st.error(
                f"Loaded CSV contains no rows for trial=`{cached_trial_id}`."
            )
            return

        if is_individual and "experiment_id" in loaded_df.columns:
            candidates = sorted(
                loaded_df["experiment_id"].astype(str).unique()
            )

    else:  # parse_dataset
        if is_individual:
            eye_events_pre, _ = load_dataset_with_bar(
                cached_dataset, sample_size,
                bar_text_prefix=(f"Loading {cached_dataset} participant list "
                                  f"(sample_size={sample_size})"),
                raw_dir=cached_raw_dir,
                stimuli_dir=cached_stim_dir,
            )
            candidates = sorted(
                eye_events_pre[eye_events_pre["trial_id"] == cached_trial_id]
                ["experiment_id"].astype(str).unique()
            )

    # Participant selectbox (both paths feed `candidates`).
    selected_participant: str | None = None
    if is_individual:
        if not candidates:
            st.warning(
                f"No participants have fixations on trial `{cached_trial_id}` "
                f"in the chosen source. "
                f"Increase sample_size (re-parse mode), or pick a different "
                f"fixations CSV."
            )
            return
        selected_participant = st.selectbox(
            f"Participant ({len(candidates)} available on trial "
            f"`{cached_trial_id}`)",
            candidates,
            key="aoi_participant_sel",
        )

    # Preview the output path so the user can confirm where the CSV
    # will go. `out_root` already lives under `aoi/` (see default
    # above), so the subfolders here are just `individual/` or
    # `group/` — no need to repeat the `aoi/` segment.
    if is_individual:
        out_csv_preview = (out_root / "individual"
                           / f"aoi_fixations_{cached_trial_id}_"
                             f"{selected_participant}.csv")
    else:
        out_csv_preview = (out_root / "group"
                           / f"aoi_fixations_{cached_trial_id}.csv")
    st.caption(f"Output → `{out_csv_preview}`")

    if not st.button("Run hit-test AOIs Mapping", type="primary",
                       key="aoi_run_hit"):
        return

    # ---- Resolve the actual `trial_data` DataFrame to hit-test ----
    if fix_source == "extracted_csv":
        # `loaded_df` already filtered to the active trial above.
        eye_events = loaded_df  # noqa: F821 — defined in the extracted_csv branch
    else:
        eye_events, _ = load_dataset_with_bar(
            cached_dataset, sample_size,
            bar_text_prefix=f"Loading {cached_dataset} fixations for hit-test",
            raw_dir=cached_raw_dir,
            stimuli_dir=cached_stim_dir,
        )

    if is_individual:
        trial_data = eye_events[
            (eye_events["experiment_id"] == selected_participant)
            & (eye_events["trial_id"] == cached_trial_id)
        ]
        mode_label = f"Individual · participant `{selected_participant}`"
    else:
        trial_data = eye_events[eye_events["trial_id"] == cached_trial_id]
        n_participants = trial_data["experiment_id"].nunique()
        mode_label = f"Group · {n_participants} participant(s)"

    if trial_data.empty:
        st.error(
            f"No fixations match trial=`{cached_trial_id}`"
            + (f", participant=`{selected_participant}`"
               if is_individual else "")
            + ".  Increase **Participants to parse** or check the trial ID."
        )
        return

    struct_dir = ensure_output_dir(out_root / "aoi_tokens_structure")
    aoi_pkg.aoi_save_tokens_structure(aoi_df, str(struct_dir) + "/")
    struct_csv = struct_dir / f"aoi_{cached_trial_id}_tokens_structure.csv"

    hit_progress = st.progress(0.0, text="Preparing hit-test …")

    def _on_hit_progress(fraction: float, message: str) -> None:
        hit_progress.progress(min(max(fraction, 0.0), 1.0), text=message)

    aoi_fixations = aoi_pkg.aoi_tokens_matcher(
        str(struct_csv), trial_data, cached_trial_id, radius=radius,
        progress_callback=_on_hit_progress,
    )

    import time as _t
    _t.sleep(0.4)
    hit_progress.empty()

    out_csv_preview.parent.mkdir(parents=True, exist_ok=True)
    aoi_fixations.to_csv(out_csv_preview, index=False)

    st.success(
        f"**Hit-test complete** — {mode_label}.  "
        f"{len(aoi_fixations):,} fixation × AOI rows written to "
        f"`{out_csv_preview}`."
    )

    # Per-token coverage summary so the user can see which AOIs got fixated.
    if "aoi_name" in aoi_fixations.columns:
        coverage = (aoi_fixations.groupby("aoi_name")
                    .agg(n_fixations=("aoi_name", "size"))
                    .reset_index()
                    .sort_values("n_fixations", ascending=False))
        st.subheader("AOI coverage")
        c9, c10 = st.columns([2, 3])
        with c9:
            st.metric("AOIs with ≥1 fixation",
                      f"{len(coverage)} / {len(aoi_df)}")
            st.metric("Total fixation×AOI rows", f"{len(aoi_fixations):,}")
        with c10:
            st.dataframe(coverage, use_container_width=True, height=240)

    with st.expander("Fixation × AOI preview (first 50 rows)"):
        st.dataframe(aoi_fixations.head(50), use_container_width=True)


def _draw_aoi_overlay(image_path: Path, aoi_df: pd.DataFrame) -> Image.Image:
    """Return a copy of the stimulus with detected AOI boxes overlaid."""
    im = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    drw = ImageDraw.Draw(overlay)
    for _, row in aoi_df.iterrows():
        x, y, w, h = (float(row["x"]), float(row["y"]),
                      float(row["width"]), float(row["height"]))
        drw.rectangle([x, y, x + w, y + h], outline=(0, 200, 0, 220), width=2)
    return Image.alpha_composite(im, overlay)
