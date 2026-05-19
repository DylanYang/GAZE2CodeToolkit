# read data
import os
import pandas as pd
from tqdm import tqdm

def aoi_tokens_matcher(aoi_tokens_struc_file_path: str, trial_data: pd.DataFrame,
                       trial_id: str, radius: int = 35,
                       redius: int | None = None,
                       progress_callback=None) -> pd.DataFrame:
    """
    Match AOI tokens to the trial data.

    Parameters
    ----------
    aoi_tokens_struc_file_path : str
        Path to the AOI tokens structure file.
    trial_data : pd.DataFrame
        DataFrame containing trial data.
    trial_id : str
        Trial ID.
    radius : int, default 35
        Pixel tolerance used when matching a fixation to an AOI box.
        The default of 35 reproduces the effective behaviour of the
        legacy implementation (which silently hard-coded 35), so newly
        generated tables remain comparable with the historical ECPG
        training data.
    redius : int, optional
        Deprecated alias for `radius` — kept for backward compatibility
        with notebooks that called this function with the older typo.
    progress_callback : callable, optional
        If given, called as ``callback(fraction, message)`` at each of
        several stages: loading the AOI table, starting the loop, every
        ~1% of fixations processed, and on completion. Allows UI
        wrappers (Streamlit, tqdm, etc.) to render a progress bar.

    Developer: Wudao(Dylan) Yang < data: 05-02-2025 >
    """
    def _report(frac, msg):
        if progress_callback is not None:
            try:
                progress_callback(frac, msg)
            except Exception:
                pass

    # Honour the deprecated `redius=` keyword if a caller still uses it.
    if redius is not None:
        radius = redius

    _report(0.02, "Loading AOI token structure …")
    # Read the AOI tokens structure file. The `./` prefix preserves the
    # legacy path-joining behaviour expected by older notebooks.
    aoi_tokens_struc = pd.read_csv(f'./{aoi_tokens_struc_file_path}')

    n_fix = len(trial_data)
    n_aoi = len(aoi_tokens_struc)
    _report(0.05,
            f"Matching {n_fix:,} fixation(s) against {n_aoi} AOI(s) "
            f"(radius={radius}px) …")

    # Forward the caller-supplied radius (the previous version hard-coded 35).
    aoi_data = __match_fixations_to_aois(trial_data, aoi_tokens_struc,
                                          radius=radius,
                                          progress_callback=progress_callback)

    _report(1.0, f"Hit-test complete — {len(aoi_data):,} fixation × AOI rows.")
    return aoi_data

def __match_fixations_to_aois(fixations: pd.DataFrame, aois: pd.DataFrame, radius: int = 35,
                               progress_callback=None) -> pd.DataFrame:
    """
    Match fixations to AOIs and calculate fixation durations.

    Parameters:
    -----------
    fixations : pd.DataFrame
        DataFrame containing fixation data with coordinates and other attributes.
    aois : pd.DataFrame
        DataFrame containing AOI data with bounding box details.
    radius : int, optional (default=35)
        Radius around the AOI to include in its region.

    Returns:
    --------
    pd.DataFrame
        DataFrame matching fixations to AOIs with the calculated fixation durations.
        Developer: Wudao(Dylan) Yang < data: 05-02-2025 >
    """

    # Ensure 'x', 'y' columns in AOIs are present for matching fixations
    output_data = []

    n_total = fixations.shape[0]
    # Report progress at ~every 1% of fixations, capped to a sensible step.
    step = max(1, n_total // 100)

    # Loop through each fixation and check for AOI matches with a progress bar.
    # The outer iteration is the bottleneck; report once per `step` rows so
    # the UI callback fires often enough to feel responsive without burning
    # CPU on the callback itself.
    for i, (_, fixation) in enumerate(tqdm(
            fixations.iterrows(), total=n_total,
            desc="Matching fixations to AOIs", unit="fixation",
            colour="green")):
        if progress_callback is not None and (i % step == 0 or i == n_total - 1):
            # Reserve [0.05, 0.95] of the parent bar for the loop body;
            # the wrapper handles 0–0.05 (load) and 0.95–1.0 (finish).
            frac = 0.05 + 0.90 * (i + 1) / max(n_total, 1)
            try:
                progress_callback(
                    frac,
                    f"Matched {i + 1:,} / {n_total:,} fixations …",
                )
            except Exception:
                pass
        fx, fy = fixation["x0"], fixation["y0"]
        for _, aoi in aois.iterrows():
            ax, ay = aoi["x"], aoi["y"]
            aw, ah = aoi["width"], aoi["height"]
            
            # Check if the fixation point is within the AOI's bounding box (with optional margin)
            if (ax <= fx <= ax + aw) and (ay <= fy <= ay + ah):
                # Add matching details to output list
                output_data.append({
                    "eye_tracker": fixation["eye_tracker"],
                    "experiment_id": fixation["experiment_id"],
                    "participant_id": fixation["participant_id"],
                    "filename": fixation["filename"],
                    "trial_id": fixation["trial_id"],
                    "stimuli_module": fixation["stimuli_module"],
                    "stimuli_name": fixation["stimuli_name"],
                    "timestamp": fixation["timestamp"],
                    "duration": fixation["duration"],
                    "x0": fixation["x0"],
                    "y0": fixation["y0"],
                    "pupil_left": fixation["pupil_l"],
                    "pupil_right": fixation["pupil_r"],
                    "aoi_kind": aoi["kind"],
                    "aoi_name": aoi["name"],
                    "aoi_token": aoi["token"],
                    "aoi_x": aoi["x"],
                    "aoi_y": aoi["y"],
                    "aoi_width": aoi["width"],
                    "aoi_height": aoi["height"],
                    "image": aoi["image"],
                    "eye_event_type": fixation["eye_event_type"]
                })
    # Convert the list to a DataFrame
    result_df = pd.DataFrame(output_data)

    return result_df