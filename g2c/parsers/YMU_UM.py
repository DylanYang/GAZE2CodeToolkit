import os
import pandas as pd
import numpy as np
from .eye_events import get_eye_event_columns
from tqdm import tqdm

EYE_TRACKER = "Tobii I-VT (Fixation)"
RAWDATA_MODULE = "datasets/YMU_UM/rawdata"
STIMULI_MODULE = "datasets/YMU_UM/stimuli"

STIMULI_NAMES = (
    "Quiz - introduction-Q1 (localhost)", "Quiz - introduction-Q3 (localhost)",
    "Quiz - introduction-Q2A (localhost)", "Quiz - introduction-Q2B (localhost)",
    "Quiz - introduction-Q4A (localhost)", "Quiz - introduction-Q4B (localhost)",
    "Quiz - introduction-Q5 (localhost)"
)

STIMULIS_PER_EXPERIMENT = 7  # 7 stimulus markers per experiment


def YMU_UM(sample_size: int = 84):
    eye_events = pd.DataFrame()
    samples = pd.DataFrame()
    all_files = []

    for r, _, f in os.walk(RAWDATA_MODULE):
        f.sort()
        all_files.extend([file for file in f if file.endswith('.tsv')])

        for file in tqdm(all_files[:sample_size], desc="Processing files", unit="file", colour="green"):
            raw_data = pd.read_csv(os.path.join(r, file), sep="\t")

            try:
                # Get first occurrence index for each unique stimulus name
                stimuli_rows = raw_data[raw_data["Event value"].isin(STIMULI_NAMES)]
                stimuli_rows = stimuli_rows.drop_duplicates(subset=["Event value"], keep="first")
                stimuli_start_idx = stimuli_rows.index.tolist()

                if len(stimuli_start_idx) != STIMULIS_PER_EXPERIMENT:
                    raise ValueError(f"Expected 7 unique stimuli, found {len(stimuli_start_idx)}")


                for idx in range(STIMULIS_PER_EXPERIMENT):
                    trial_start = stimuli_start_idx[idx]
                    trial_end = stimuli_start_idx[idx + 1] if idx < STIMULIS_PER_EXPERIMENT - 1 else len(raw_data)
                    trial_samples = raw_data.iloc[trial_start: trial_end].copy()

                    participant_id = raw_data["Participant name"].iloc[0]
                    stimuli_name_raw = raw_data["Event value"].iloc[trial_start]
                    stimuli_name = "{}.png".format(stimuli_name_raw.split()[2])  # extract "Q1", etc.
                    trial_id = stimuli_name.split(".")[0]

                    trial_samples["eye_tracker"] = EYE_TRACKER
                    trial_samples["experiment_id"] = participant_id
                    trial_samples["participant_id"] = participant_id
                    trial_samples["filename"] = file
                    trial_samples["trial_id"] = trial_id
                    trial_samples["stimuli_module"] = STIMULI_MODULE
                    trial_samples["stimuli_name"] = stimuli_name

                    samples = pd.concat([samples, trial_samples])

                    trial_eye_events = trial_samples[[
                        "Recording timestamp", "Gaze event duration",
                        "Fixation point X", "Fixation point Y",
                        "Eye movement type", "Pupil diameter left", "Pupil diameter right"
                    ]].copy()

                    trial_eye_events.rename(columns={
                        "Recording timestamp": "timestamp",
                        "Gaze event duration": "duration",
                        "Fixation point X": "x0",
                        "Fixation point Y": "y0",
                        "Eye movement type": "eye_event_type",
                        "Pupil diameter left": "pupil_l",
                        "Pupil diameter right": "pupil_r"
                    }, inplace=True)

                    trial_eye_events["eye_tracker"] = EYE_TRACKER
                    trial_eye_events["experiment_id"] = participant_id
                    trial_eye_events["participant_id"] = participant_id
                    trial_eye_events["filename"] = file
                    trial_eye_events["trial_id"] = trial_id
                    trial_eye_events["stimuli_module"] = STIMULI_MODULE
                    trial_eye_events["stimuli_name"] = stimuli_name
                    trial_eye_events["x1"] = np.nan
                    trial_eye_events["y1"] = np.nan
                    trial_eye_events["token"] = None
                    trial_eye_events["amplitude"] = np.nan
                    trial_eye_events["peak_velocity"] = np.nan

                    trial_eye_events = trial_eye_events[
                        trial_eye_events["eye_event_type"] == "Fixation"]
                    trial_eye_events["eye_event_type"] = "fixation"

                    eye_events = pd.concat([eye_events, trial_eye_events])

                print("Processed file:", file)

            except Exception as e:
                print(f"Stimuli not found or error in file: {file} | Error: {e}")
                continue

    # Drop unnecessary columns safely
    cols_to_drop = [
        "Computer timestamp", "Sensor", "Eyetracker timestamp", "Event", "Event value",
        "Gaze point X (MCSnorm)", "Gaze point Y (MCSnorm)",
        "Gaze point left X (MCSnorm)", "Gaze point left Y (MCSnorm)",
        "Gaze point right X (MCSnorm)", "Gaze point right Y (MCSnorm)",
        "Fixation point X (MCSnorm)", "Fixation point Y (MCSnorm)",
        "Fixation point X", "Fixation point Y"
    ]
    samples.drop(columns=[col for col in cols_to_drop if col in samples.columns], inplace=True)

    # Rearrange columns
    eye_events = eye_events[get_eye_event_columns()]
    samples = pd.concat([samples.loc[:, "eye_tracker":],
                         samples.loc[:, : "Eye movement type index"]], axis=1)

    eye_events.reset_index(drop=True, inplace=True)
    samples.reset_index(drop=True, inplace=True)
    return eye_events, samples
