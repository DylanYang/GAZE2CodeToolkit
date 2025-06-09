import os
import pandas as pd
import numpy as np
from .eye_events import get_eye_event_columns
from tqdm import tqdm

EYE_TRACKER = "Tobii I-VT (Fixation)"
RAWDATA_MODULE = "datasets/UNL_UM24_30July/rawdata"
STIMULI_MODULE = "datasets/UNL_UM24_30July/stimuli"

STIMULI_NAMES = (
    "Q1 (localhost)","Q3 (localhost)",
    "Q2A (localhost)","Q2B (localhost)",
    "Q4A (localhost)","Q4B (localhost)",
    "Q5 (localhost)"
)
# STIMULI_NAMES = (
#     "Q2A (localhost)",
# )

STIMULIS_PER_EXPERIMENT = 7 # 7 stimuli per experiment


def UNL_UM(sample_size: int = 44):
    """Import the UNL dataset.
    Fixation and saccades 

    Parameters
    ----------
    sample_size : int, optional (default 1)
        Number of subjects to be processed.

    Returns
    -------
    pandas.DataFrame
        Pandas dataframe of eye events from every experiment in the dataset.
    """
    eye_events = pd.DataFrame()
    samples = pd.DataFrame()
    all_files = []
    for r, _, f in os.walk(RAWDATA_MODULE):
        f.sort()
        all_files.extend([file for file in f if file.endswith('.tsv')])
    
        # Progress bar for file processing
        for file in tqdm(all_files[:sample_size], desc="Processing files", unit="file", colour="green"):
            raw_data = pd.read_csv(
                os.path.join(r, file), sep="\t")

            # Delete AOI hit test (from column 52 to the end)
            raw_data.drop(
                raw_data.columns[99:], axis=1, inplace=True)

            # Extract data from the three trials only
            # write a try except block to handle the case where the stimuli are not present
            
            try:
                stimuli_start_end_idx = raw_data.loc[raw_data["Event value"].isin(
                    STIMULI_NAMES)].index
                
                for idx in range(STIMULIS_PER_EXPERIMENT):
                    # Extract samples data
                    # there are 3 trials per stimuli (see the raw data)
                    trial_start = stimuli_start_end_idx[idx * 3]
                    trial_end = stimuli_start_end_idx[idx * 3 + 2]
                    trial_samples = raw_data.iloc[trial_start: trial_end].copy()

                    # experiment_id = file.split(".")[0].split("/")[-1]
                    # experiment_id = raw_data["Event value"].iloc[trial_end]
                    participant_id = raw_data["Participant name"].iloc[0]
                    stimuli_name = "{}.png".format(raw_data["Event value"].iloc[trial_start])
                    trial_id = str(stimuli_name).split()[0]
                    # stimuli_name = "0{}_{}.png".format(
                    #     idx + 1, raw_data.at[trial_start, "Event value"])
                    pupil_l = raw_data["Pupil diameter left"]
                    pupil_r = raw_data["Pupil diameter right"]

                    # Structure samples dataframe
                    trial_samples["eye_tracker"] = EYE_TRACKER
                    trial_samples["experiment_id"] = participant_id
                    trial_samples["participant_id"] = participant_id
                    trial_samples["filename"] = file
                    trial_samples["trial_id"] = str(trial_id)
                    trial_samples["stimuli_module"] = STIMULI_MODULE
                    trial_samples["stimuli_name"] = stimuli_name

                    samples = pd.concat([samples, trial_samples])

                    # Extract fixation data to create eye events dataframe
                    trial_eye_events = trial_samples[["Recording timestamp", "Gaze event duration",
                                                    #   "Gaze point X (MCSnorm)", "Gaze point Y (MCSnorm)",
                                                    "Fixation point X", "Fixation point Y",
                                                    "Eye movement type","Pupil diameter left",
                                                    "Pupil diameter right"]].copy()

                    trial_eye_events.rename(columns={
                        "Recording timestamp": "timestamp",
                        "Gaze event duration": "duration",
                        # "Gaze point X (MCSnorm)": "x0",
                        # "Gaze point Y (MCSnorm)": "y0",
                        "Fixation point X": "x0",
                        "Fixation point Y": "y0",
                        "Eye movement type": "eye_event_type",
                        "Pupil diameter left": "pupil_l",
                        "Pupil diameter right": "pupil_r"
                    }, inplace=True)

                    # Remove duplicate eye events
                    trial_eye_events.loc[
                        trial_eye_events["eye_event_type"].shift() !=
                        trial_eye_events["eye_event_type"]]

                    # Structure eye events dataframe
                    trial_eye_events["eye_tracker"] = EYE_TRACKER
                    trial_eye_events["experiment_id"] = participant_id
                    trial_eye_events["participant_id"] = participant_id
                    trial_eye_events["filename"] = file
                    trial_eye_events["trial_id"] = str(trial_id)
                    trial_eye_events["stimuli_module"] = STIMULI_MODULE
                    trial_eye_events["stimuli_name"] = stimuli_name
                    trial_eye_events["x1"] = np.nan
                    trial_eye_events["y1"] = np.nan
                    trial_eye_events["token"] = None
                    # TODO: Parse pupil data from the dataset
                    trial_eye_events["pupil_l"] = pupil_l
                    trial_eye_events["pupil_r"] = pupil_r
                    trial_eye_events["amplitude"] = np.nan
                    trial_eye_events["peak_velocity"] = np.nan
                    trial_eye_events["peak_velocity"] = np.nan
                    trial_eye_events = trial_eye_events[
                        trial_eye_events["eye_event_type"] == "Fixation"]
                    trial_eye_events["eye_event_type"] = "fixation"

                    eye_events = pd.concat(
                        [eye_events, trial_eye_events])
                    
                    # convert x0, y0 from normalized to pixel
                    # eye_events['x0'] = eye_events['x0'] * 1920
                    # eye_events['y0'] = eye_events['y0'] * 1036
                    
                    # convert x1, y1 from normalized to pixel
                    # eye_events['x1'] = eye_events['x1'] * 1920
                    # eye_events['y1'] = eye_events['y1'] * 1036
            
                # print the current file being processed
                print("Processed file:", file)
                
            except:
                print("Stimuli not found in the file:", file)
                continue
            
            # Stop parsing condition
            # sample_size -= 1
            # if sample_size == 0:
            #     break

    # Drop unnecessary information
    samples.drop(["Computer timestamp", "Sensor", "Eyetracker timestamp",
                  "Event", "Event value", "Gaze point X (MCSnorm)", "Gaze point Y (MCSnorm)",
                  "Gaze point left X (MCSnorm)", "Gaze point left Y (MCSnorm)",
                  "Gaze point right X (MCSnorm)", "Gaze point right Y (MCSnorm)",
                  "Fixation point X (MCSnorm)", "Fixation point Y (MCSnorm)",
                  "Fixation point X", "Fixation point Y"],
                 axis=1, inplace=True)

    # Rearrange columns
    eye_events = eye_events[get_eye_event_columns()]
    samples = pd.concat([samples.loc[:, "eye_tracker":],
                        samples.loc[:, : "Eye movement type index"]], axis=1)

    eye_events.reset_index(drop=True, inplace=True)
    samples.reset_index(drop=True, inplace=True)
    return eye_events, samples
