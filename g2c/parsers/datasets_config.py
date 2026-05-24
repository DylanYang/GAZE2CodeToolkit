"""Dataset configurations for the unified Tobii parser.

Adding a new dataset that uses the same hardware (Tobii Pro Nano with the
Tobii I-VT fixation filter export) is a configuration change only — add
an entry here, no code edits required.
"""

# Tobii Pro Nano export column names are identical across UNL_UM and YMU_UM
# (same hardware, same Tobii Pro Lab export format). Keep this map in one place.
TOBII_PRO_COLUMNS = {
    "event_value": "Event value",
    "timestamp": "Recording timestamp",
    "duration": "Gaze event duration",
    "x": "Fixation point X",
    "y": "Fixation point Y",
    "event_type": "Eye movement type",
    "pupil_l": "Pupil diameter left",
    "pupil_r": "Pupil diameter right",
}

# Columns to drop from the samples dataframe at the end of parsing.
# Shared across the two Tobii datasets because they use the same export.
TOBII_PRO_SAMPLES_DROP_COLS = [
    "Computer timestamp", "Sensor", "Eyetracker timestamp",
    "Event", "Event value",
    "Gaze point X (MCSnorm)", "Gaze point Y (MCSnorm)",
    "Gaze point left X (MCSnorm)", "Gaze point left Y (MCSnorm)",
    "Gaze point right X (MCSnorm)", "Gaze point right Y (MCSnorm)",
    "Fixation point X (MCSnorm)", "Fixation point Y (MCSnorm)",
    "Fixation point X", "Fixation point Y",
]


DATASETS = {

    "UNL_UM": {
        "eye_tracker": "Tobii I-VT (Fixation)",
        "raw_dir": "datasets/UNL_UM24_30July/rawdata",
        "stimuli_dir": "datasets/UNL_UM24_30July/stimuli",
        "stimuli_names": (
            "Q1 (localhost)", "Q2A (localhost)", "Q2B (localhost)",
            "Q3 (localhost)", "Q4A (localhost)", "Q4B (localhost)",
            "Q5 (localhost)",
        ),
        "n_stimuli": 7,
        "columns": TOBII_PRO_COLUMNS,
        "participant_col": "Participant name",
        # UNL_UM marker rows: each stimulus appears 3 times (start / mid / end).
        # Trial span = [first marker, third marker].
        "trial_split": {"strategy": "paired_markers", "per_trial": 3},
        # Stimuli filename pattern, e.g. "Q1 (localhost).png".
        "stimuli_name_template": "{event_value}.png",
        # trial_id derivation: first whitespace token, e.g. "Q1".
        "trial_id_strategy": "first_word",
        "fixation_label": "Fixation",
        # UNL_UM raw files have AOI hit-test columns from index 99 onward.
        "drop_cols_from": 99,
        "samples_drop_cols": TOBII_PRO_SAMPLES_DROP_COLS,
        "default_sample_size": 44,
    },
    "YMU_UM": {
        "eye_tracker": "Tobii I-VT (Fixation)",
        "raw_dir": "datasets/YMU_UM/rawdata",
        "stimuli_dir": "datasets/YMU_UM/stimuli",
        "stimuli_names": (
            "Quiz - introduction-Q1 (localhost)",
            "Quiz - introduction-Q2A (localhost)",
            "Quiz - introduction-Q2B (localhost)",
            "Quiz - introduction-Q3 (localhost)",
            "Quiz - introduction-Q4A (localhost)",
            "Quiz - introduction-Q4B (localhost)",
            "Quiz - introduction-Q5 (localhost)",
        ),
        "n_stimuli": 7,
        "columns": TOBII_PRO_COLUMNS,
        "participant_col": "Participant name",
        # YMU_UM marker rows: each stimulus appears multiple times; the legacy
        # parser deduplicates and keeps only the first occurrence, then treats
        # the span up to the next stimulus marker as the trial.
        "trial_split": {"strategy": "dedup_sequential"},
        # Stimuli filename pattern, e.g. "introduction-Q1.png" — third
        # whitespace token of the marker value.
        "stimuli_name_template": "{third_word}.png",
        # trial_id derivation: third whitespace token, e.g. "introduction-Q1".
        "trial_id_strategy": "third_word",
        "fixation_label": "Fixation",
        "drop_cols_from": None,
        "samples_drop_cols": TOBII_PRO_SAMPLES_DROP_COLS,
        "default_sample_size": 84,
    },
}

DATASETS["UNL002"] = {
    "eye_tracker": "Tobii I-VT (Fixation)",
    "raw_dir": "datasets/UNL002/rawdata",
    "stimuli_dir": "datasets/UNL002/stimuli",
    "stimuli_names": (
        'Q1 (localhost)',
        'Q2A (localhost)',
        'Q2B (localhost)',
        'Q3 (localhost)',
        'Q4A (localhost)',
        'Q4B (localhost)',
        'Q5 (localhost)',
    ),
    "n_stimuli": 7,
    "columns": {**TOBII_PRO_COLUMNS, 'duration': 'Eye movement event duration'},
    "participant_col": "Participant name",
    "trial_split": {"strategy": "paired_markers", "per_trial": 3},
    "stimuli_name_template": "{event_value}.png",
    "trial_id_strategy": "first_word",
    "fixation_label": "Fixation",
    "drop_cols_from": 99,             # TODO verify — AOI block boundary
    "samples_drop_cols": TOBII_PRO_SAMPLES_DROP_COLS,
    "default_sample_size": 48,
}
