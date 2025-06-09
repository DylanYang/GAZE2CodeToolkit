# read data
import os
import pandas as pd
from tqdm import tqdm

def aoi_tokens_matcher(aoi_tokens_struc_file_path: str, trial_data: pd.DataFrame, 
                       trial_id: str, redius: int = 25) -> pd.DataFrame:
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
        Developer: Wudao(Dylan) Yang < data: 05-02-2025 >
    """
    
    # read the AOI tokens structure file
    aoi_tokens_struc = pd.read_csv(f'./{aoi_tokens_struc_file_path}')
    
    # get aoi_data from __match_fixations_to_aois
    aoi_data = __match_fixations_to_aois(trial_data, aoi_tokens_struc, radius=35)
    
    return aoi_data

def __match_fixations_to_aois(fixations: pd.DataFrame, aois: pd.DataFrame, radius: int = 25) -> pd.DataFrame:
    """
    Match fixations to AOIs and calculate fixation durations.

    Parameters:
    -----------
    fixations : pd.DataFrame
        DataFrame containing fixation data with coordinates and other attributes.
    aois : pd.DataFrame
        DataFrame containing AOI data with bounding box details.
    radius : int, optional (default=25)
        Radius around the AOI to include in its region.

    Returns:
    --------
    pd.DataFrame
        DataFrame matching fixations to AOIs with the calculated fixation durations.
        Developer: Wudao(Dylan) Yang < data: 05-02-2025 >
    """

    # Ensure 'x', 'y' columns in AOIs are present for matching fixations
    output_data = []

    # Loop through each fixation and check for AOI matches with a progress bar
    for _, fixation in tqdm(fixations.iterrows(), total=fixations.shape[0], 
                            desc="Matching fixations to AOIs", unit="fixation", 
                            colour="green"):
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