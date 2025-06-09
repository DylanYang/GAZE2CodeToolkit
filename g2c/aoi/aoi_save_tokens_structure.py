import os
import pandas as pd
from tqdm import tqdm

def aoi_save_tokens_structure(aoi_result_df: pd.DataFrame, file_path: str):
    """
    Save AOI tokens to a CSV file.
    Parameters
    ----------
    aoi_result_df : pd.DataFrame
        DataFrame containing AOI tokens.
    trial_id : str
        Trial ID.
    file_path : str
        Path to save the CSV file.
        Developer: Wudao(Dylan) Yang < data: 05-02-2025 >
    """
    
    # get trial_id from the dataframe
    trial_id = aoi_result_df['trial_id'].iloc[0]
    
    # check if the file path exists
    if not file_path:
        raise ValueError("file_path cannot be None")
    output_dir = file_path
    os.makedirs(f'./{output_dir}', exist_ok=True)
    
    # Initialize tqdm progress bar
    # add tqdm progress bar
    with tqdm(total=len(aoi_result_df), desc="Saving AOI tokens structure", colour="green") as pbar:
        aoi_result_df.to_csv(f'./{output_dir}/aoi_{trial_id}_tokens_structure.csv', index=False, header=True)
        pbar.update(len(aoi_result_df))
    
    print(f"Completed! AOI tokens saved to {f'./{output_dir}/aoi_{trial_id}_tokens_structrue.csv'}")