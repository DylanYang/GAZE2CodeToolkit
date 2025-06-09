from tqdm import tqdm
import pandas as pd
import os

# Initialize tqdm for pandas
tqdm.pandas()

# Export fixations to CSV
def export_aoi_fixations(aoi_df: pd.DataFrame, file_path: str, 
                         bytask: bool=False):
    """
    Export aoi fixations to CSV files.
    Parameters
    ----------
    eye_events : pd.DataFrame
        DataFrame containing eye events.
    file_path : str
        Path to save the CSV files.
    bytask : bool, optional (default False)
        Export fixations by task.
        Export all fixations.
        @ Developer Wudao(Dylan) Yang < data: 05-02-2025 >
    """
    if aoi_df.empty:
        print(f"Eye event dataframe is empty")
    else:
        # get trial_id from the dataframe
        trial_id = aoi_df['trial_id'].iloc[0]
        if bytask:
            # Create directory if it does not exist
            output_dir = file_path
            os.makedirs(f'./{output_dir}/group/aoi', exist_ok=True)
            # Export the sorted DataFrame to a CSV file
            output_path = f'./{output_dir}/aoi_fixations_{trial_id}.csv'
        elif not bytask:
            # get experiment_id from the dataframe
            participant_id = aoi_df['participant_id'].iloc[0]
            # Create directory if it does not exist
            output_dir = f'./{file_path}/individual/aoi/{trial_id}'
            os.makedirs(output_dir, exist_ok=True)
            # Export the sorted DataFrame to a CSV file
            output_path = f'./{output_dir}/aoi_fixations_{trial_id}_{participant_id}.csv'
        
        # Initialize tqdm for Pandas operations
        aoi_df.to_csv(output_path, header=True, index=False)
                
        print(f"Completed! AOI fixations saved to {output_path}")
    