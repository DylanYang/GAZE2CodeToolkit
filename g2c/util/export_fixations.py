from tqdm import tqdm
import pandas as pd
import os

# Initialize tqdm for pandas
tqdm.pandas()

# Export fixations to CSV
def export_fixations(eye_events: pd.DataFrame, samples: pd.DataFrame, 
                     experiment_range: range, trial_range: range, 
                     file_path: str, bytask: bool=False, byall: bool=False):
    """
    Export fixations to CSV files.
    Parameters
    ----------
    eye_events : pd.DataFrame 
        DataFrame containing eye events.
    samples : pd.DataFrame
        DataFrame containing samples.
    experiment_range : range
        Range of experiment IDs.
    trial_range : range
        Range of trial IDs.
    file_path : str 
        Path to save the CSV files.
    bytask : bool, optional (default False)
        Export fixations by task.
    byall : bool, optional (default False)
        Export all fixations.
        @ Developer Wudao(Dylan) Yang < data: 05-02-2025 >
    """
    if byall:
        if eye_events.empty:
            print(f"Eye event dataframe is empty")
        else:
            # # with tqdm(total=len(eye_events), desc="Processing eye events", colour="green") as pbar:
            # #     fixations = eye_events.loc[eye_events['eye_event_type'] == 'fixation']
                
            #     # Create directory if it does not exist
            #     output_dir = file_path
            #     os.makedirs(f'./{output_dir}', exist_ok=True)
                
            #     # Export the sorted DataFrame to a CSV file
            #     output_path = f'./{output_dir}/fixations_all.csv'
            #     with tqdm(total=len(eye_events), desc="Processing eye events", colour="green") as pbar:
            #         with open(output_path, 'w') as f:
            #             for _, row in eye_events.iterrows():
            #                 if row['eye_event_type'] == 'fixation':
            #                     row.to_csv(f, header=False, index=False)
            #                 pbar.update(1)
            # Create directory if it does not exist
            # Create directory if it does not exist
            output_dir = file_path  # Ensure file_path is correctly set
            os.makedirs(output_dir, exist_ok=True)

            # Define output path
            output_path = os.path.join(output_dir, 'fixations_all.csv')

            # Initialize tqdm for Pandas operations
            tqdm.pandas(desc="Processing eye events")

            # Filter fixation events
            fixations = eye_events[eye_events['eye_event_type'] == 'fixation']

            # Apply tqdm progress bar to row processing
            fixations.progress_apply(lambda row: None, axis=1)  # Just to visualize progress

            # Export all fixations in one go
            fixations.to_csv(output_path, index=False)

            print("Export completed successfully.")  # Print success message
    elif not byall:
        if bytask:
            for trial_id in tqdm(trial_range, desc="Processing trial_id", colour="green"):
                trial_data = eye_events.loc[eye_events['trial_id'] == trial_id]
                # samples_data = samples.loc[samples['experiment_id'] == experiment_id]
                
                if trial_data.empty:
                    print(f"Eye event dataframe is empty for trial_id {trial_id}")
                else:
                    fixations = trial_data.loc[trial_data['eye_event_type'] == 'fixation']
                    
                    # Create directory if it does not exist
                    output_dir = file_path
                    os.makedirs(f'./{output_dir}', exist_ok=True)
                    
                    # Export the sorted DataFrame to a CSV file
                    output_path = f'./{output_dir}/fixations_{trial_id}.csv'
                    fixations.to_csv(output_path, index=False, header=True)
            print("completed")
        elif not bytask:
            # Iterate over experiment_range
            for experiment_id in tqdm(experiment_range, desc="Processing experiment_id", colour="green"):
                for trial_id in trial_range:
                    trial_data = eye_events.loc[(eye_events['experiment_id'] == experiment_id) & 
                                                (eye_events['trial_id'] == trial_id)]
                    
                    samples_data = samples.loc[(samples['experiment_id'] == experiment_id) & 
                                                (samples['trial_id'] == trial_id)]
                    
                    if trial_data.empty:
                        print(f"Eye event dataframe is empty for experiment_id {experiment_id} and trial_id {trial_id}")
                        continue
                    
                    fixations = trial_data.loc[trial_data['eye_event_type'] == 'fixation']
                    
                    # Create directory if it does not exist
                    output_dir = file_path
                    os.makedirs(f'./{output_dir}/{trial_id}', exist_ok=True)
                    
                    # Export the sorted DataFrame to a CSV file
                    output_path = f'./{output_dir}/{trial_id}/fixations_{trial_id}_{experiment_id}.csv'
                    fixations.to_csv(output_path, index=False, header=True)
            print("completed")
