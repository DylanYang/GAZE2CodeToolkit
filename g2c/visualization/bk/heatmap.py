import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter
from emtk.util import _get_meta_data, _get_stimuli


def heatmap(eye_events: pd.DataFrame,
            figsize: tuple[int, int] = (15, 10), color: str = 'Reds',
            alpha: float = .6, sigma_value: float = 15,
            eye_tracker_col: str = "eye_tracker",
            x0_col: str = "x0", y0_col: str = "y0",
            duration_col: str = "duration",
            stimuli_module_col="stimuli_module",
            stimuli_name_col="stimuli_name", eye_event_type_col="eye_event_type") -> None:
    '''Draw a heatmap to show where the fixations focus on the stimuli image, displaying fixation duration, with Gaussian smoothing.

    Parameters
    ----------
    eye_events : pd.DataFrame
        Pandas dataframe for eye events.

    figsize : tuple[int], optional (default (15, 10))
        Size of the plot.

    color : str, optional (default "Reds")
        Color of the heatmap.

    alpha : float in [0, 1], optional (default .6)
        Opacity level of heatmap.

    sigma_value : float, optional (default 15)
        The standard deviation for Gaussian kernel. This controls the amount of smoothing applied.

    x0_col : str, optional (default "x0")
        Name of the column in the eye events dataframe that contains the x-coordinates of the eye events.

    y0_col : str, optional (default "y0")
        Name of the column in the eye events dataframe that contains the y-coordinates of the eye events.

    duration_col : str, optional (default "duration")
        Name of the column in the eye events dataframe that contains the duration of the fixations.

    stimuli_module_col : str, optional (default "stimuli_module")
        Name of the column in eye_events dataframe that contains the path to the stimuli module.

    stimuli_name_col : str, optional (default "stimuli_name")
        Name of the column in eye_events dataframe that contains the name of the stimuli.

    eye_event_type_col : str, optional (default "eye_event_type")
        Name of the column in the eye events dataframe that contains the types of the eye events.
    '''

    eye_tracker, stimuli_module, \
        stimuli_name = _get_meta_data(eye_events, eye_tracker_col,
                                      stimuli_module_col, stimuli_name_col)

    stimuli = _get_stimuli(stimuli_module, stimuli_name, eye_tracker)
    width, height = stimuli.size

    fixations = eye_events.loc[eye_events[eye_event_type_col] == "fixation"]
    x_cords = fixations[x0_col].astype(int)
    y_cords = fixations[y0_col].astype(int)
    durations = fixations[duration_col]

    # Initialize heatmap array
    heatmap_data = np.zeros((height, width))

    # Accumulate the duration for each (x0, y0) coordinate
    for x, y, duration in zip(x_cords, y_cords, durations):
        if 0 <= x < width and 0 <= y < height:
            heatmap_data[y, x] += duration

    # Apply Gaussian smoothing
    heatmap_data = gaussian_filter(heatmap_data, sigma=sigma_value)

    # Plot the smoothed heatmap
    _, ax = plt.subplots(figsize=figsize)
    ax.imshow(stimuli)
    sns.heatmap(heatmap_data, ax=ax, cmap=color, alpha=alpha)

    plt.show()

# Example usage:
heatmap(eye_events, figsize=(15, 10), color='Reds', alpha=0.6, sigma_value=15)
