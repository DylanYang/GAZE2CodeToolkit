###color map###
"""
These colormaps are ideal for data that has a natural progression from low to high values.

'viridis': A perceptually uniform colormap that is often used for heatmaps.
'plasma': A sequential colormap with a range from yellow to purple.
'inferno': A colormap that ranges from yellow to dark purple/black.
'magma': Similar to inferno, but with a slightly different color progression.
'cividis': A colorblind-friendly colormap with a blue-to-yellow progression.
'Blues': Sequential colormap in shades of blue.
'Greens': Sequential colormap in shades of green.
'Reds': Sequential colormap in shades of red.
'Purples': Sequential colormap in shades of purple.
'Oranges': Sequential colormap in shades of orange.
Diverging Colormaps
These are useful when you want to emphasize deviation from a center point.

'coolwarm': A popular diverging colormap with a blue to red gradient.
'bwr': Blue-white-red colormap, great for showing positive and negative deviations.
'PiYG': A pink-green colormap useful for showing two extremes.
'PRGn': Purple-green colormap, another option for showing deviations.
'RdBu': Red-blue colormap, good for data that diverges around zero.
'RdYlBu': Red-yellow-blue colormap, often used in climate data visualizations.
'Spectral': A rainbow-like colormap suitable for showing diverse ranges.
Cyclic Colormaps
These colormaps are useful for data that is cyclical, like phases or directions.

'twilight': A smooth cyclic colormap that goes from purple to green.
'twilight_shifted': Similar to 'twilight' but with a different phase.
'hsv': The hue-saturation-value colormap, useful for cyclical data.
Qualitative Colormaps
These are used for categorical data.

'tab10': A ten-color palette suitable for categorical data.
'tab20': A twenty-color palette, an extension of 'tab10'.
'Set1': A set of distinct colors ideal for different categories.
'Set2': A more pastel-like palette.
'Paired': A paired palette with colors that go well together.
'Accent': A palette with distinct colors, good for categorical data.
These colormaps are ideal for data that has a natural progression from low to high values.

'viridis': A perceptually uniform colormap that is often used for heatmaps.
'plasma': A sequential colormap with a range from yellow to purple.
'inferno': A colormap that ranges from yellow to dark purple/black.
'magma': Similar to inferno, but with a slightly different color progression.
'cividis': A colorblind-friendly colormap with a blue-to-yellow progression.
'Blues': Sequential colormap in shades of blue.
'Greens': Sequential colormap in shades of green.
'Reds': Sequential colormap in shades of red.
'Purples': Sequential colormap in shades of purple.
'Oranges': Sequential colormap in shades of orange.
Diverging Colormaps
These are useful when you want to emphasize deviation from a center point.

'coolwarm': A popular diverging colormap with a blue to red gradient.
'bwr': Blue-white-red colormap, great for showing positive and negative deviations.
'PiYG': A pink-green colormap useful for showing two extremes.
'PRGn': Purple-green colormap, another option for showing deviations.
'RdBu': Red-blue colormap, good for data that diverges around zero.
'RdYlBu': Red-yellow-blue colormap, often used in climate data visualizations.
'Spectral': A rainbow-like colormap suitable for showing diverse ranges.
Cyclic Colormaps
These colormaps are useful for data that is cyclical, like phases or directions.

'twilight': A smooth cyclic colormap that goes from purple to green.
'twilight_shifted': Similar to 'twilight' but with a different phase.
'hsv': The hue-saturation-value colormap, useful for cyclical data.
Qualitative Colormaps
These are used for categorical data.

'tab10': A ten-color palette suitable for categorical data.
'tab20': A twenty-color palette, an extension of 'tab10'.
'Set1': A set of distinct colors ideal for different categories.
'Set2': A more pastel-like palette.
'Paired': A paired palette with colors that go well together.
'Accent': A palette with distinct colors, good for categorical data.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
from scipy.ndimage import gaussian_filter
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from g2c.util import sample_offset

# Assuming _get_meta_data and _get_stimuli functions are already defined

def _get_meta_data(eye_events: pd.DataFrame,
                   eye_tracker_col: str = "eye_tracker",
                   stimuli_module_col: str = "stimuli_module",
                   stimuli_name_col: str = "stimuli_name") -> tuple:
    '''Retrieve name of eye tracker, path to stimuli folder of the experiment,
    and name of stimuli from dataframe of eye events.'''

    col_names = [eye_tracker_col, stimuli_module_col, stimuli_name_col]

    for col in col_names:
        if len(eye_events[col].unique()) > 1:
            raise Exception(f"Error, there are more than one unique value in {col} column")

    return tuple(eye_events[col].unique()[0] for col in col_names)

def _get_stimuli(stimuli_module: str, stimuli_name: str, eye_tracker: str) -> Image:
    '''Retrieve stimuli image.'''

    stimuli = Image.open(os.path.join(stimuli_module, stimuli_name))

    # Preprocessing example for Tobii I-VT (Fixation)
    if eye_tracker == "Tobii I-VT (Fixation)":
        background = Image.new('RGB', (1930, 1036), color='black')
        background.paste(stimuli, (0, 0), stimuli.convert('RGBA'))
        return background.copy()

    return stimuli

def __heatmap_without_contours(eye_events: pd.DataFrame,
            figsize: tuple[int, int] = (12, 6),
            alpha: float = .7,  # Adjustable transparency
            thresh: float = .5,
            eye_tracker_col: str = "eye_tracker",
            x0_col: str = "x0", y0_col: str = "y0",
            duration_col: str = "duration",  # Column name for duration
            stimuli_module_col="stimuli_module",
            stimuli_name_col="stimuli_name", 
            eye_event_type_col="eye_event_type",
            colormap: str = "RdYlGn_r", 
            sigma_value: float = 15,  # Smoothing factor
            vmin: float = None, vmax: float = None) -> None:
    '''Draw a heatmap to show where the fixations focus on the stimuli image with Gaussian smoothing and intensity control, including duration.'''

    eye_tracker, stimuli_module, stimuli_name = _get_meta_data(
        eye_events, eye_tracker_col, stimuli_module_col, stimuli_name_col
    )

    # Load the stimuli image to get its dimensions
    stimuli = _get_stimuli(stimuli_module, stimuli_name, eye_tracker)
    width, height = stimuli.size

    # Filter for fixation events
    fixations = eye_events.loc[eye_events[eye_event_type_col] == "fixation"]
    x_cords = fixations[x0_col].dropna()
    y_cords = fixations[y0_col].dropna()
    durations = fixations[duration_col].dropna()

    # Check if x_cords and y_cords are non-empty
    if x_cords.empty or y_cords.empty:
        print("No valid data points available for heatmap generation.")
        return

    # Create an empty heatmap
    heatmap = np.zeros((height, width))

    # Accumulate the heatmap based on duration
    for x, y, duration in zip(x_cords, y_cords, durations):
        if 0 <= x < width and 0 <= y < height:
            heatmap[int(y), int(x)] += duration

    # Apply Gaussian smoothing with adjusted sigma value
    heatmap = gaussian_filter(heatmap, sigma=sigma_value)

    # Plot the heatmap with the correct extent
    extent = [0, width, height, 0]  # Maintain the correct orientation
    _, ax = plt.subplots(figsize=figsize)

    ax.imshow(stimuli, extent=extent)
    # ax.imshow(heatmap, extent=extent, cmap=colormap, alpha=alpha, vmin=vmin, vmax=vmax)
    img = ax.imshow(heatmap, extent=extent, cmap=colormap, alpha=alpha, vmin=vmin, vmax=vmax)
    
    # Define X, Y, Z for contour plotting
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    Z = heatmap
    
    # fontsize=25
    ax.set_title("Attention Heatmap of Fixations", fontsize=25)
    # Set the font size for the ticks
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.set_xlabel("x0", fontsize=25)
    ax.set_ylabel("y0", fontsize=25)
    
    # Add colorbar with adjusted fontsize
    cbar = plt.colorbar(img, ax=ax, orientation='horizontal', 
                    label="Intensity", pad=0.11, aspect=40, shrink=0.7)

    # Adjust tick labels and label fontsize
    cbar.ax.tick_params(labelsize=25)  # Set tick label fontsize
    cbar.set_label("Intensity", fontsize=25)  # Set label fontsize

    plt.show()

def __heatmap_with_contours(eye_events: pd.DataFrame,
            figsize: tuple[int, int] = (12, 6),
            alpha: float = .7,  # Adjustable transparency
            thresh: float = .5,
            eye_tracker_col: str = "eye_tracker",
            x0_col: str = "x0", y0_col: str = "y0",
            duration_col: str = "duration",  # Column name for duration
            stimuli_module_col="stimuli_module",
            stimuli_name_col="stimuli_name", eye_event_type_col="eye_event_type",
            colormap: str = "RdYlGn_r", 
            sigma_value: float = 15,  # Smoothing factor
            vmin: float = None, vmax: float = None) -> None:
    '''Draw a heatmap to show where the fixations focus on the stimuli image with Gaussian smoothing and intensity control, including duration.
       Overlay the heatmap with contour lines for better visualization of intensity levels.'''

    eye_tracker, stimuli_module, stimuli_name = _get_meta_data(
        eye_events, eye_tracker_col, stimuli_module_col, stimuli_name_col
    )

    # Load the stimuli image to get its dimensions
    stimuli = _get_stimuli(stimuli_module, stimuli_name, eye_tracker)
    width, height = stimuli.size

    # Filter for fixation events
    fixations = eye_events.loc[eye_events[eye_event_type_col] == "fixation"]
    x_cords = fixations[x0_col].dropna()
    y_cords = fixations[y0_col].dropna()
    durations = fixations[duration_col].dropna()

    # Check if x_cords and y_cords are non-empty
    if x_cords.empty or y_cords.empty:
        print("No valid data points available for heatmap generation.")
        return

    # Create an empty heatmap
    heatmap = np.zeros((height, width))

    # Accumulate the heatmap based on duration
    for x, y, duration in zip(x_cords, y_cords, durations):
        if 0 <= x < width and 0 <= y < height:
            heatmap[int(y), int(x)] += duration

    # Apply Gaussian smoothing with adjusted sigma value
    heatmap = gaussian_filter(heatmap, sigma=sigma_value)

    # Plot the heatmap with the correct extent
    extent = [0, width, height, 0]  # Maintain the correct orientation
    _, ax = plt.subplots(figsize=figsize)

    ax.imshow(stimuli, extent=extent)
    img = ax.imshow(heatmap, extent=extent, cmap=colormap, alpha=alpha, vmin=vmin, vmax=vmax)
    grid_resolution = max(width, height) // 2  # Dynamic resolution scaling

    # Add contour lines
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    Z = heatmap
    contour = ax.contour(X, Y, Z, colors='gray', linewidths=2, alpha=1) 
    
    # Add labels to the contour lines
    ax.clabel(contour, inline=True, fontsize=10, fmt="%.0f", colors='black')

    # Add annotations for high-density regions
    peak_indices = np.unravel_index(np.argsort(Z.ravel())[-5:], Z.shape)
    for x_peak, y_peak in zip(X[peak_indices], Y[peak_indices]):
        plt.text(
            x_peak, y_peak, "★", color="red", fontsize=12, ha="center", va="center"
            
    )


    # Adjust plot title and labels
    ax.set_title("Attention Heatmap of Fixations With Contours", fontsize=25)
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.set_xlabel("x0", fontsize=25)
    ax.set_ylabel("y0", fontsize=25)

    # Add colorbar with adjusted fontsize
    cbar = plt.colorbar(img, ax=ax, orientation='horizontal', 
                    label="Intensity", pad=0.11, aspect=40, shrink=0.7)

    # Adjust tick labels and label fontsize
    cbar.ax.tick_params(labelsize=25)  # Set tick label fontsize
    cbar.set_label("Intensity", fontsize=25)  # Set label fontsize

    plt.show()


# Example usage with duration
# You can modify `alpha`, `sigma_value`, and `duration_col` according to your dataset
# heatmap(eye_events, figsize=(12.8, 7.2), alpha=0.5, sigma_value=15, colormap="RdYlGn_r", vmin=0, vmax=3)

# get customed color map
def _get_custom_colormap():
    # Define a custom colormap
    colors = [
              (17/255, 18/255, 85/255),  # White
              (0, 1, 0),  # Green
              (1, 1, 0),  # Yellow
              (1, 0, 0)]  # Red
                    

    n_bins = 100  # Discretize the colormap into 100 bins
    cmap_name = 'custom_red_yellow_green_white'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    return custom_cmap

# Example usage in your heatmap function
# sns.heatmap(heatmap_data, ax=ax, cmap=custom_cmap, alpha=alpha)
# 
def draw_heatmap(trail_data: pd.DataFrame,
            contours: bool = False,
            figsize: tuple[int, int] = (12, 6),
            alpha: float = .7,  # Adjustable transparency
            thresh: float = .5,
            eye_tracker_col: str = "eye_tracker",
            x0_col: str = "x0", y0_col: str = "y0",
            duration_col: str = "duration",  # Column name for duration
            stimuli_module_col="stimuli_module",
            stimuli_name_col="stimuli_name", eye_event_type_col="eye_event_type",
            colormap: str = "RdYlGn_r", 
            sigma_value: float = 15,  # Smoothing factor
            vmin: float = None, vmax: float = None) -> None:
    
    # get the colormap
    custom_cmap = _get_custom_colormap()
    
    # Filter for fixation events
    trail_data = trail_data[['eye_tracker', 'stimuli_module', 'stimuli_name', 'x0', 'y0', 'duration', 'eye_event_type']]

    # Create a copy of the data with an offset
    trail_data_copy = sample_offset(trail_data, 0, -40)
    
    # Plot the heatmap with the custom colormap
    if contours:
        __heatmap_with_contours(trail_data_copy, figsize=figsize,
                                   alpha=alpha, sigma_value=sigma_value, colormap=custom_cmap,
                                   vmin=vmin, vmax=vmax)
    else:
        __heatmap_without_contours(trail_data_copy, figsize=figsize,
                                   alpha=alpha, sigma_value=sigma_value, colormap=custom_cmap,
                                   vmin=vmin, vmax=vmax)
