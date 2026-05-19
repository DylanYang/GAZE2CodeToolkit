import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from scipy.stats import gaussian_kde
from skimage import exposure

def heatmap(eye_events: pd.DataFrame,
            stimuli_path: str,
            figsize: tuple[int, int] = (15, 10), color: str = 'inferno',
            alpha: float = .8, sigma_value: float = 10,
            x0_col: str = "x0", y0_col: str = "y0",
            duration_col: str = "duration",
            eye_event_type_col: str = "eye_event_type") -> None:
    """
    Draw a heatmap to show where the fixations focus on the stimuli image, displaying fixation duration, with Gaussian KDE smoothing.

    Parameters
    ----------
    eye_events : pd.DataFrame
        Pandas dataframe for eye events.

    stimuli_path : str
        Path to the stimuli image.

    figsize : tuple[int, int], optional (default (15, 10))
        Size of the plot.

    color : str, optional (default "inferno")
        Colormap of the heatmap.

    alpha : float, optional (default .8)
        Opacity level of heatmap.

    sigma_value : float, optional (default 10)
        Bandwidth for Gaussian KDE.

    x0_col : str, optional (default "x0")
        Column name for x-coordinates.

    y0_col : str, optional (default "y0")
        Column name for y-coordinates.

    duration_col : str, optional (default "duration")
        Column name for fixation durations.

    eye_event_type_col : str, optional (default "eye_event_type")
        Column name for the type of eye events.
    """
    # Drop missing values
    eye_events = eye_events.dropna(subset=[x0_col, y0_col, duration_col])
    
    # Filter fixation data
    fixations = eye_events.loc[eye_events[eye_event_type_col] == "fixation"]

    # Check if there are enough data points
    if fixations.empty:
        print("No fixation data available to generate the heatmap.")
        return

    # Extract x, y coordinates and fixation durations
    x = fixations[x0_col].values
    y = fixations[y0_col].values
    durations = fixations[duration_col].values
    # save x, y, durations to a csv file
    df = pd.DataFrame({'x': x, 'y': y, 'durations': durations})
    df.to_csv('../datasets/UNL2024/output/fixations.csv', index=False)
    

    # Load the background image and enhance brightness
    background_image = mpimg.imread(stimuli_path)
    background_image = exposure.rescale_intensity(background_image, in_range=(0, 255), out_range=(100, 255))

    img_height, img_width, _ = background_image.shape

    # Gaussian KDE
    values = np.vstack([x, y])
    kernel = gaussian_kde(values, weights=durations, bw_method=sigma_value / np.std(values))
    X, Y = np.meshgrid(np.linspace(0, img_width, 500), np.linspace(0, img_height, 500))
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel(positions).T, X.shape)

    # Enhance visibility of sparse data
    Z = Z ** 0.5  # Apply square root scaling

    # Plot
    plt.figure(figsize=figsize)

    # Display the brightened background image
    plt.imshow(background_image, extent=[0, img_width, 0, img_height], aspect='auto')

    # Overlay the heatmap
    plt.imshow(
        Z,
        extent=[0, img_width, 0, img_height],
        origin="lower",
        cmap=color,
        alpha=alpha
    )

    # Overlay raw fixation points
    plt.scatter(x, y, c='white', s=10, alpha=0.6, label="Fixation Points")

    # Add a colorbar
    cbar = plt.colorbar(orientation='horizontal', pad=0.05)
    cbar.set_label('Normalized Fixation Density (Gaussian KDE)', fontsize=10)

    # Customize plot
    plt.title("Fixation Heatmap", fontsize=14)
    plt.xlabel("X Coordinate", fontsize=12)
    plt.ylabel("Y Coordinate", fontsize=12)
    plt.grid(False)
    plt.legend(loc="upper right")

    # Display the plot
    plt.tight_layout()
    plt.show()
