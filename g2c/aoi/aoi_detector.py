import cv2
import pytesseract
import pandas as pd
import numpy as np
from pytesseract import Output
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

def preprocess_image(image, enhance_contrast=True, apply_morphology=True):
    """
    Optimized OCR preprocessing for enhanced text clarity and minimal distortion.
    
    Returns:
        binary (numpy.ndarray): Clean binary image (sharp text with white background).
        morphed (numpy.ndarray): Morphologically enhanced image (bold, refined text).
    """

    if isinstance(image, Image.Image):  # Convert PIL Image to NumPy array
        image = np.array(image)
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Use RGB to match PIL format
    else:
        gray = image.copy()
        
    # Step 1: Apply CLAHE for Advanced Contrast Enhancement
    if enhance_contrast:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # Step 2: Apply Adaptive Thresholding for Better Text Segmentation
    adaptive_binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 15, 8
    )

    # Step 3: Apply Otsu's Binarization on Top for Extra Refinement
    _, binary = cv2.threshold(adaptive_binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 4: Apply Minimal Morphological Processing for Clarity
    if apply_morphology:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))  # Small kernel to refine text
        morphed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  # Remove small noise while preserving characters
    else:
        morphed = binary.copy()  # If no morphology, return the binary as morphed

    # Step 5: Ensure White Background and Black Text (OCR prefers this)
    if np.mean(binary) > 127:  # If background is darker than text, invert it
        binary = cv2.bitwise_not(binary)
        morphed = cv2.bitwise_not(morphed)

    # Save debug images
    cv2.imwrite("./output/imgs/debug_gray.png", gray)
    cv2.imwrite("./output/imgs/debug_adaptive_binary.png", adaptive_binary)
    cv2.imwrite("./output/imgs/debug_final_binary.png", binary)
    cv2.imwrite("./output/imgs/debug_morphed.png", morphed)

    return binary, morphed  # ✅ Return both images

def aoi_detector(image_path, scale_factor=2, debug=False, 
                               use_preprocessing=False, min_confidence=60,
                               psm="6", oem="3"):
    """Detect text using Tesseract OCR with proper scaling adjustments."""

    # Load image using PIL
    image = Image.open(image_path)

    # Scale image for better OCR accuracy
    image_scaled = image.resize(
        (int(image.width * scale_factor), int(image.height * scale_factor)), 
        Image.Resampling.LANCZOS
    )
    
    # Convert PIL image to a NumPy array before preprocessing
    image_scaled_np = np.array(image_scaled)

    # Preprocess image if needed
    if use_preprocessing:
        binary_image, ocr_image = preprocess_image(image_scaled_np)  # ✅ Pass NumPy array
    else:
        ocr_image = image_scaled_np

    # # Convert PIL image to NumPy array before preprocessing
    # image_scaled_np = np.array(image_scaled)

    # # Preprocess image if needed
    # if use_preprocessing:
    #     binary_image, ocr_image = preprocess_image(image_scaled_np)  # ✅ Pass NumPy array
    # else:
    #     ocr_image = image_scaled

    # # Preprocess image if needed
    
    # if use_preprocessing:
    #     binary_image, ocr_image = preprocess_image(image_scaled)
    # else:
    #     ocr_image = image_scaled

    # Run Tesseract OCR on scaled image
    config = f"--psm {psm} --oem {oem}"
    ocr_data = pytesseract.image_to_data(ocr_image, output_type=Output.DICT, config=config)
    
    # get trial_id from the dataframe
    trial_id = str(image_path.split(".")[0].split("/")[-1]).split()[0]

    # Initialize list to store detected text and bounding boxes
    aoi_list = []
    current_line_y = None  # ✅ Initialize current_line_y
    line_count = 1  # ✅ Initialize line tracking
    part_count = 1  # ✅ Initialize part tracking

    # Process each detected word
    for i in range(len(ocr_data["text"])):
        text = ocr_data["text"][i].strip()
        if not text:
            continue

        try:
            # Extract bounding box data and convert to integers
            x = int(ocr_data["left"][i] / scale_factor)
            y = int(ocr_data["top"][i] / scale_factor)
            width = int(ocr_data["width"][i] / scale_factor)
            height = int(ocr_data["height"][i] / scale_factor)

            conf = int(float(ocr_data["conf"][i]))  # Convert confidence safely

            # Skip low-confidence words
            if conf < min_confidence:
                continue

            # Detect line breaks based on vertical spacing
            if current_line_y is None or abs(y - current_line_y) > height * 1.5:
                line_count += 1
                part_count = 1  # Reset part count for new line
                current_line_y = y

            # Append token details to AOI list
            token_name = f"line {line_count} part {part_count}"
            aoi_list.append(["sub-line", token_name, trial_id, x, y, width, height, text, image_path])
            part_count += 1  # Increment part count

        except ValueError:
            continue  # Skip entries with invalid confidence values

    # Convert results to a DataFrame
    aoi_df = pd.DataFrame(
        aoi_list, columns=["kind", "name", "trial_id", "x", "y", "width", "height", "token", "image"]
    )
    
    
    # Save detected tokens to CSV
    trial_name = trial_id  # You can adjust this if you want a different trial_name logic
    output_path = f"output/orc_detection/{trial_name}_detected_tokens.csv"
    aoi_df.to_csv(output_path, index=False)

    # Debugging: Visualize detected text on the original image
    if debug:
        if use_preprocessing:
            binary_image, morphed_image = preprocess_image(image_scaled)
            visualize_images_separately(image_scaled, binary_image, morphed_image)
            visualize_detected_tokens(image, aoi_df)
        else:
            visualize_detected_tokens(image, aoi_df)

    return aoi_df

def visualize_images_separately(image, binary_image, morphed_image):
    """Visualize the original, binary, and morphed images separately."""
    
    # print("Original Image Shape:", image.shape)
    # print("Binary Image Shape:", binary_image.shape)
    # print("Morphed Image Shape:", morphed_image.shape)

    # Display Original Image
    plt.figure(figsize=(10, 6))
    plt.imshow(image, cmap='gray')
    plt.title("Original Image", fontsize=16)
    plt.axis("off")
    plt.show()

    # Display Binary Image
    plt.figure(figsize=(10, 6))
    plt.imshow(binary_image, cmap='gray')
    plt.title("Binary Image", fontsize=16)
    plt.axis("off")
    plt.show()

    # Display Morphed Image
    plt.figure(figsize=(10, 6))
    plt.imshow(morphed_image, cmap='gray')
    plt.title("Morphed Image", fontsize=16)
    plt.axis("off")
    plt.show()

def visualize_detected_tokens(image, aoi_df, padding=5):
    """
    Visualize detected tokens on the image with padding and better alignment.
    """
    # Convert PIL image to NumPy array for OpenCV
    image_np = np.array(image)

    # Set up the plot
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))  # ✅ Fixed conversion error

    # Colors for visualization
    colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan"]

    # Draw bounding boxes with padding
    for _, row in aoi_df.iterrows():
        x, y, width, height, name, token = row["x"], row["y"], row["width"], row["height"], row["name"], row["token"]
        line_number = int(name.split(" ")[1])  # Extract line number
        color = colors[line_number % len(colors)]
        rect = Rectangle(
            (x - padding, y - padding), width + 2 * padding, height + 2 * padding,
            linewidth=1, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        # ax.text(x, y - 10, name, fontsize=7, color=color)#, bbox=dict(facecolor="white", alpha=0.7))
        # x ticks and y ticks font size
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        
    plt.show()