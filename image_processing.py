import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.utils import path
from sklearn.cluster import KMeans
import utils
from feature_extraction import calculate_texture_features
from preprocessing import enhance_contrast_clahe


'''



def calculate_exg(image_rgb):
    """Calculate Excess Green Index (ExG)."""
    red_channel = image_rgb[:, :, 0].astype(float)
    green_channel = image_rgb[:, :, 1].astype(float)
    blue_channel = image_rgb[:, :, 2].astype(float)
    exg = 2 * green_channel - red_channel - blue_channel
    return exg


def fine_tune_exg(exg_image, lower_threshold=0.3, upper_threshold=1.0):
    """Fine-tune the ExG threshold to improve detection."""
    normalized_exg = cv2.normalize(exg_image, None, 0, 255, cv2.NORM_MINMAX)
    normalized_exg = normalized_exg.astype(np.uint8)
    _, binary_mask = cv2.threshold(normalized_exg, lower_threshold * 255, upper_threshold * 255, cv2.THRESH_BINARY)
    return binary_mask


def adaptive_threshold(image):
    """Apply adaptive thresholding to the image."""
    image = (image * 255).astype(np.uint8)
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def bilateral_filtering(mask):
    """Apply Bilateral Filtering for advanced smoothing."""
    return cv2.bilateralFilter(mask.astype(np.float32), d=5, sigmaColor=50, sigmaSpace=50)


def canny_edge_detection(smoothed_mask):
    """Tweak parameters for Canny edge detection."""
    return cv2.Canny((smoothed_mask * 255).astype(np.uint8), 30, 100)


def morphological_operations(mask):
    """Apply dilation and erosion to clean up the mask."""
    kernel = np.ones((2, 2), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    cleaned_mask = cv2.erode(dilated_mask, kernel, iterations=1)
    return cleaned_mask


def apply_clahe(image_rgb):
    return enhance_contrast_clahe(image_rgb)


def combine_exg_and_texture(exg_mask, image):
    """Combine ExG mask with texture features to refine detection."""
    contrast, dissimilarity, homogeneity = calculate_texture_features(image)
    if contrast > 0.1 > dissimilarity:
        refined_mask = exg_mask
    else:
        refined_mask = np.zeros_like(exg_mask)
    return refined_mask


def basic_image_processing(image_path, output_format='png'):
    try:
        # Load the image using the utils module
        image_rgb = utils.load_image(image_path)

        if len(image_rgb.shape) != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Image at {image_path} does not have 3 channels (RGB).")

        # Enhance contrast for each channel
        image_eq = np.zeros_like(image_rgb)
        for i in range(3):
            image_eq[:, :, i] = apply_clahe(image_rgb[:, :, i])

        # Step 1: Calculate Excess Green Index (ExG)
        exg = calculate_exg(image_eq)
        visualize_image(exg, title='Excess Green Index')

        # Step 2: Fine-tune ExG
        fine_tuned_mask = fine_tune_exg(exg)
        visualize_image(fine_tuned_mask, title='Fine-Tuned ExG Mask')

        # Combine ExG with additional features
        refined_mask = combine_exg_and_texture(fine_tuned_mask, image_eq)
        visualize_image(refined_mask, title='Refined Mask')

        # Step 3: Apply Adaptive Thresholding
        mask = adaptive_threshold(refined_mask)
        visualize_image(mask, title='Vegetation Mask')

        # Step 4: Flatten the image for K-Means classification
        flat_image = image_eq.reshape(-1, 3)
        if flat_image is None or flat_image.size == 0:
            raise ValueError("Image data is empty or could not be flattened properly.")

        kmeans = KMeans(n_clusters=3, random_state=0).fit(flat_image)
        classified_image = kmeans.labels_.reshape(image_eq.shape[:2])
        visualize_image(classified_image, title='Classified Image')

        # Step 5: Apply Bilateral Filtering (Advanced Smoothing)
        smoothed_mask = bilateral_filtering(mask)
        visualize_image(smoothed_mask, title='Smoothed Mask')

        # Step 6: Apply Canny Edge Detection
        edges = canny_edge_detection(smoothed_mask)
        visualize_image(edges, title='Edges')

        # Step 7: Apply Morphological Operations
        final_mask = morphological_operations(smoothed_mask)
        visualize_image(final_mask, title='Final Mask')

        # Step 7: Overlay the classification results
        classified_colored = cv2.applyColorMap((classified_image * 85).astype(np.uint8), cv2.COLORMAP_JET)
        classified_colored = cv2.resize(classified_colored, (image_rgb.shape[1], image_rgb.shape[0]))

        final_mask_rgb = cv2.cvtColor((final_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        final_mask_rgb = cv2.resize(final_mask_rgb, (image_rgb.shape[1], image_rgb.shape[0]))

        # Overlay the classified image and final mask on the original image
        overlay_image = cv2.addWeighted(image_rgb, 0.5, classified_colored, 0.5, 0)
        final_overlay = cv2.addWeighted(overlay_image, 0.7, final_mask_rgb, 0.3, 0)

        # Step 8: Save the processed image
        output_filename = f"{image_path.rsplit('.', 1)[0]}_final.{output_format}"
        cv2.imwrite(output_filename, final_overlay)
        print(f"Processed image saved as {output_filename}")

    except Exception as e:
        print(f"Error during image processing: {e}")'''


