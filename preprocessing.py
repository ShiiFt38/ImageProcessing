import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import utils
from feature_extraction import calculate_texture_features


def process(image):
    resize_image = cv2.resize(image, (512, 512))
    return resize_image


# Noise Reduction
# Preserves edges while reducing noise
def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


# Enhances local contrast
def enhance_contrast_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    if len(image.shape) == 3:  # Check if image is colored
        yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv_image[..., 0] = clahe.apply(yuv_image[..., 0])
        clahe_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
    else:
        # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_image = clahe.apply(image)

    return clahe_image


def calculate_exg(image_rgb):
    """Calculate Excess Green Index (ExG)."""
    red_channel = image_rgb[:, :, 0].astype(float)
    green_channel = image_rgb[:, :, 1].astype(float)
    blue_channel = image_rgb[:, :, 2].astype(float)
    exg = 2 * green_channel - red_channel - blue_channel
    return exg


def fine_tune_exg(exg_image, lower_threshold=0.4, upper_threshold=1.0):
    """Fine-tune the ExG threshold to improve detection."""
    normalized_exg = cv2.normalize(exg_image, None, 0, 255, cv2.NORM_MINMAX)
    normalized_exg = normalized_exg.astype(np.uint8)
    _, binary_mask = cv2.threshold(normalized_exg, lower_threshold * 255, upper_threshold * 255, cv2.THRESH_BINARY)
    return binary_mask


def combine_exg_and_texture(exg_mask, image):
    texture_features = calculate_texture_features(image)
    """Combine ExG mask with texture features to refine detection."""
    if texture_features['contrast'] > 0.1 > texture_features['dissimilarity']:
        refined_mask = exg_mask
    else:
        refined_mask = np.zeros_like(exg_mask)
    return refined_mask


def adaptive_threshold(image):
    """Apply adaptive thresholding to the image."""
    image = (image * 255).astype(np.uint8)
    _, thresh = cv2.threshold(image, 0.4, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def canny_edge_detection(smoothed_mask):
    """Tweak parameters for Canny edge detection."""
    return cv2.Canny((smoothed_mask * 255).astype(np.uint8), 30, 100)


def morphological_operations(mask):
    """Apply dilation and erosion to clean up the mask."""
    kernel = np.ones((2, 2), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    cleaned_mask = cv2.erode(dilated_mask, kernel, iterations=1)
    return cleaned_mask


def process_sentinel_images(image_path):
    try:
        # Load the image using the utils module
        image_rgb = utils.load_image(image_path)
        image_size = process(image_rgb)

        if len(image_rgb.shape) != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Image at {image_path} does not have 3 channels (RGB).")

        # Enhance contrast for each channel
        image_eq = np.zeros_like(image_rgb)
        for i in range(3):
            image_eq[:, :, i] = enhance_contrast_clahe(image_rgb[:, :, i])

        # Step 1: Calculate Excess Green Index (ExG)
        exg = calculate_exg(image_eq)

        # Step 2: Fine-tune ExG
        fine_tuned_mask = fine_tune_exg(exg)

        # Combine ExG with additional features
        refined_mask = combine_exg_and_texture(fine_tuned_mask, image_eq)

        # Step 3: Apply Adaptive Thresholding
        mask = adaptive_threshold(refined_mask)

        # Step 4: Flatten the image for K-Means classification
        flat_image = image_eq.reshape(-1, 3)
        if flat_image is None or flat_image.size == 0:
            raise ValueError("Image data is empty or could not be flattened properly.")

        kmeans = KMeans(n_clusters=3, random_state=0).fit(flat_image)
        classified_image = kmeans.labels_.reshape(image_eq.shape[:2])

        # Step 5: Apply Bilateral Filtering (Advanced Smoothing)
        smoothed_mask = bilateral_filter(mask)

        # Step 6: Apply Canny Edge Detection
        edges = canny_edge_detection(smoothed_mask)

        # Step 7: Apply Morphological Operations
        final_mask = morphological_operations(smoothed_mask)

        return image_size, exg, fine_tuned_mask, refined_mask, mask, classified_image, smoothed_mask, edges, final_mask

    except Exception as e:
        print(f"Error during image processing: {e}")
        raise
