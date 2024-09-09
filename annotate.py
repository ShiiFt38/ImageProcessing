import cv2
import numpy as np
import image_processing
import feature_extraction
from skimage.feature import graycomatrix, graycoprops, hog


'''
def extract_features(image):
    hog_features, hog_image = feature_extraction.extract_hog_features(image)
    color_histogram = feature_extraction.extract_color_histogram(image)
    texture_features = feature_extraction.extract_texture_features(image)
    return hog_features, color_histogram, texture_features

def alpha_blend_images(image_rgb, overlay_image, alpha=0.5):
    """Blend two images using alpha blending."""
    return cv2.addWeighted(image_rgb, alpha, overlay_image, 1 - alpha, 0)

def overlay_on_original(image_rgb, mask):
    """Overlay the mask on the original image."""
    # Convert mask to the same size as image_rgb
    mask_resized = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]))

    # Create an overlay image
    overlay = np.zeros_like(image_rgb)

    # Set the color for the overlay (e.g., red for the mask)
    overlay[mask_resized > 0] = [255, 255, 0]  # color for mask

    # Blend the original image with the overlay
    alpha = 0.5  # Transparency factor
    blended_image = cv2.addWeighted(image_rgb, 1 - alpha, overlay, alpha, 0)

    return blended_image


def overlay_hog_features(image_rgb, hog_image):
    if image_rgb.shape[0] != hog_image.shape[0] or image_rgb.shape[1] != hog_image.shape[1]:
        raise ValueError("Original image and HOG image dimensions do not match.")
    # Overlay HOG features
    hog_overlay = cv2.resize((hog_image * 255).astype(np.uint8), (image_rgb.shape[1], image_rgb.shape[0]))
    hog_overlay_colored = cv2.applyColorMap(hog_overlay, cv2.COLORMAP_JET)
    combined_image = alpha_blend_images(image_rgb, hog_overlay_colored, 0.3)
    return combined_image


def overlay_blobs_and_contours(image, binary_image, segmented_image):
    # Convert original image to RGB for visualization if it's not already
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # Find contours in the segmented image
    contours, _ = cv2.findContours(segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = image.copy()

    # Draw contours on the base image
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)  # Green contours

    # Overlay blobs
    blob_overlay = cv2.bitwise_and(image, image, mask=binary_image)

    # Combine contour and blob overlays
    final_overlay = cv2.addWeighted(contour_image, 0.7, blob_overlay, 0.3, 0)

    return final_overlay


def process_image(image_path):
    try:

        image = utils.load_image(image_path)

        # image_processing.basic_image_processing(image_path, output_format='png')

        basic_image_processing_image = image_processing.basic_image_processing(image_path)

        if image is None:
            print(f"Error loading image from {image_path}")
            return {'error': 'Image not loaded'}

        # Check image channels
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 1:  # Single-channel image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if basic_image_processing_image is None:
            raise ValueError("Basic image processing failed.")

        # Applying preprocessing
        preprocessed_image = pre_p.bilateral_filter(image)
        # preprocessed_image = pre_p.clahe(preprocessed_image)

        # Extract Features
        color_histograms = f_e.extract_color_histogram(image)
        texture_features = f_e.extract_texture_features(gray_image)
        hog_features, hog_image = f_e.extract_hog_features(gray_image)

        if hog_image is None:
            raise ValueError("HOG feature extraction failed.")

        segmented_image = f_e.segment_image(gray_image, method='fixed', fixed_threshold=0.5)

        if segmented_image is None or segmented_image.size == 0:
            raise ValueError("Segmentation failed or returned an empty image.")

        # Create a base image for overlay
        image = image.copy()

        # Create binary image from segmentation for blobs
        binary_image = segmented_image > 0.5

        # Detect blobs and contours
        blob_image, _ = f_e.detect_blobs(segmented_image)
        contour_image, _ = f_e.detect_contours(segmented_image)

        water_bodies, water_mask = f_e.segment_water_bodies(image)

        # Create overlay images
        hog_overlay = image_processing.overlay_hog_features(basic_image_processing_image, hog_image)

        if hog_overlay is None:
            raise ValueError("HOG overlay creation failed.")

        segmented_colored = cv2.applyColorMap(segmented_image * 255, cv2.COLORMAP_JET)

        if segmented_colored is None:
            raise ValueError("Segmented image coloring failed.")

        segmented_colored_resized = cv2.resize(segmented_colored, (image.shape[1], image.shape[0]))

        if segmented_colored_resized is None:
            raise ValueError("Resizing segmented image failed.")

        # Combination of overlays
        final_image = image_processing.overlay_hog_features(basic_image_processing_image, hog_image)
        final_blob_image = image_processing.overlay_blobs_and_contours(image, binary_image.astype(np.uint8) * 255, segmented_image)

        # Combine final images using alpha blending
        combined_image = image_processing.alpha_blend_images(final_image, final_blob_image, 0.5)

        if combined_image is None:
            raise ValueError("Combining images failed.")

        # Save the combined image in the same directory as the input image
        base_filename, ext = os.path.splitext(image_path)
        combined_image_filename = f"{base_filename}_combined_feature_extraction_overlay{ext}"
        cv2.imwrite(combined_image_filename, combined_image)

        print("Processing complete. Combined feature extraction image saved.")


    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return {
            'error': str(e)}'''
