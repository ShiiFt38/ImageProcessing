import csv
import cv2
from matplotlib import pyplot as plt
from skimage.feature import graycomatrix, graycoprops, hog


def ensure_grayscale(image):
    """Ensure the image is in grayscale. Convert if not."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) != 2:
        raise ValueError("Input image is not valid grayscale or RGB.")
    return image


def extract_hog_features(gray_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    gray_image = ensure_grayscale(gray_image)
    try:
        features, hog_image = hog(gray_image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                                  cells_per_block=cells_per_block,  visualize=True, block_norm='L2-Hys')
        if hog_image is not None:
            print("HOG Image shape:", hog_image.shape)
        else:
            print("HOG Image: None")
        return features, hog_image
    except Exception as e:
        raise RuntimeError(f"Error extracting HOG features: {e}")


def extract_color_histogram(image):
    histograms = {}
    for channel, color_name in enumerate(['blue', 'green', 'red']):
        hist = cv2.calcHist([image], [channel], None, [256], [0, 256])
        histograms[color_name] = hist.flatten().tolist()
    return histograms


def calculate_texture_features(image):
    """Calculate texture features using Gray-Level Co-occurrence Matrix (GLCM)."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray_image, [1], [0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return {'contrast': contrast, 'dissimilarity': dissimilarity, 'homogeneity': homogeneity}


# Function to save metrics as CSV
def save_metrics_as_csv(image_path, color_histogram, texture_features):
    base_filename = image_path.split('.')[0]
    output_filename = f"{base_filename}_metrics.csv"

    # Open a CSV file to save data
    with open(output_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write headers
        writer.writerow(['Metric', 'Value'])

        # Write color histograms
        for color, hist in color_histogram.items():
            writer.writerow([f"{color} channel histogram", hist[:10]])

        # Write texture features
        for feature, value in texture_features.items():
            writer.writerow([feature, value])

    print(f"Metrics saved as CSV in {output_filename}")


def features_extract(image):
    gray_image = ensure_grayscale(image)
    hog_features, hog_image = extract_hog_features(gray_image)
    color = extract_color_histogram(image)
    texture = calculate_texture_features(image)

    return hog_image, {'color_histogram': color, 'texture': texture}
'''

def segment_image(gray_image, method='fixed', fixed_threshold=0.5):
    gray_image = ensure_grayscale(gray_image)
    if method == 'adaptive':
        segmented_image = cv2.adaptiveThreshold((gray_image * 255).astype(np.uint8), 255,
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    elif method == 'fixed':
        _, segmented_image = cv2.threshold(gray_image, int(fixed_threshold * 255), 255, cv2.THRESH_BINARY)
    else:
        raise ValueError("Invalid thresholding method. Choose 'fixed' or 'adaptive'.")
    return segmented_image

# Overlay with alpha blending
def overlay_with_alpha(image, overlay_image, alpha=0.3):
    return cv2.addWeighted(image, 1 - alpha, overlay_image, alpha, 0)


def extract_texture_features(gray_image):
    gray_image = ensure_grayscale(gray_image)
    gray_image_uint8 = (gray_image * 255).astype(np.uint8)
    glcm = graycomatrix(gray_image_uint8, [1], [0], symmetric=True, normed=True)

    texture_features = {
        # Extract GLCM properties
        'contrast': graycoprops(glcm, 'contrast')[0, 0],
        'dissimilarity': graycoprops(glcm, 'dissimilarity')[0, 0],
        'homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
        'energy': graycoprops(glcm, 'energy')[0, 0],
        'correlation': graycoprops(glcm, 'correlation')[0, 0]
    }
    return texture_features


def calculate_coverage(segmented_image):
    area = np.sum(segmented_image)/255
    return area / segmented_image.size


def detect_blobs(binary_image):
    if len(binary_image.shape) != 2:
        raise ValueError("Input image for blob detection should be a binary mask (single channel).")

    # Blob detection parameters
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 100
    params.filterByCircularity = True
    params.minCircularity = 0.1
    params.filterByConvexity = True
    params.minConvexity = 0.1
    params.filterByInertia = True
    params.minInertiaRatio = 0.1

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    # Detect blobs
    keypoints = detector.detect(binary_image)
    # Draw detected blobs on the image
    output_image = cv2.drawKeypoints(np.zeros_like(binary_image), keypoints, np.zeros((1, 1), dtype=np.uint8),
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return output_image, keypoints



def detect_contours(binary_image):
    if len(binary_image.shape) != 2:
        raise ValueError("Input image for contour detection should be a binary mask (single channel).")

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours on a blank image
    output_image = cv2.drawContours(np.zeros_like(binary_image), contours, -1, (0, 255, 0), 2)
    return output_image, contours


def segment_water_bodies(image):
    # Define thresholds for water body detection
    lower_blue = np.array([0, 0, 100])
    upper_blue = np.array([100, 100, 255])

    # Create a binary mask for water bodies
    water_mask = cv2.inRange(image, lower_blue, upper_blue)
    # Apply the mask to the image to get water bodies
    water_bodies = cv2.bitwise_and(image, image, mask=water_mask)
    return water_bodies, water_mask


def detect_water_hyacinth(image):
    # Segment the image for water hyacinth detection
    segmented_image = segment_image(image)

    # Detect blobs and contours in the segmented image
    blob_image, keypoints = detect_blobs(segmented_image)
    contour_image, contours = detect_contours(segmented_image)

    return blob_image, contour_image'''











