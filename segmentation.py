'''import cv2
import numpy as np


def segment_image(image):
    # Ensure the image is RGB
    image = ensure_rgb(image)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define color range for water
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])


    # Segmentation
    # Thresholding
    def adaptive_threshold(image):
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

    # K-Means Clustering:
    def kmeans_segmentation(image, k=3):
        image_flat = image.reshape((-1, 3))
        kmeans = cv2.KMeans(n_clusters=k, random_state=0).fit(image_flat)
        segmented_image = kmeans.labels_.reshape(image.shape[:2])
        return segmented_image

    # Morphological Operations
    def morphological_operations(image, operation='open', kernel_size=(5, 5)):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    if operation == 'open':
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation == 'close':
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


    # Create a binary mask
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    # Post-process the mask: Remove small areas and fill holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))  # Fill holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # Remove small areas

    return mask


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

    return blob_image, contour_image


def remove_clouds(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, cloud_mask = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
    cloud_mask = cv2.bitwise_not(cloud_mask)
    cleaned_image = cv2.bitwise_and(image, image, mask=cloud_mask)
    return cleaned_image


def denoise_image(image):
    denoised_image = cv2.GaussianBlur(image, (5, 5), 0)
    return denoised_image


def ensure_rgb(image):
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        return image
    else:
        raise ValueError("Unsupported image format or channels.")'''

# Dont uncomment this one out(under)
'''
def segment_image(image):
    # Check if the image is in RGB format
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Image should have 3 channels (RGB).")

    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

   


def detect_blobs(image):
    # Check if the image is in RGB format
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Image should have 3 channels (RGB).")

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    _, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 100
    params.filterByCircularity = True
    params.minCircularity = 0.8
    params.filterByConvexity = True
    params.minConvexity = 0.8
    params.filterByInertia = True
    params.minInertiaRatio = 0.5

    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(binary_image)

    output_image = cv2.drawKeypoints(image, keypoints, np.zeros((1, 1), dtype=np.uint8), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return output_image, keypoints


def detect_contours(image):
    # Check if the image is in RGB format
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Image should have 3 channels (RGB).")

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    _, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_image = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)
    return output_image, contours'''
