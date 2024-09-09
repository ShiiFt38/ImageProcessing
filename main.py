import os
import cv2
import numpy as np
import preprocessing
import feature_extraction
import utils


def overlay_features(image, refined_mask, edges, hog_image):
    # Convert single-channel masks to 3-channel
    refined_mask_rgb = cv2.cvtColor(refined_mask, cv2.COLOR_GRAY2BGR)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Layer the refined mask with transparency (e.g., 0.3 alpha)
    layer_1 = cv2.addWeighted(image, 0.3, refined_mask_rgb, 0.5, 0)

    # Layer the edges with transparency (e.g., 0.2 alpha)
    layer_2 = cv2.addWeighted(layer_1, 0.2, edges_rgb, 0.5, 0)

    # Overlay HOG feature visualization
    hog_rgb = cv2.cvtColor((hog_image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    final_overlay = cv2.addWeighted(layer_2, 0.2, hog_rgb, 0.5, 0)

    return final_overlay


def overlay_image_processing_features(image, exg_mask, edges, hog_image):
    # Convert floating-point images to 8-bit format
    exg_mask = np.clip(exg_mask * 255, 0, 255).astype(np.uint8)
    edges = np.clip(edges * 255, 0, 255).astype(np.uint8)
    hog_image = np.clip(hog_image * 255, 0, 255).astype(np.uint8)

    # Convert single-channel masks to 3-channel
    exg_mask_rgb = cv2.cvtColor(exg_mask, cv2.COLOR_GRAY2BGR)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Layer the ExG mask with transparency (e.g., 0.3 alpha)
    layer_1 = cv2.addWeighted(image, 0.7, exg_mask_rgb, 0.5, 0)

    # Layer the edges with transparency (e.g., 0.2 alpha)
    layer_2 = cv2.addWeighted(layer_1, 0.7, edges_rgb, 0.5, 0)

    # Overlay HOG feature visualization
    hog_rgb = cv2.cvtColor(hog_image, cv2.COLOR_GRAY2BGR)
    final_overlay = cv2.addWeighted(layer_2, 0.7, hog_rgb, 0.5, 0)

    return final_overlay


def process_images_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith('.tif'):
            image_path = os.path.join(directory, filename)
            print(f"Processing Found Image: {filename}...")

            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = np.clip(image, 0, 255).astype(np.uint8)

            # Image Processing
            processed_data = preprocessing.process_sentinel_images(image_path)
            _, exg, fine_tuned_mask, refined_mask, mask, classified_image, smoothed_mask, edges, final_mask = processed_data

            # Feature extraction
            hog_image, features = feature_extraction.features_extract(image)

            # Adjust brightness of the original image
            bright_image = cv2.convertScaleAbs(image, alpha=1.2, beta=30)

            # Normalize HOG image and apply color map
            hog_image_normalized = cv2.normalize(hog_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            color_mapped_hog = cv2.applyColorMap(hog_image_normalized, cv2.COLORMAP_JET)


            # Check if `features` and `hog_image` have the expected content
            print("HOG Image shape:", hog_image.shape if hog_image is not None else "None")
            print("Features:", features)

            # Save feature extraction metrics as CSV
            feature_extraction.save_metrics_as_csv(image_path, features['color_histogram'], features['texture'])

            # Apply overlays
            feature_overlay = overlay_features(image, refined_mask, edges, hog_image)
            image_processing_overlay = overlay_image_processing_features(image, exg, edges, hog_image)

            # Combine both overlays (if needed, otherwise choose one)
            final_overlay = cv2.addWeighted(feature_overlay, 0.5, image_processing_overlay, 0.5, 0)

            # Save the final overlaid image
            output_image_path = os.path.join(directory, f"final_overlay_{os.path.splitext(filename)[0]}.png")
            cv2.imwrite(output_image_path, final_overlay)

            print(f"Saved final overlay image: {output_image_path}")


def main():
    directory = ("images_2023")

    process_images_in_directory(directory)

    print("Processing complete. Results saved.")


if __name__ == "__main__":
    main()

