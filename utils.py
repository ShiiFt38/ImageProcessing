import cv2
import rasterio
import numpy as np
from rasterio.enums import Resampling


def load_image(image_path):
    with rasterio.open(image_path) as src:
        image = src.read([1, 2, 3])
        image = image.transpose(1, 2, 0)
    return image


def load_cloud_mask(mask_path):
    with rasterio.open(mask_path) as src:
        cloud_mask = src.read(1)  # Read the cloud mask band
    return cloud_mask


def create_cloud_free_composite(image_paths, cloud_mask_paths):
    # Load images and cloud masks
    images = [load_image(img_path) for img_path in image_paths]
    cloud_masks = [load_cloud_mask(mask_path) for mask_path in cloud_mask_paths]

    # Stack the images and cloud masks
    stacked_images = np.stack(images, axis=-1)
    stacked_masks = np.stack(cloud_masks, axis=-1)

    # Create a mask to identify the least cloudy pixels
    # Assuming cloud mask values are 0 for clear and 1 for cloudy
    clear_pixel_mask = np.min(stacked_masks, axis=-1) == 0

    # Create a composite by selecting the least cloudy pixels
    composite_image = np.where(clear_pixel_mask[:, :, np.newaxis],
                               np.max(stacked_images, axis=-1),
                               0)  # Replace cloudy pixels with 0
    return composite_image


def save_image(image, path):
    # Ensure the image is in the correct format
    if len(image.shape) == 2:  # Grayscale image
        with rasterio.open(
            path, 'w', driver='GTiff', height=image.shape[0], width=image.shape[1],
            count=1, dtype=image.dtype, crs='EPSG:4326', transform=rasterio.transform.Affine.identity()
        ) as dst:
            dst.write(image, 1)
    elif len(image.shape) == 3:  # RGB image
        with rasterio.open(
            path, 'w', driver='GTiff', height=image.shape[0], width=image.shape[1],
            count=image.shape[2], dtype=image.dtype, crs='EPSG:4326', transform=rasterio.transform.Affine.identity()
        ) as dst:
            for i in range(image.shape[2]):
                dst.write(image[:, :, i], i + 1)
    else:
        raise ValueError("Unsupported image shape for saving.")


