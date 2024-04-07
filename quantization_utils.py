from PIL import Image
import cv2
import numpy as np
from scipy.spatial import distance


def determine_best_k(image: Image, max_k: int):
    # Prepare arrays for distortion calculation
    pixels = np.array(image)
    pixel_indices = np.reshape(pixels, (-1, 3))

    # Calculate distortion for different values of k
    distortions = []
    for k in range(1, max_k + 1):
        quantized_image = image.quantize(
            colors=k, method=Image.Quantize.FASTOCTREE, kmeans=k, dither=0
        )
        centroids = np.array(quantized_image.getpalette()[: k * 3]).reshape(-1, 3)

        # Calculate distortions
        distances = np.linalg.norm(pixel_indices[:, np.newaxis] - centroids, axis=2)
        min_distances = np.min(distances, axis=1)
        distortions.append(np.sum(min_distances**2))

    # Calculate the rate of change of distortions
    rate_of_change = np.diff(distortions) / np.array(distortions[:-1])

    # Find the elbow point (best k value)
    if len(rate_of_change) == 0:
        best_k = 2
    else:
        elbow_index = np.argmax(rate_of_change) + 1
        best_k = elbow_index + 2

    return best_k


def find_nearest_color(palette, color):
    distances = distance.cdist([color], palette, "euclidean")
    return palette[np.argmin(distances)]


def kmeans_quantization(image: Image, k, palette_image: Image = None) -> Image:
    # Convert image from BGR to Lab colorspace
    lab_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)

    # Reshape the image into a 2D array of pixels
    pixels = lab_image.reshape((-1, 3))

    # Apply K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels.astype(np.float32), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    if palette_image:
        custom_palette_lab = cv2.cvtColor(
            np.unique(np.array(palette_image.convert("RGB"))), cv2.COLOR_RGB2LAB
        )

        # Map each pixel to the nearest color in your custom palette
        mapped_pixels = np.array(
            [find_nearest_color(custom_palette_lab, pixel) for pixel in pixels]
        )

        # Reshape back to original image shape
        mapped_image = mapped_pixels.reshape(lab_image.shape)

        # Convert back to RGB color space
        mapped_image_bgr = cv2.cvtColor(
            mapped_image.astype(np.uint8), cv2.COLOR_Lab2RGB
        )

        # Convert the image to Pillow format
        pillow_image = Image.fromarray(mapped_image_bgr)

        return pillow_image

    # Replace each pixel's value with the nearest cluster center
    quantized_pixels = centers[labels.flatten()]

    # Reshape the quantized image to its original shape
    quantized_image = quantized_pixels.reshape(lab_image.shape).astype(np.uint8)

    # Convert the quantized image from Lab to BGR colorspace
    quantized_image = cv2.cvtColor(quantized_image, cv2.COLOR_LAB2RGB)

    # Convert the image to Pillow format
    pillow_image = Image.fromarray(quantized_image)

    return pillow_image


# Define the palette_reduction function for node usage
def palette_reduction(
    image: Image, palette_size: int, method: str, palette_image: Image = None
) -> Image:
    match method:
        case "Quantize.MEDIANCUT":
            return image.quantize(
                colors=palette_size,
                method=Image.Quantize.MEDIANCUT,
                kmeans=palette_size,
                palette=palette_image,
                dither=Image.Dither.NONE,
            ).convert("RGB")
        case "Quantize.MAXCOVERAGE":
            return image.quantize(
                colors=palette_size,
                method=Image.Quantize.MAXCOVERAGE,
                kmeans=palette_size,
                palette=palette_image,
                dither=Image.Dither.NONE,
            ).convert("RGB")
        case "Quantize.FASTOCTREE":
            return image.quantize(
                colors=palette_size,
                method=Image.Quantize.FASTOCTREE,
                kmeans=palette_size,
                palette=palette_image,
                dither=Image.Dither.NONE,
            ).convert("RGB")
        case "Elbow Method":
            best_k = determine_best_k(image, palette_size)
            return image.quantize(
                colors=best_k,
                method=Image.Quantize.MAXCOVERAGE,
                kmeans=best_k,
                palette=palette_image,
                dither=Image.Dither.NONE,
            ).convert("RGB")
        case "cv2.kmeans":
            return kmeans_quantization(image, palette_size, palette_image=palette_image)
        case _:
            raise ValueError(f"Unknown method: {method}")


# Define the palette_swap function for node usage
def palette_swap(image: Image, palette_image: Image, method: str):
    if palette_image is None:
        raise ValueError("Palette image must be provided for palette swapping.")

    print("Palette image:", palette_image)

    return palette_reduction(
        image,
        0,
        method,
        palette_image.convert("P", palette=Image.Palette.ADAPTIVE),
    )
