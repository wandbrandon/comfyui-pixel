from PIL import Image
import numpy as np


def determine_best_k(image: Image, max_k: int):
    # Convert the image to RGB mode
    image = image.convert("RGB")

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


def elbow_method(image: Image, palette_size: int, palette_image: Image = None) -> int:
    best_k = determine_best_k(image, palette_size)
    return image.quantize(
        colors=best_k,
        method=Image.Quantize.MAXCOVERAGE,
        kmeans=best_k,
        palette=palette_image,
        dither=Image.Dither.NONE,
    ).convert("RGB")


def palette_reduction(
    image: Image, palette_size: int, method: str, palette_image: Image = None
) -> Image:
    image = image.convert("RGB")
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
            return elbow_method(image, palette_size, palette_image)
        case _:
            raise ValueError(f"Unknown method: {method}")


def palette_swap(image: Image, palette_image: Image):
    if palette_image is None:
        raise ValueError("Palette image must be provided for palette swapping.")

    return palette_reduction(image, 0, "Quantize.MAXCOVERAGE", palette_image)
