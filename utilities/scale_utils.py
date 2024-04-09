import math
from PIL import Image
import numpy as np
from itertools import product
import cv2

from oe_utils import outline_expansion, match_color


def k_centroid(image: Image, downscale_factor: int, centroids=2) -> Image:
    new_height = int(image.height / downscale_factor)
    new_width = int(image.width / downscale_factor)

    # Create an empty array for the downscaled image
    downscaled = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Iterate over each tile in the downscaled image
    for x, y in product(range(new_width), range(new_height)):
        # Crop the tile from the original image
        tile = image.crop(
            (
                x * downscale_factor,
                y * downscale_factor,
                (x * downscale_factor) + downscale_factor,
                (y * downscale_factor) + downscale_factor,
            )
        )

        # Quantize the colors of the tile using k-means clustering
        tile = tile.quantize(colors=centroids, method=1, kmeans=centroids).convert(
            "RGB"
        )

        # Get the color counts and find the most common color
        color_counts = tile.getcolors()
        most_common_color = max(color_counts, key=lambda x: x[0])[1]

        # Assign the most common color to the corresponding pixel in the downscaled image
        downscaled[y, x, :] = most_common_color

    return Image.fromarray(downscaled, mode="RGB")


def nearest_neighbors(image: Image, downscale_factor: int) -> Image:
    new_height = round(image.height / downscale_factor)
    new_width = round(image.width / downscale_factor)

    return image.resize((new_width, new_height), Image.NEAREST)


# Define the downscale function for node usage
def downscale(image, downscale_factor: float, method: str) -> Image:
    match method:
        case "k-centroid":
            downscaled = k_centroid(image, downscale_factor)
        case "nearest-neighbors":
            downscaled = nearest_neighbors(image, downscale_factor)
        case _:
            raise ValueError(f"Unknown method: {method}")

    return downscaled


# Define the downscale function with outline expansion for node usage
def oe_downscale(image, downscale_factor: float, method: str, centroids=2) -> Image:
    # Calculate outline expansion inputs
    # patch_size = round(math.sqrt((image.width / width) * (image.height / height)) * 0.9)
    patch_size = round(math.sqrt(downscale_factor**2) * 0.9)
    thickness = round(patch_size / 5)

    # Convert to cv2 format
    cv2img = np.array(image)

    # Perform outline expansion
    org_img = cv2img.copy()
    cv2img = outline_expansion(cv2img, thickness, thickness, patch_size, 9, 4)

    # Convert back to PIL format
    image = Image.fromarray(cv2img).convert("RGB")

    # Downscale outline expanded image with k-centroid
    downscaled: Image
    match method:
        case "k-centroid":
            downscaled = k_centroid(image, downscale_factor, centroids)
        case "nearest-neighbors":
            downscaled = nearest_neighbors(image, downscale_factor)
        case _:
            raise ValueError(f"Unknown method: {method}")

    # Color matching
    downscaled_cv2 = cv2.cvtColor(np.array(downscaled), cv2.COLOR_RGB2BGR)
    org_img = cv2.cvtColor(org_img, cv2.COLOR_RGB2BGR)
    cv2img = match_color(
        downscaled_cv2,
        cv2.resize(
            org_img,
            (downscaled.width, downscaled.height),
            interpolation=cv2.INTER_LINEAR,
        ),
    )

    return Image.fromarray(cv2.cvtColor(cv2img, cv2.COLOR_BGR2RGB)).convert("RGB")
