from functools import lru_cache
from PIL import Image
import cv2
import numpy as np
import sklearn.cluster._kmeans as kmeans
import kmedoids
from sklearn.metrics import pairwise_distances
from skimage import io, color
import deltae
import torch
from tqdm import tqdm
import torch_utils as torch_utils
import kornia
from concurrent.futures import ProcessPoolExecutor


def deltaE(key):
    X, Y = key
    p1_dict = {"L": X[0], "a": X[1], "b": X[2]}
    p2_dict = {"L": Y[0], "a": Y[1], "b": Y[2]}
    val = deltae.deltaE(p1_dict, p2_dict)
    # print("p1:", p1_dict, "p2:", p2_dict, "val:", val)
    return val


def determine_best_k(image, max_k):
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


def kmeans_quantization_bgr(image: Image, k: int) -> Image:
    # Convert image from RGB to Lab colorspace
    rgb_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Reshape the image into a 2D array of pixels
    pixels = rgb_image.reshape((-1, 3))

    # Apply K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels.astype(np.float32), k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    # Replace each pixel's value with the nearest cluster center
    quantized_pixels = centers[labels.flatten()]

    # Reshape the quantized image to its original shape
    quantized_image = quantized_pixels.reshape(rgb_image.shape).astype(np.uint8)

    # Convert the quantized image from Lab to RGB colorspace
    quantized_image = cv2.cvtColor(quantized_image, cv2.COLOR_BGR2RGB)

    # Convert the image to Pillow format
    pillow_image = Image.fromarray(quantized_image)

    return pillow_image


def kmeans_quantization_rgb(image: Image, k: int) -> Image:
    # Convert image from RGB to Lab colorspace
    rgb_image = np.asarray(image)

    # Reshape the image into a 2D array of pixels
    pixels = rgb_image.reshape((-1, 3))

    # Apply K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels.astype(np.float32), k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    # Replace each pixel's value with the nearest cluster center
    quantized_pixels = centers[labels.flatten()]

    # Reshape the quantized image to its original shape
    quantized_image = quantized_pixels.reshape(rgb_image.shape).astype(np.uint8)

    # Convert the image to Pillow format
    pillow_image = Image.fromarray(quantized_image)

    return pillow_image


def kmeans_quantization_lab(image: Image, k: int) -> Image:
    # Convert image from RGB to Lab colorspace
    lab_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)

    # Reshape the image into a 2D array of pixels
    pixels = lab_image.reshape((-1, 3))

    # Apply K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels.astype(np.float32), k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    # Replace each pixel's value with the nearest cluster center
    quantized_pixels = centers[labels.flatten()]

    # Reshape the quantized image to its original shape
    quantized_image = quantized_pixels.reshape(lab_image.shape).astype(np.uint8)

    # Convert the quantized image from Lab to RGB colorspace
    quantized_image = cv2.cvtColor(quantized_image, cv2.COLOR_LAB2RGB)

    # Convert the image to Pillow format
    pillow_image = Image.fromarray(quantized_image)

    return pillow_image


def kmeans_quantization_lab_dE00(image: Image, k: int) -> Image:
    # Convert image from RGB to Lab colorspace
    lab_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)

    # Reshape the image into a 2D array of pixels
    lab_image_pixels = lab_image.reshape((-1, 3))

    def custom_distances(X, Y=None, Y_norm_squared=None, squared=False):
        return np.asarray([[deltaE((tuple(p1), tuple(p2))) for p2 in Y] for p1 in X])

    kmeans._euclidean_distances = custom_distances
    kmeans.euclidean_distances = custom_distances

    km = kmeans.KMeans(init="k-means++", n_clusters=k, random_state=0)

    labels = km.fit_predict(lab_image_pixels)

    # Replace each pixel's value with the nearest cluster center
    quantized_pixels = km.cluster_centers_[labels]

    # Reshape the quantized image to its original shape
    quantized_image = quantized_pixels.reshape(lab_image.shape).astype(np.uint8)

    # Convert the quantized image from Lab to RGB colorspace
    quantized_image = cv2.cvtColor(quantized_image, cv2.COLOR_LAB2RGB)

    # Convert the image to Pillow format
    pillow_image = Image.fromarray(quantized_image)

    return pillow_image


def kmedoids_quantization_lab_dE00(image: Image, k: int) -> Image:
    # # Convert image from RGB to Lab colorspace

    rgb_image_tensor = torch.from_numpy(np.array(image))

    # normalize to 0, 1
    rgb_image_tensor = rgb_image_tensor / 255.0

    # add a dimension for batch and permute to NCHW
    rgb_image_tensor = rgb_image_tensor.unsqueeze(0)
    rgb_image_tensor = rgb_image_tensor.permute(0, 3, 1, 2)

    # conversion
    lab_image_tensor = kornia.color.rgb_to_lab(rgb_image_tensor)
    lab_image_tensor = lab_image_tensor.permute(0, 2, 3, 1)
    print(lab_image_tensor.shape)

    # image_size = lab_image_tensor.shape[2] * lab_image_tensor.shape[3]

    # # pre compute distance matrix
    # m = torch_utils.ciede2000_diff(
    #     lab_image_tensor.repeat_interleave(image_size, dim=3),
    #     lab_image_tensor.repeat(1, 1, 1, image_size),
    #     device="mps",
    # )

    flattened = torch.reshape(lab_image_tensor, (-1, 3))
    print(flattened.shape)

    # num = len(flattened)
    # similarity_matrix = np.zeros((num, num))

    # for i in tqdm(range(0, num)):
    #     for j in tqdm(range(i + 1, num)):
    #         key = (tuple(np.asarray(flattened[i])), tuple(np.asarray(flattened[j])))
    #         # print(key)
    #         dist_tmp = deltaE(key)
    #         similarity_matrix[i][j] = dist_tmp
    #         similarity_matrix[j][i] = dist_tmp

    def compute_distance(args):
        i, j, flattened_i, flattened_j = args
        key = (flattened_i, flattened_j)
        dist_tmp = deltaE(key)
        print(key, dist_tmp)
        return i, j, dist_tmp

    # Prepare arguments for parallel processing

    # indices of all unique pairwise comparisons (triu=upper triangle). Offset 1 since we don't need to compare A to A
    x, y = torch.triu_indices(row=len(flattened), col=len(flattened), offset=1)
    distance_matrix = torch_utils.ciede2000_diff(
        flattened[x], flattened[y], device="mps"
    )

    print(distance_matrix.cpu().numpy())

    km = kmedoids.KMedoids(k)

    labels = km.fit_predict()

    # replace
    quantized_pixels = km.cluster_centers_[labels]

    # reshape
    quantized_image = quantized_pixels.reshape(lab_image_tensor.shape)

    # Convert the quantized image from Lab to RGB colorspace
    quantized_image = color.lab2rgb(quantized_image)

    # multiply by 255
    quantized_image = (quantized_image * 255).astype(np.uint8)

    print(quantized_image)

    # Convert the image to Pillow format
    pillow_image = Image.fromarray(quantized_image)

    return pillow_image


# Define the palette_quantization function for node usage
def palette_quantization(
    image: Image,
    palette_size: int,
    method: str,
    elbow_method: bool = False,
) -> Image:
    if elbow_method:
        palette_size = determine_best_k(image, palette_size)
    match method:
        case "Quantize.MEDIANCUT":
            return image.quantize(
                colors=palette_size,
                method=Image.Quantize.MEDIANCUT,
                kmeans=palette_size,
                dither=Image.Dither.NONE,
            ).convert("RGB")
        case "Quantize.MAXCOVERAGE":
            return image.quantize(
                colors=palette_size,
                method=Image.Quantize.MAXCOVERAGE,
                kmeans=palette_size,
                dither=Image.Dither.NONE,
            ).convert("RGB")
        case "cv2.kmeans_BGR":
            return kmeans_quantization_bgr(image, palette_size)
        case "cv2.kmeans_RGB":
            return kmeans_quantization_rgb(image, palette_size)
        case "cv2.kmeans_LAB":
            return kmeans_quantization_lab(image, palette_size)
        case "sklearn.kmeans_LAB_deltaE00":
            return kmeans_quantization_lab_dE00(image, palette_size)
        case "sklearn.kmedoids_LAB_deltaE00":
            return kmedoids_quantization_lab_dE00(image, palette_size)
        case _:
            raise ValueError(f"Unknown method: {method}")


# Define the palette_swap function for node usage
def palette_swap(image: Image, palette_image: Image, method: str):
    if palette_image is None:
        raise ValueError("Palette image must be provided for palette swapping.")

    match method:
        case "Quantize.MAXCOVERAGE":
            return image.quantize(
                palette_image=palette_image.convert(
                    "P", palette=Image.Palette.ADAPTIVE
                ),
                method=Image.Quantize.MAXCOVERAGE,
            ).convert("RGB")
        case "CIELAB DELTA E":
            return delta_cie_2000()

    return