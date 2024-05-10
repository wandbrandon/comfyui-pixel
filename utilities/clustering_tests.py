import math
from PIL import Image
import cv2
import numpy as np
import skimage
from PIL import Image, ImageEnhance
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from skimage.exposure import match_histograms
from sklearn.cluster import (
    DBSCAN,
    AffinityPropagation,
    KMeans,
    MeanShift,
    OPTICS,
    SpectralClustering,
    AgglomerativeClustering,
    Birch,
    MiniBatchKMeans,
    MeanShift,
    OPTICS,
    SpectralClustering,
)
from correction_utils import (
    brightness_correction_michelson,
    brightness_correction_std,
    contrast_correction_michelson,
    contrast_correction_std,
    gamma_correction,
    constrast_correction_mask,
    hue_color_match,
    lab_color_match,
    ycbcr_color_match,
)
from functools import lru_cache
import math
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import colour
import cv2
import numpy as np
import deltae
import torch
import kornia
from tqdm import tqdm
import torch_utils
import skimage
import palette_swapping_utils
import scale_utils
from PIL import Image
from itertools import product
from PIL import ImageDraw
from PIL import ImageFont
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.datasets import load_iris

font = ImageFont.truetype(
    "/Users/brandonwand/Documents/projects/comfyui-pixel/utilities/treebyfivemodifi.ttf",
    6,
)

testing_image = Image.open("examples/tigerrobot.png").convert("RGB")
custom_image = Image.open("examples/custom.png").convert("RGB")
# testing_image = scale_utils.oe_downscale(testing_image, 8, "nearest-neighbors")
testing_palette_image = Image.open("examples/oil-6-1x.png")


# testing_image = gamma_correction(testing_image)
testing_image = contrast_correction_michelson(testing_image, testing_palette_image)
testing_image = lab_color_match(testing_image, testing_palette_image)
# testing_image = ycbcr_color_match(testing_image, testing_palette_image)
testing_image.show()

source_image = np.asarray(testing_image) / 255.0

# Get the palettes of the images
source_palette = testing_image.convert("P", palette=Image.Palette.ADAPTIVE).getpalette()
target_paltte = testing_palette_image.convert(
    "P", palette=Image.Palette.ADAPTIVE
).getpalette()


source = np.asarray(source_palette).reshape(-1, 3) / 255.0
target = np.asarray(target_paltte).reshape(-1, 3) / 255.0

# turn palette to HSV
source_hsv_palette = skimage.color.rgb2hsv(source)
target_hsv_palette = skimage.color.rgb2hsv(target)

source_h = source_hsv_palette[:, 0].reshape(-1, 1) + 0.01
target_h = target_hsv_palette[:, 0].reshape(-1, 1) + 0.01
source_s = source_hsv_palette[:, 1].reshape(-1, 1) + 0.01
target_s = target_hsv_palette[:, 1].reshape(-1, 1) + 0.01
source_v = source_hsv_palette[:, 2].reshape(-1, 1) + 0.01
target_v = target_hsv_palette[:, 2].reshape(-1, 1) + 0.01

# data_source_x = np.cos(2 * np.pi * source_h) * (source_v ** (-1 / 2))
# data_source_y = np.sin(2 * np.pi * source_h) * (source_v ** (-1 / 2))
# data_target_x = np.cos(2 * np.pi * target_h) * (target_v ** (-1 / 2))
# data_target_y = np.sin(2 * np.pi * target_h) * (target_v ** (-1 / 2))
data_source_x = np.cos(2 * np.pi * source_h)
data_source_y = np.sin(2 * np.pi * source_h)
data_target_x = np.cos(2 * np.pi * target_h)
data_target_y = np.sin(2 * np.pi * target_h)

# polar convert
source_h = np.column_stack((data_source_x, data_source_y)).reshape(-1, 2)
target_h = np.column_stack((data_target_x, data_target_y)).reshape(-1, 2)

print(source_h.shape, target_h.shape)

# Plot and Cluster Source
plt.figure(1)
plot = sns.scatterplot(x=source_h[:, 0], y=source_h[:, 1], c=source, s=200)
plt.show()

source_cluster_labels = DBSCAN(eps=0.5, min_samples=1, n_jobs=-1).fit(source_h).labels_
print(source_cluster_labels)


plt.figure(2)
plot = sns.scatterplot(
    x=source_h[:, 0],
    y=source_h[:, 1],
    c=source_cluster_labels,
    s=200,
)
plt.show()


# Plot and cluster Target
plt.figure(3)
plot = sns.scatterplot(x=target_h[:, 0], y=target_h[:, 1], c=target, s=200)
plt.show()

target_cluster_labels = DBSCAN(eps=5, min_samples=1, n_jobs=-1).fit(target_h).labels_

print(target_cluster_labels)

plt.figure(4)
plot = sns.scatterplot(
    x=target_h[:, 0],
    y=target_h[:, 1],
    c=target_cluster_labels,
    s=200,
)
plt.show()

source_cluster_length = max(source_cluster_labels) + 1
target_cluster_length = max(target_cluster_labels) + 1

print("Source cluster length: ", source_cluster_length)
print("Target cluster length: ", target_cluster_length)

# compute the mean hue and saturation for each cluster
lab_source_cluster_means = np.zeros((source_cluster_length, 3))
for source_cluster_index in range(source_cluster_length):
    source_cluster_lab = skimage.color.rgb2lab(
        skimage.color.hsv2rgb(
            source_hsv_palette[source_cluster_labels == source_cluster_index]
        )
    )
    source_cluster_lab = source_cluster_lab.reshape(-1, 3)
    lab_source_cluster_means[source_cluster_index] = np.mean(source_cluster_lab, axis=0)

lab_target_cluster_means = np.zeros((target_cluster_length, 3))
for target_cluster_index in range(target_cluster_length):
    target_cluster_lab = skimage.color.rgb2lab(
        skimage.color.hsv2rgb(
            target_hsv_palette[target_cluster_labels == target_cluster_index]
        )
    )
    target_cluster_lab = target_cluster_lab.reshape(-1, 3)
    lab_target_cluster_means[target_cluster_index] = np.mean(target_cluster_lab, axis=0)

print(lab_source_cluster_means.shape)
print(lab_target_cluster_means.shape)

mapping = np.zeros(source_cluster_length)
for cluster, mean in enumerate(lab_source_cluster_means):
    print(f"Cluster {cluster}, Mean Color: {mean}")
    distances = np.zeros(len(lab_target_cluster_means))
    for target_cluster_index, target_mean in enumerate(lab_target_cluster_means):
        current_distance = palette_swapping_utils.deltaE(
            (tuple(mean), tuple(target_mean))
        )
        distances[target_cluster_index] = current_distance
        print(
            f" -> Distance to Target: Cluster {target_cluster_index}, Mean: {target_mean} -> {current_distance}"
        )
    print(f"Distances: {distances}")
    min_target_cluster = np.argmin(distances)
    print(f"Closest Target Cluster: {min_target_cluster}\n")
    mapping[cluster] = min_target_cluster

for source_cluster_index, target_cluster_index in enumerate(mapping):
    print(
        f"Source Cluster {source_cluster_index} -> Target Cluster {target_cluster_index}"
    )
    source_cluster_colors_rgb = source[source_cluster_labels == source_cluster_index]
    source_cluster_colors_hsv = source_hsv_palette[
        source_cluster_labels == source_cluster_index
    ]

    target_cluster_colors_rgb = target[target_cluster_labels == target_cluster_index]
    target_cluster_colors_hsv = target_hsv_palette[
        target_cluster_labels == target_cluster_index
    ]

    source_cluster_colors_hsv_adj = np.copy(source_cluster_colors_hsv)

    # # Brightness Correction
    # source_b = source_cluster_colors_hsv[:, 2]
    # target_b = target_cluster_colors_hsv[:, 2]

    # # Calculate min and max of both datasets
    # source_b_min, source_b_max = np.min(source_b), np.max(source_b)
    # target_b_min, target_b_max = np.min(target_b), np.max(target_b)

    # # Calculate scale and shift factors
    # scale = (
    #     (target_b_max - target_b_min) / (source_b_max - source_b_min)
    #     if source_b_max != source_b_min
    #     else 1
    # )
    # shift = target_b_min - source_b_min * scale

    # # Apply transformation
    # source_b_adj = source_b * scale + shift

    # source_cluster_colors_hsv_adj[:, 2] = source_b_adj

    for source_index, source_color in enumerate(source_cluster_colors_hsv_adj):
        print(f" -> Source Color #{source_index}: {source_color}")

        distances = np.zeros(len(target_cluster_colors_hsv))
        for target_color_indices, target_color in enumerate(target_cluster_colors_hsv):
            weighted_source = np.array((source_color[2], source_color[1]))
            weighted_target = np.array((target_color[2], source_color[1]))
            distances[target_color_indices] = np.linalg.norm(
                weighted_source - weighted_target
            )
        print(f" -> Distances: {distances}")
        min_distance = np.argmin(distances)
        print(f" -> Closest Target Color: {target_cluster_colors_hsv[min_distance]}\n")
        # replace all the colors in the source image with the target color
        source_image[
            (source_image[:, :, 0] == source_cluster_colors_rgb[source_index][0])
            & (source_image[:, :, 1] == source_cluster_colors_rgb[source_index][1])
            & (source_image[:, :, 2] == source_cluster_colors_rgb[source_index][2])
        ] = target_cluster_colors_rgb[min_distance]

    plt.figure(5 + source_cluster_index)
    sns.scatterplot(
        x=source_cluster_colors_hsv_adj[:, 1],
        y=source_cluster_colors_hsv_adj[:, 2],
        c=source_cluster_colors_rgb,
        s=200,
    )
    plt.figure(5 + source_cluster_index)
    sns.scatterplot(
        x=target_cluster_colors_hsv[:, 1] + 10,
        y=target_cluster_colors_hsv[:, 2],
        c=target_cluster_colors_rgb,
        s=200,
    )
    plt.show()

source_image = Image.fromarray((source_image * 255).astype(np.uint8))
source_image.show()
