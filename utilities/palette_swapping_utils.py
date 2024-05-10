from functools import lru_cache
import math
from PIL import Image, ImageEnhance
import colour
import cv2
import numpy as np
import deltae
import torch
import kornia
from tqdm import tqdm
from correction_utils import (
    brightness_correction_michelson,
    contrast_correction_michelson,
    contrast_correction_std,
    gamma_correction,
    constrast_correction_mask,
    hue_color_match,
    lab_color_match,
)
import torch_utils
import skimage
import palette_swapping_utils
import scale_utils
from PIL import Image
from itertools import product
from PIL import ImageDraw
from PIL import ImageFont


@lru_cache
def deltaE(key):
    X, Y = key
    p1_dict = {"L": X[0], "a": X[1], "b": X[2]}
    p2_dict = {"L": Y[0], "a": Y[1], "b": Y[2]}
    val = deltae.delta_e_2000(p1_dict, p2_dict)
    # val = colour.delta_E(X, Y, method="CIE 2000")
    # print("p1:", p1_dict, "p2:", p2_dict, "val:", val)
    return val


def slow_cie_delta_e(image: Image, palette_image: Image):
    source_palette_img = image.convert("P", palette=Image.Palette.ADAPTIVE)
    source_palette = source_palette_img.getpalette()

    # Extract the binary palette data (this includes RGB values for each entry)
    source_image = np.asarray(source_palette_img.convert("RGB")) / 255.0
    source_palette_rgb = np.asarray(source_palette).reshape(-1, 3) / 255.0
    target_palette_rgb = np.asarray(palette_image.getpalette()).reshape(-1, 3) / 255.0

    print("Source Palette Shape: ", source_palette_rgb.shape)
    print("Target Palette Shape: ", target_palette_rgb.shape)

    lab_source_img = skimage.color.rgb2lab(source_image)
    lab_source_palette = skimage.color.rgb2lab(source_palette_rgb)
    lab_target_palette = skimage.color.rgb2lab(target_palette_rgb)

    for x in tqdm(range(len(lab_source_palette))):
        distances = np.zeros(len(lab_target_palette))
        source_lab = lab_source_palette[x]

        for y in range(len(lab_target_palette)):
            target_lab = lab_target_palette[y]
            key = (tuple(source_lab), tuple(target_lab))
            distances[y] = deltaE(key)
        min_dis = np.argmin(distances)

        # Find where all the channels match the old_value
        matches = np.all(lab_source_img == lab_source_palette[x], axis=-1)

        # Replace the found locations with the new RGB value
        lab_source_img[matches] = lab_target_palette[min_dis]
        # print(f"At {x} index -> {distances} to the new palette.")
        # print(f"{matches.sum()} pixels were replaced.")

    pillow_image = Image.fromarray(
        (skimage.color.lab2rgb(lab_source_img) * 255).astype(np.uint8)
    )

    return pillow_image


# Define the palette_swap function for node usage
def palette_swap(image: Image, palette_image: Image, method: str, gamma: float = None):
    if palette_image is None:
        raise ValueError("Palette image must be provided for palette swapping.")
    else:
        palette_image.convert("P", palette=Image.Palette.ADAPTIVE),

    match method:
        case "Pillow Quantize":
            return image.quantize(
                palette=palette_image, method=Image.Quantize.MAXCOVERAGE, dither=0
            ).convert("RGB")
        case "DeltaE 2000":
            return slow_cie_delta_e(image, palette_image)
        case "CC1":
            return contrast_correction_michelson(image, palette_image)
        case "CC2":
            return contrast_correction_std(image, palette_image)
        case "Wildcard Testing":
            image = gamma_correction(image)
            image = lab_color_match(image, palette_image)
            return image
        case "Wildcard Testing2":
            # image = gamma_correction(image)
            # image = contrast_correction_michelson(image, palette_image)
            image = brightness_correction_michelson(image, palette_image)
            return image.quantize(
                palette=palette_image, method=Image.Quantize.MAXCOVERAGE, dither=0
            ).convert("RGB")
            return image
        case _:
            raise ValueError(f"Method {method} is not supported for palette swapping.")

    return
