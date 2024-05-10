import math
from PIL import Image
import cv2
import numpy as np
import skimage
from PIL import Image, ImageEnhance
from tqdm import tqdm


def gamma_correction(image: Image):
    # convert img to HSV
    source_hsv = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2HSV)
    source_hue, source_sat, source_val = cv2.split(source_hsv)
    # target_hue, target_sat, target_val = cv2.split(target_hsv)

    # compute gamma = log(mid*255)/log(mean)
    mid = 0.50
    mean = np.mean(source_val)
    gamma = math.log(mid * 255) / math.log(mean)
    print("Gamma of Image: ", gamma)

    # do gamma correction on value channel
    val_gamma = np.power(source_val, gamma).clip(0, 255).astype(np.uint8)

    # combine new value channel with original hue and sat channels
    hsv_gamma = cv2.merge([source_hue, source_sat, val_gamma])
    img_gamma = cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2RGB)

    pillow_gamma_corrected = Image.fromarray(img_gamma)

    return pillow_gamma_corrected


def contrast_correction_michelson(image: Image, palette_image: Image):

    # converter = ImageEnhance.Color(image)
    # image = converter.enhance(0.5)

    source_pil = np.asarray(image) / 255.0
    target_pil = np.asarray(palette_image.convert("RGB")) / 255.0

    # Convert both images to LAB color space
    source_lab = skimage.color.rgb2lab(source_pil)
    target_lab = skimage.color.rgb2lab(target_pil)

    # Extract the Luminance channel
    source_l = source_lab[:, :, 0]
    target_l = target_lab[:, :, 0]

    # compute min and max of Y
    min = np.min(source_l)
    max = np.max(source_l)

    min_palette = np.min(target_l)
    max_palette = np.max(target_l)

    # compute contrast
    contrast = (max - min) / (max + min)
    print(f"Contrast: {contrast}")
    contrast_palette = (max_palette - min_palette) / (max_palette + min_palette)
    print(f"Contrast Palette: {contrast_palette}")

    # # modify contrast of original image to match palette
    source_l_adj = (source_l - min) * (contrast_palette / contrast) + min_palette

    # compute min and max of new adjusted
    min = np.min(source_l_adj)
    max = np.max(source_l_adj)
    # compute contrast
    contrast = (max - min) / (max + min)
    print(f"Adjusted Contrast: {contrast}")

    # Replace the L channel in the source image LAB
    adjusted_lab = np.copy(source_lab)
    adjusted_lab[:, :, 0] = source_l_adj

    # Convert LAB back to RGB
    adjusted_rgb = skimage.color.lab2rgb(adjusted_lab)

    new_image = Image.fromarray((adjusted_rgb * 255).astype(np.uint8))

    return new_image


def brightness_correction_michelson(image: Image, palette_image: Image):

    source_pil = np.asarray(image) / 255.0
    target_pil = np.asarray(palette_image.convert("RGB")) / 255.0

    # Convert both images to LAB color space
    source_lab = skimage.color.rgb2hsv(source_pil)
    target_lab = skimage.color.rgb2hsv(target_pil)

    # Extract the Luminance channel
    source_l = source_lab[:, :, -1]
    target_l = target_lab[:, :, -1]

    # compute min and max of Y
    min = np.min(source_l)
    max = np.max(source_l)

    print("Source -> Min: ", min, "Max: ", max)

    min_palette = np.min(target_l)
    max_palette = np.max(target_l)

    print("Palette -> Min: ", min_palette, "Max: ", max_palette)

    # compute contrast
    contrast = (max - min) / (max + min)
    print(f"Contrast: {contrast}")
    contrast_palette = (max_palette - min_palette) / (max_palette + min_palette)
    print(f"Contrast Palette: {contrast_palette}")

    # # modify contrast of original image to match palette
    source_l_adj = (source_l - min) * (contrast_palette / contrast) + min_palette

    # compute min and max of new adjusted
    min = np.min(source_l_adj)
    max = np.max(source_l_adj)
    # compute contrast
    contrast = (max - min) / (max + min)
    print(f"Adjusted Contrast: {contrast}")

    # Replace the L channel in the source image LAB
    adjusted_lab = np.copy(source_lab)
    adjusted_lab[:, :, -1] = source_l_adj

    # Convert LAB back to RGB
    adjusted_rgb = skimage.color.hsv2rgb(adjusted_lab)

    new_image = Image.fromarray((adjusted_rgb * 255).astype(np.uint8))

    return new_image


def contrast_correction_std(image: Image, palette_image: Image):

    source_pil = np.asarray(image) / 255.0
    target_pil = np.asarray(palette_image.convert("RGB")) / 255.0

    # Convert both images to LAB
    source_lab = skimage.color.rgb2lab(source_pil)
    target_lab = skimage.color.rgb2lab(target_pil)

    # Extract the Luminance channel
    source_l = source_lab[:, :, 0]
    target_l = target_lab[:, :, 0]

    # Calculate the mean and standard deviation of the luminance channel
    source_mean, source_std = np.mean(source_l), np.std(source_l)
    target_mean, target_std = np.mean(target_l), np.std(target_l)

    # Adjust the source intensity to match the target
    source_l_adj = (source_l - source_mean) * (target_std / source_std) + target_mean

    # Clip the adjusted intensity to the range [0, 100]
    source_l_adj = np.clip(source_l_adj, 0, 100)

    # Replace the L channel in the source image LAB
    adjusted_lab = np.copy(source_lab)
    adjusted_lab[:, :, 0] = source_l_adj

    # Convert the adjusted intensity back to RGB
    adjusted_rgb = skimage.color.lab2rgb(adjusted_lab)

    new_image = Image.fromarray((adjusted_rgb * 255).astype(np.uint8))

    return new_image


def brightness_correction_std(image: Image, palette_image: Image):

    source_pil = np.asarray(image) / 255.0
    target_pil = np.asarray(palette_image.convert("RGB")) / 255.0

    # Convert both images to LAB
    source_hsv = skimage.color.rgb2hsv(source_pil)
    target_hsv = skimage.color.rgb2hsv(target_pil)

    # Extract the Luminance channel
    source_b = source_hsv[:, :, -1]
    target_b = target_hsv[:, :, -1]

    # Calculate the mean and standard deviation of the luminance channel
    source_mean, source_std = np.mean(source_b), np.std(source_b)
    target_mean, target_std = np.mean(target_b), np.std(target_b)

    # Adjust the source intensity to match the target
    source_b_adj = (source_b - source_mean) * (target_std / source_std) + target_mean

    # Replace the L channel in the source image LAB
    adjusted_hsv = np.copy(source_hsv)
    adjusted_hsv[:, :, -1] = source_b_adj

    # Convert the adjusted intensity back to RGB
    adjusted_rgb = skimage.color.hsv2rgb(adjusted_hsv)

    new_image = Image.fromarray((adjusted_rgb * 255).astype(np.uint8))

    return new_image


def constrast_correction_mask(
    image: Image, palette_swapped_image: Image, palette_image: Image
):

    source_pil = np.asarray(image) / 255.0
    target_pil = np.asarray(palette_swapped_image) / 255.0

    # Convert both images to LAB color space
    source_lab = skimage.color.rgb2lab(source_pil)
    target_lab = skimage.color.rgb2lab(target_pil)

    # Extract the Luminance channel
    source_l = source_lab[:, :, 0]
    target_l = target_lab[:, :, 0]

    mask = np.zeros_like(source_lab)
    mask[:, :, 0] = source_l - target_l

    # Image.fromarray((skimage.color.lab2rgb(mask) * 255).astype(np.uint8)).show()

    luminance_mask = mask[:, :, 0]

    # constrast_corrected_l_mask = np.asarray(
    #     contrast_correction_std(mask, palette_image)
    # )[:, :, 0]

    adjusted_lab = np.copy(source_lab)
    adjusted_lab[:, :, 0] = np.clip(source_l + np.clip(luminance_mask, 50, 100), 0, 100)

    # Convert LAB back to RGB
    masked_rgb = skimage.color.lab2rgb(adjusted_lab)

    masked_image = Image.fromarray((masked_rgb * 255).astype(np.uint8))

    return masked_image


def lab_color_match(image: Image, palette_image: Image):
    source_palette_img = image.convert("P", palette=Image.Palette.ADAPTIVE)
    source_palette = source_palette_img.getpalette()

    # Extract the binary palette data (this includes RGB values for each entry)
    source_image = np.asarray(source_palette_img.convert("RGB")) / 255.0
    source_palette_rgb = np.asarray(source_palette).reshape(-1, 3) / 255.0
    target_palette_rgb = np.asarray(palette_image.getpalette()).reshape(-1, 3) / 255.0

    print("Source Palette Shape: ", source_palette_rgb.shape)
    print("Target Palette Shape: ", target_palette_rgb.shape)

    lab_source_img = skimage.color.rgb2lab(source_image)
    source_pal_hsv = skimage.color.rgb2lab(source_palette_rgb)
    target_pal_hsv = skimage.color.rgb2lab(target_palette_rgb)

    for x in tqdm(range(len(source_pal_hsv))):
        distances = np.zeros(len(target_pal_hsv))
        source_lab = source_pal_hsv[x]

        for y in range(len(target_pal_hsv)):
            target_lab = target_pal_hsv[y]
            key = (tuple(source_lab), tuple(target_lab))
            distances[y] = np.linalg.norm(source_lab[1:] - target_lab[1:])
        min_dis = np.argmin(distances)

        # Find where all the channels match the old_value
        matches = np.all(lab_source_img == source_pal_hsv[x], axis=-1)

        # Replace the found locations with the new RGB value
        lab_source_img[matches] = [
            source_pal_hsv[x][0],
            target_pal_hsv[min_dis][1],
            target_pal_hsv[min_dis][2],
        ]
        # print(f"At {x} index -> {distances} to the new palette.")
        # print(f"{matches.sum()} pixels were replaced.")

    pillow_image = Image.fromarray(
        (skimage.color.lab2rgb(lab_source_img) * 255).astype(np.uint8)
    )

    return pillow_image


def hue_color_match(image: Image, palette_image: Image):
    source_palette_img = image.convert("P", palette=Image.Palette.ADAPTIVE)
    source_palette = source_palette_img.getpalette()

    # Extract the binary palette data (this includes RGB values for each entry)
    source_image = np.asarray(source_palette_img.convert("RGB")) / 255.0
    source_palette_rgb = np.asarray(source_palette).reshape(-1, 3) / 255.0
    target_palette_rgb = np.asarray(palette_image.getpalette()).reshape(-1, 3) / 255.0

    # Convert both images to HSV color space
    source_hsv = skimage.color.rgb2hsv(source_image)
    source_pal_hsv = skimage.color.rgb2hsv(source_palette_rgb)
    target_pal_hsv = skimage.color.rgb2hsv(target_palette_rgb)

    # convert hue to polar coordinates
    source_h = source_pal_hsv[:, 0].reshape(-1, 1)
    target_h = target_pal_hsv[:, 0].reshape(-1, 1)
    source_s = source_pal_hsv[:, 1].reshape(-1, 1)
    target_s = target_pal_hsv[:, 1].reshape(-1, 1)
    data_source_x = np.cos(2 * np.pi * source_h)
    data_source_y = np.sin(2 * np.pi * source_h)
    data_target_x = np.cos(2 * np.pi * target_h)
    data_target_y = np.sin(2 * np.pi * target_h)

    # polar convert
    source_h = np.column_stack((data_source_x, data_source_y)).reshape(-1, 2)
    target_h = np.column_stack((data_target_x, data_target_y)).reshape(-1, 2)

    print(source_h.shape, target_h.shape, source_pal_hsv.shape, target_pal_hsv.shape)

    for x in tqdm(range(len(source_pal_hsv))):
        distances = np.zeros(len(target_pal_hsv))
        for y in range(len(target_pal_hsv)):
            distances[y] = np.linalg.norm(source_h[x] - target_h[y])
        min_dis = np.argmin(distances)

        # Find where all the channels match the old_value
        matches = np.all(source_hsv == source_pal_hsv[x], axis=-1)

        # Replace the found locations with the new RGB value
        source_hsv[matches] = [
            target_pal_hsv[min_dis][0],
            *source_pal_hsv[x][1:],
        ]
        # print(f"At {x} index -> {distances} to the new palette.")
        # print(f"{matches.sum()} pixels were replaced.")

    pillow_image = Image.fromarray(
        (skimage.color.hsv2rgb(source_hsv) * 255).astype(np.uint8)
    )

    return pillow_image


def ycbcr_color_match(image: Image, palette_image: Image):
    source_palette_img = image.convert("P", palette=Image.Palette.ADAPTIVE)
    source_palette = source_palette_img.getpalette()

    # Extract the binary palette data (this includes RGB values for each entry)
    source_image = np.asarray(source_palette_img.convert("RGB")) / 255.0
    source_palette_rgb = np.asarray(source_palette).reshape(-1, 3) / 255.0
    target_palette_rgb = np.asarray(palette_image.getpalette()).reshape(-1, 3) / 255.0

    print("Source Palette Shape: ", source_palette_rgb.shape)
    print("Target Palette Shape: ", target_palette_rgb.shape)

    ycbcr_source = skimage.color.rgb2ycbcr(source_image)
    ycbcr_source_palette = skimage.color.rgb2ycbcr(source_palette_rgb)
    ycbcr_target_palette = skimage.color.rgb2ycbcr(target_palette_rgb)

    for x in tqdm(range(len(ycbcr_source_palette))):
        distances = np.zeros(len(ycbcr_target_palette))
        source_ycbcr = ycbcr_source_palette[x]
        for y in range(len(ycbcr_target_palette)):
            target_ycbcr = ycbcr_target_palette[y]
            distances[y] = np.linalg.norm(source_ycbcr[1:] - target_ycbcr[1:])
        min_dis = np.argmin(distances)

        # Find where all the channels match the old_value
        matches = np.all(ycbcr_source == ycbcr_source_palette[x], axis=-1)

        # Replace the found locations with the new RGB value
        ycbcr_source[matches] = [
            ycbcr_source_palette[x][0],
            ycbcr_target_palette[min_dis][1],
            ycbcr_target_palette[min_dis][2],
        ]
        # print(f"At {x} index -> {distances} to the new palette.")
        # print(f"{matches.sum()} pixels were replaced.")

    pillow_image = Image.fromarray(
        (skimage.color.ycbcr2rgb(ycbcr_source) * 255).astype(np.uint8)
    )

    return pillow_image
