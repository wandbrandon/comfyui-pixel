from functools import lru_cache
from PIL import Image
import numpy as np
import deltae
import torch
import kornia
from tqdm import tqdm
import torch_utils
import skimage


@lru_cache
def deltaE(key):
    X, Y = key
    p1_dict = {"L": X[0], "a": X[1], "b": X[2]}
    p2_dict = {"L": Y[0], "a": Y[1], "b": Y[2]}
    val = deltae.delta_e_2000(p1_dict, p2_dict)
    # print("p1:", p1_dict, "p2:", p2_dict, "val:", val)
    return val


def torch_cie_delta_e(image: Image, palette_image: Image):

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        print("Warning: CIE Delta E 2000 is slow on CPU. Consider using CUDA or MPS.")

    original_palette = image.convert("P", palette=Image.Palette.ADAPTIVE).getpalette()
    # Extract the binary palette data (this includes RGB values for each entry)
    original_palette_rgb = np.asarray(original_palette).reshape(-1, 3)
    palette_image_rgb = np.asarray(palette_image.getpalette()).reshape(-1, 3)

    # turn palettes to tensors here.
    rgb_image_tensor = torch.from_numpy(original_palette_rgb)
    palette_rgb_image_tensor = torch.from_numpy(palette_image_rgb)

    # normalize to 0, 1
    rgb_image_tensor = rgb_image_tensor / 255.0
    palette_rgb_image_tensor = palette_rgb_image_tensor / 255.0

    print(rgb_image_tensor.shape, palette_rgb_image_tensor.shape)

    # add a dimension for batch and permute to NCHW
    rgb_image_tensor = rgb_image_tensor.expand(1, 1, -1, -1)
    rgb_image_tensor = rgb_image_tensor.permute(0, 3, 1, 2)
    palette_rgb_image_tensor = palette_rgb_image_tensor.expand(1, 1, -1, -1)
    palette_rgb_image_tensor = palette_rgb_image_tensor.permute(0, 3, 1, 2)

    print(rgb_image_tensor.shape, palette_rgb_image_tensor.shape)

    # conversion
    lab_image_tensor = kornia.color.rgb_to_lab(rgb_image_tensor)
    lab_image_tensor = lab_image_tensor.permute(0, 2, 3, 1)

    lab_palette_image_tensor = kornia.color.rgb_to_lab(palette_rgb_image_tensor)
    lab_palette_image_tensor = lab_palette_image_tensor.permute(0, 2, 3, 1)

    print(lab_image_tensor.shape, lab_palette_image_tensor.shape)

    # flatten the image
    flattened = torch.reshape(lab_image_tensor, (-1, 3)).to(device)
    flattened_palette = torch.reshape(lab_palette_image_tensor, (-1, 3)).to(device)

    print(flattened.shape, flattened_palette.shape)

    # indices of all unique pairwise comparisons (triu=upper triangle). Offset 1 since we don't need to compare A to A
    x, y = torch.triu_indices(row=len(flattened), col=len(flattened_palette), offset=1)

    x = x.to(device)
    y = y.to(device)

    distance_matrix = torch_utils.ciede2000_diff(
        flattened[x],
        flattened_palette[y],
        device=device,
    )

    # TODO Actually implement this...

    # create the full matrix
    # distance_matrix = torch_utils.unflatten_upper_triangular(distance_matrix)

    print(distance_matrix.shape)
    print(distance_matrix)

    return image


def slow_cie_delta_e(image: Image, palette_image: Image):
    palettized_image = image.convert("P", palette=Image.Palette.ADAPTIVE)
    original_palette = palettized_image.getpalette()
    # Extract the binary palette data (this includes RGB values for each entry)
    original_palette_rgb = np.asarray(original_palette).reshape(-1, 3) / 255.0
    palette_image_rgb = np.asarray(palette_image.getpalette()).reshape(-1, 3) / 255.0

    print(original_palette_rgb.shape, palette_image_rgb.shape)

    lab_complete_image = (
        skimage.color.rgb2lab(np.asarray(palettized_image.convert("RGB")) / 255.0)
    ).astype(np.float16)
    lab_original_palette = skimage.color.rgb2lab(original_palette_rgb).astype(
        np.float16
    )
    lab_palette_image = skimage.color.rgb2lab(palette_image_rgb).astype(np.float16)

    for x in tqdm(range(len(lab_original_palette))):
        distances = np.zeros(len(lab_palette_image))
        for y in range(len(lab_palette_image)):
            distances[y] = deltaE(
                (tuple(lab_original_palette[x]), tuple(lab_palette_image[y]))
            )
        min_dis = np.argmin(distances)

        # Find where all the RGB channels match the old_value
        # matches = (lab_complete_image == lab_original_palette[x]).all(axis=-1)
        matches = np.all(lab_complete_image == lab_original_palette[x], axis=-1)

        # Replace the found locations with the new RGB value
        lab_complete_image[matches] = lab_palette_image[min_dis]
        # print(f"At {x} index -> {distances} to the new palette.")
        # print(f"{matches.sum()} pixels were replaced.")

    pillow_image = Image.fromarray(
        (skimage.color.lab2rgb(lab_complete_image) * 255).astype(np.uint8)
    )

    return pillow_image


# Define the palette_swap function for node usage
def palette_swap(image: Image, palette_image: Image, method: str):
    if palette_image is None:
        raise ValueError("Palette image must be provided for palette swapping.")
    else:
        palette_image.convert("P", palette=Image.Palette.ADAPTIVE),

    match method:
        case "Pillow Quantize":
            return image.quantize(
                palette=palette_image, method=Image.Quantize.MAXCOVERAGE, dither=0
            ).convert("RGB")
        case "CIELAB Delta E 20000":
            return slow_cie_delta_e(image, palette_image)
        case _:
            raise ValueError(f"Method {method} is not supported for palette swapping.")

    return
