from comfyui_pixel import oe_utils, quantization_utils, scale_utils
from PIL import Image
from PIL import ImageEnhance

downscale_factor = 8
palette_size = 4
palette_method = "Quantize.MAXCOVERAGE"

testing_image = Image.open("examples/original.png")
palette_image = Image.open("examples/apollo-1x.png")

# Create a grid to display the images
grid_size = (2, 2)
grid_image = Image.new(
    "RGB", (testing_image.width * grid_size[1], testing_image.height * grid_size[0])
)

# Downscale using k-centroid method
k_centroid_scale = scale_utils.downscale(
    image=testing_image, downscale_factor=downscale_factor, method="k-centroid"
)
k_centroid_scale = quantization_utils.palette_reduction(
    k_centroid_scale,
    palette_size=palette_size,
    method=palette_method,
)
k_centroid_scale = quantization_utils.palette_swap(k_centroid_scale, palette_image)
k_centroid_scale = k_centroid_scale.resize(
    (testing_image.width, testing_image.height), resample=Image.Resampling.NEAREST
)

grid_image.paste(k_centroid_scale, (0, 0))

# Downscale using nearest-neighbors method
nearest_neighbors_scale = scale_utils.downscale(
    image=testing_image, downscale_factor=downscale_factor, method="nearest-neighbors"
)
nearest_neighbors_scale = quantization_utils.palette_reduction(
    nearest_neighbors_scale,
    palette_size=palette_size,
    method=palette_method,
)
nearest_neighbors_scale = nearest_neighbors_scale.resize(
    (testing_image.width, testing_image.height), resample=Image.Resampling.NEAREST
)
nearest_neighbors_scale = quantization_utils.palette_swap(
    nearest_neighbors_scale, palette_image
)
grid_image.paste(nearest_neighbors_scale, (testing_image.width, 0))

# Downscale using oe_k-centroid method
oe_k_centroid_scale = scale_utils.oe_downscale(
    image=testing_image, downscale_factor=downscale_factor, method="k-centroid"
)
oe_k_centroid_scale = quantization_utils.palette_reduction(
    oe_k_centroid_scale,
    palette_size=palette_size,
    method=palette_method,
)
oe_k_centroid_scale = oe_k_centroid_scale.resize(
    (testing_image.width, testing_image.height), resample=Image.Resampling.NEAREST
)
oe_k_centroid_scale = quantization_utils.palette_swap(
    oe_k_centroid_scale, palette_image
)
grid_image.paste(oe_k_centroid_scale, (0, testing_image.height))

# Downscale using oe_nearest-neighbors method
oe_nearest_neighbors_scale = scale_utils.oe_downscale(
    image=testing_image, downscale_factor=downscale_factor, method="nearest-neighbors"
)
oe_nearest_neighbors_scale = quantization_utils.palette_reduction(
    oe_nearest_neighbors_scale,
    palette_size=palette_size,
    method=palette_method,
)
oe_nearest_neighbors_scale = oe_nearest_neighbors_scale.resize(
    (testing_image.width, testing_image.height), resample=Image.Resampling.NEAREST
)

# enhancer = ImageEnhance.Contrast(oe_nearest_neighbors_scale)
# # Factor of 2.0 gives a contrast increase
# oe_nearest_neighbors_scale = enhancer.enhance(2)


oe_nearest_neighbors_scale = quantization_utils.palette_swap(
    oe_nearest_neighbors_scale, palette_image
)
grid_image.paste(
    oe_nearest_neighbors_scale, (testing_image.width, testing_image.height)
)

# Show the grid image
grid_image.show()
