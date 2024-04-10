import scale_utils
import quantization_utils
from PIL import Image
from itertools import product
from PIL import ImageDraw
from PIL import ImageFont

font = ImageFont.truetype(
    "/Users/brandonwand/Documents/projects/comfyui-pixel/utilities/treebyfivemodifi.ttf",
    6,
)
palette_size = 8
methods = [
    [
        "Quantize.MAXCOVERAGE",
        "Quantize.MEDIANCUT",
        "cv2.kmeans_RGB",
    ],
    [
        "cv2.kmeans_LAB",
        "sklearn.kmeans_LAB_deltaE00",
        "torch.kmedoids_LAB_deltaE00",
    ],
]

testing_image = Image.open("examples/tigerrobot.png").convert("RGB")
testing_image = scale_utils.oe_downscale(testing_image, 2, "nearest-neighbors")
testing_palette_image = Image.open("examples/retrotronic-dx-1x.png")

# Create a grid to display the images
grid_size = len(methods) + 1, len(methods[0])
grid_image = Image.new(
    "RGB", (testing_image.width * grid_size[1], testing_image.height * grid_size[0])
)

# Paste the original image to the grid
grid_image.paste(testing_image, (0, 0))

for x, y in product(range(1, len(methods) + 1), range(len(methods[0]))):
    curr_method = methods[x - 1][y]
    print(curr_method)

    quantized_image = quantization_utils.palette_quantization(
        image=testing_image, palette_size=palette_size, method=curr_method
    )
    img_draw = ImageDraw.Draw(quantized_image)
    img_draw.text((0, 0), curr_method, (0, 0, 0), font=font)
    grid_image.paste(
        quantized_image, (testing_image.width * y, testing_image.height * x)
    )

grid_image = grid_image.resize(
    (grid_image.width * 2, grid_image.height * 2), Image.Resampling.NEAREST
)
grid_image.show()
