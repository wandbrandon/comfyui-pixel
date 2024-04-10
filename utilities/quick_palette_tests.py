import palette_swapping_utils
import scale_utils
from PIL import Image
from itertools import product
from PIL import ImageDraw
from PIL import ImageFont

font = ImageFont.truetype(
    "/Users/brandonwand/Documents/projects/comfyui-pixel/utilities/treebyfivemodifi.ttf",
    6,
)
palette_size = 6
methods = [
    ["Pillow Quantize", "CIELAB Delta E 20000"],
]

testing_image = Image.open("examples/tigerrobot.png").convert("RGB")
# testing_image = scale_utils.oe_downscale(testing_image, 4, "nearest-neighbors")
testing_palette_image = Image.open("examples/retrotronic-dx-1x.png")

# Create a grid to display the images
grid_size = len(methods), max([len(i) for i in methods])

grid_image = Image.new(
    "RGB", (testing_image.width * grid_size[1], testing_image.height * grid_size[0])
)

for x in range(len(methods)):
    for y in range(len(methods[x])):
        curr_method = methods[x][y]
        print(curr_method)

        quantized_image = palette_swapping_utils.palette_swap(
            image=testing_image, palette_image=testing_palette_image, method=curr_method
        )

        img_draw = ImageDraw.Draw(quantized_image)
        img_draw.text((1, 1), curr_method, (0, 0, 0), font=font)
        grid_image.paste(
            quantized_image, (testing_image.width * y, testing_image.height * x)
        )

final_img = Image.new(
    "RGB", (grid_image.width, grid_image.height + testing_image.height)
)
final_img.paste(testing_image, (0, 0))
final_img.paste(grid_image, (0, testing_image.height))

# double image size
final_img = final_img.resize((final_img.width * 3, final_img.height * 3), Image.NEAREST)
final_img.show()
