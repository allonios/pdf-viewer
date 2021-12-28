import numpy as np
from pdf2image import convert_from_path

from image_handlers.base import BaseImageHandler
from image_processors.background_color_processor import \
    BackgroundColorProcessor
from image_processors.eye_comfort_processor import EyeComfort


images = convert_from_path("test_files/part-4.pdf")

images = list(
    map(
        lambda image: np.asarray(image),
        images
    )
)

image = images[0]

processors = [
    # EyeComfort(),
    BackgroundColorProcessor(
        ((125, 222, 0))
    ),

]

handler = BaseImageHandler(
    processors=processors,
    init_image=image,
)

handler.run()