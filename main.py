import cv2
import numpy as np
from pdf2image import convert_from_path

from image_handlers.base import BaseImageHandler
from image_processors.callback_processor import CallbackProcessor
from image_processors.background_color_processor import \
    BackgroundColorProcessor
from image_processors.eye_comfort_processor import EyeComfort
from image_processors.text_color_processor import TextColorProcessor

images = convert_from_path("test_files/test.pdf")

images = list(
    map(
        lambda image: cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR),
        images
    )
)

image = images[1]

processors = [
    # BackgroundColorProcessor(
    #     ((0, 222, 125))
    # ),
    CallbackProcessor(
        lambda image: cv2.resize(
            image,
            None,
            fx=1.5,
            fy=1.5,
            interpolation=cv2.INTER_NEAREST
        )
    ),
    TextColorProcessor()
    # EyeComfort(),
]

handler = BaseImageHandler(
    processors=processors,
    init_image=image,
)

handler.register_windows_names(
    ["background_colored_mask", "background_cleaned_mask", "text_mask"]
    # ["Text Mask", "foreground", "color_mask"]
    # ["small_image_mask", "original_image_mask", "larg_image_mask", "larger_image_mask", ]
    # ["mask", "mask not", "foreground_mask", "background_mask"]
)

handler.run()
cv2.waitKey()