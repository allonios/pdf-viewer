import warnings

import cv2
import numpy as np
from numba import NumbaDeprecationWarning, NumbaWarning
from numba import jit

from image_processors.base import BaseProcessor
from utils.utils import get_most_frequent_color, get_mask_from_color, \
    get_text_cleaned_background_mask, create_text_mask

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaWarning)

class BackgroundColorProcessor(BaseProcessor):
    def __init__(self, background_color, image=None):
        super().__init__(image)
        self.background_color = background_color

    @jit
    def set_background_color(self):
        background_color = get_most_frequent_color(self.image)

        new_image = self.image.copy()
        new_image[
            np.all(
                new_image == background_color,
                axis=-1
            )
        ] = self.background_color

        return new_image


    def process_image(self, image=None):
        self.image = super().process_image(image)
        return self.set_background_color()
