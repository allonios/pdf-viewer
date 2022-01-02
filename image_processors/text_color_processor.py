import cv2
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from scipy.spatial.distance import euclidean

from image_processors.base import BaseProcessor
from skimage.morphology import disk

from utils.utils import (
    apply_mask,
    create_text_mask,
    get_most_none_background_frequent_color, get_color_difference
)


class TextColorProcessor(BaseProcessor):

    def __init__(self, text_color=(0, 222, 125), image=None):
        super().__init__(image)
        self.text_color = text_color

    @jit
    def detect_text(self):
        text_mask_5 = create_text_mask(self.image)
        text_mask_15 = create_text_mask(self.image, disk_radius=15)

        text_mask = cv2.bitwise_or(text_mask_5, text_mask_15)

        main_font_color, color_mask = get_most_none_background_frequent_color(
            self.image, text_mask
        )

        height, width, _ = color_mask.shape

        for i in range(height):
            for j in range(width):
                pixel = color_mask[i, j]
                if not (
                        ((pixel[0] == -1) or (pixel[1] == -1) or (
                                pixel[2] == -1))
                ):
                    if get_color_difference(
                        color_mask[i, j],
                        main_font_color
                    ) < 37:
                        color_mask[i, j] = self.text_color
                elif ((pixel[0] == -1) or (pixel[1] == -1) or (pixel[2] == -1)):
                    color_mask[i, j] = np.array([0, 0, 0])

        color_mask = color_mask.astype("uint8")

        cv2.imshow("color_mask", color_mask)

        return apply_mask(color_mask, self.image, text_mask)

    def process_image(self, image=None):
        self.image = super().process_image(image)
        return self.detect_text()
