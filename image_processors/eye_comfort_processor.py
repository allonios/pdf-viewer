import cv2
import numpy as np

from image_processors.base import BaseProcessor
from utils.utils import get_lut


class EyeComfort(BaseProcessor):
    def __init__(self, image=None, apply_increase_saturation=True):
        super().__init__(image)
        self.apply_increase_saturation = apply_increase_saturation
        self.decrease_channel_lut = get_lut(
            x=[0, 64, 128, 192, 255],
            y=[0, 70, 140, 210, 255],
        )
        self.increase_channel_lut_red = get_lut(
            x=[0, 64, 128, 192, 255],
            y=[0, 30, 80, 120, 192],
        )
        self.increase_channel_lut_green = get_lut(
            x=[0, 64, 128, 192, 255],
            y=[0, 50, 100, 150, 220],
        )

    def increase_saturation(self, image):
        hue_channel, saturation_channel, value_channel = cv2.split(
            cv2.cvtColor(
                image,
                cv2.COLOR_RGB2HSV
            )
        )

        updated_saturation_channel = cv2.LUT(
            saturation_channel,
            self.increase_channel_lut_red
        ).astype(np.uint8)


        return cv2.cvtColor(
            cv2.merge(
                (
                    hue_channel,
                    updated_saturation_channel,
                    value_channel
                )
            ),
            cv2.COLOR_HSV2RGB
        )

    def apply_eye_comfort(self):
        red_channel, green_channel, blue_channel = cv2.split(self.image)
        updated_red_channel = cv2.LUT(
            red_channel,
            self.increase_channel_lut_red
        ).astype(np.uint8)

        updated_green_channel = cv2.LUT(
            red_channel,
            self.increase_channel_lut_green
        ).astype(np.uint8)

        updated_blue_channel = cv2.LUT(
            blue_channel,
            self.decrease_channel_lut
        ).astype(np.uint8)

        image = cv2.merge(
            (updated_red_channel, updated_green_channel, updated_blue_channel)
        )

        if self.apply_increase_saturation:
            return self.increase_saturation(image)
        else:
            return image

    def process_image(self, image=None):
        self.image = super().process_image(image)
        return self.apply_eye_comfort()
