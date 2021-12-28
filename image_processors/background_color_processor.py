import cv2
import numpy as np

from image_processors.base import BaseProcessor


class BackgroundColorProcessor(BaseProcessor):
    def __init__(self, background_color, image=None):
        super().__init__(image)
        self.background_color = background_color

    def set_background_color(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        mask = cv2.adaptiveThreshold(
            gray_image,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (3, 3)
            ),
            iterations=1
        )

        cv2.imshow("mask1", mask)
        mask = cv2.bitwise_not(mask)
        cv2.imshow("mask2", mask)

        background = np.full(self.image.shape, self.background_color, np.uint8)

        foreground_mask = cv2.bitwise_and(self.image, self.image, mask=mask)

        mask = cv2.bitwise_not(mask)
        background_mask = cv2.bitwise_and(background, background, mask=mask)

        final = cv2.bitwise_or(foreground_mask, background_mask)
        mask = cv2.bitwise_not(mask)

        return final

    def process_image(self, image=None):
        self.image = super().process_image(image)
        return self.set_background_color()






