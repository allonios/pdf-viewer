import cv2
import numpy as np
from pdf2image import convert_from_path
from scipy.interpolate import UnivariateSpline


class WarmingEffect():
    def __init__(self, image):
        self.image = image
        self.decrease_channel_lut = self._get_lut(
            x=[0, 64, 128, 192, 255],
            y=[0, 70, 140, 210, 255],
        )
        self.increase_channel_lut_red = self._get_lut(
            x=[0, 64, 128, 192, 255],
            y=[0, 30,  80, 120, 192],
        )
        self.increase_channel_lut_green = self._get_lut(
            x=[0, 64, 128, 192, 255],
            y=[0, 50,  100, 150, 220],
        )

    def _get_lut(self, x, y):
        spline_interpolate = UnivariateSpline(x, y)
        return spline_interpolate(range(256))


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


    def render_image(self, apply_increase_saturation=True):
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

        if apply_increase_saturation:
            return self.increase_saturation(image)
        else:
            return image


class BackGroundColorEditor():
    def __init__(self, image):
        self.image = image

    def set_background_color(self, rgb_color):
        lower_white = np.array([220, 220, 220], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(self.image, lower_white, upper_white)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (3, 3)
            )
        )
        cv2.imshow("mask1", mask)
        mask = cv2.bitwise_not(mask)
        cv2.imshow("mask2", mask)

        background = np.full(self.image.shape, rgb_color, np.uint8)

        foreground_mask = cv2.bitwise_and(self.image, self.image, mask=mask)

        mask = cv2.bitwise_not(mask)
        background_mask = cv2.bitwise_and(background, background, mask=mask)

        final = cv2.bitwise_or(foreground_mask, background_mask)
        mask = cv2.bitwise_not(mask)

        return final


# image = cv2.imread("test_files/test_image.jpg")

images = convert_from_path("test_files/part-4.pdf")

images = list(
    map(
        lambda image: np.asarray(image),
        images
    )
)


# w_e = WarmingEffect(images[0])
# # w_e = WarmingEffect(image)
# warm_image = w_e.render_image()
#
# cv2.imshow("Original Image", images[0])
# # cv2.imshow("Original Image", image)
# cv2.imshow("New Image", warm_image)
#
# cv2.waitKey()

bg = BackGroundColorEditor(images[0])
new_image = bg.set_background_color((125, 222, 0))

cv2.imshow("Original Image", images[0])
cv2.imshow("New Image", new_image)

cv2.waitKey()
#
