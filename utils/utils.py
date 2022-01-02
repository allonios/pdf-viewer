import cv2
import numpy as np
from numba import jit
from pyciede2000 import ciede2000
from scipy.interpolate import UnivariateSpline
from skimage.morphology import disk


def get_lut(x, y):
    spline_interpolate = UnivariateSpline(x, y)
    return spline_interpolate(range(256))


@jit
def create_text_mask(image, disk_radius=5):
    _, binary_image = cv2.threshold(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    binary_image = cv2.bitwise_not(binary_image)

    blue_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    red_channel = image[:, :, 2]

    y = red_channel * 0.299 + green_channel * 0.587 + blue_channel * 0.114
    y = y.astype("uint8")

    structuring_element = disk(disk_radius)

    # open = cv2.morphologyEx(y, cv2.MORPH_OPEN, disk(5))
    open = cv2.morphologyEx(y, cv2.MORPH_OPEN, structuring_element)
    # close = cv2.morphologyEx(y, cv2.MORPH_CLOSE, disk(5))
    close = cv2.morphologyEx(y, cv2.MORPH_CLOSE, structuring_element)
    diff = close - open

    kernel = np.ones((5, 5), np.float32) / 25
    low_pass_smoothing = cv2.filter2D(diff, -1, kernel)

    _, otsu_thresholding = cv2.threshold(
        low_pass_smoothing,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    text_mask = cv2.filter2D(otsu_thresholding, -1, kernel)

    return cv2.bitwise_and(text_mask, binary_image)


def apply_mask(foreground, background, mask):
    foreground_mask = cv2.bitwise_and(foreground, foreground, mask=mask)

    mask = cv2.bitwise_not(mask)
    background_mask = cv2.bitwise_and(background, background, mask=mask)

    final = cv2.bitwise_or(foreground_mask, background_mask)

    return final


def get_most_none_background_frequent_color(image, text_mask):
    background_mask = np.where(
        text_mask > 0, 0, -256
    )

    color_mask = np.add(
        image,
        np.dstack(
            [
                background_mask,
                background_mask,
                background_mask,
            ]
        )
    )

    color_mask = np.where(
        color_mask < 0, -1, color_mask
    )

    unique, counts = np.unique(
        color_mask.reshape(-1, 3),
        axis=0,
        return_counts=True
    )

    greatest_count = np.max(counts)
    greatest_count_index = np.argwhere(counts == greatest_count)[0][0]

    counts = counts[counts != greatest_count]
    unique = np.delete(unique, greatest_count_index, axis=0)

    return unique[np.argmax(counts)], color_mask


def get_most_frequent_color(image):
    unique, counts = np.unique(
        image.reshape(-1, 3),
        axis=0,
        return_counts=True
    )
    return unique[np.argmax(counts)]


def rgb2lab(input_color):
    num = 0
    RGB = [0, 0, 0]

    for value in input_color:
        value = float(value) / 255

        if value > 0.04045:
            value = ((value + 0.055) / 1.055) ** 2.4
        else:
            value = value / 12.92

        RGB[num] = value * 100
        num = num + 1

    XYZ = [0, 0, 0, ]

    X = RGB[0] * 0.4124 + RGB[1] * 0.3576 + RGB[2] * 0.1805
    Y = RGB[0] * 0.2126 + RGB[1] * 0.7152 + RGB[2] * 0.0722
    Z = RGB[0] * 0.0193 + RGB[1] * 0.1192 + RGB[2] * 0.9505
    XYZ[0] = round(X, 4)
    XYZ[1] = round(Y, 4)
    XYZ[2] = round(Z, 4)

    XYZ[0] = float(
        XYZ[0]) / 95.047  # ref_X =  95.047   Observer= 2°, Illuminant= D65
    XYZ[1] = float(XYZ[1]) / 100.0  # ref_Y = 100.000
    XYZ[2] = float(XYZ[2]) / 108.883  # ref_Z = 108.883

    num = 0
    for value in XYZ:

        if value > 0.008856:
            value = value ** (0.3333333333333333)
        else:
            value = (7.787 * value) + (16 / 116)

        XYZ[num] = value
        num = num + 1

    lab = [0, 0, 0]

    L = (116 * XYZ[1]) - 16
    a = 500 * (XYZ[0] - XYZ[1])
    b = 200 * (XYZ[1] - XYZ[2])

    lab[0] = round(L, 4)
    lab[1] = round(a, 4)
    lab[2] = round(b, 4)

    return lab


def get_color_difference(color1, color2):
    lab1 = rgb2lab(color1)
    lab2 = rgb2lab(color2)

    return ciede2000(tuple(lab1), tuple(lab2))["delta_E_00"]


def get_mask_from_color(image, existing_color, new_color):
    return np.apply_along_axis(
        lambda pixel: 255 if all(
            np.equal(pixel, existing_color)
        ) else 0,
        2,
        image
    ).astype("uint8")

def get_text_cleaned_background_mask(background_mask, text_mask):
    return cv2.bitwise_and(
        background_mask,
        background_mask,
        mask=cv2.bitwise_not(text_mask)
    )
