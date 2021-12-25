import cv2
import numpy as np
from pdf2image import convert_from_path

# Store Pdf with convert_from_path function


images = convert_from_path("test_files/resume.pdf")

images = list(
    map(
        lambda image: np.asarray(image),
        images
    )
)

def reduce_channel_by(channel, percentage=0.8):
    reduced_channel = np.zeros_like(channel)

    for row_index, row in enumerate(channel):
        for element_index, element in enumerate(row):
            reduced_channel[row_index][element_index] = element - element * percentage

    return reduced_channel

def increase_channel_by(channel, percentage=0.5):
    increased_channel = np.zeros_like(channel)

    for row_index, row in enumerate(channel):
        for element_index, element in enumerate(row):
            increased_channel[row_index][element_index] = (
                element + element * percentage if (
                        element + element * percentage < 255
                ) else 255
            )


    return increased_channel



image = images[0]
image = cv2.imread("test_files/test_image.jpg")

red_channel, green_channel, blue_channel = cv2.split(image)

updated_red_channel = increase_channel_by(red_channel, .3)
updated_green_channel = increase_channel_by(green_channel, .4)
updated_blue_channel = increase_channel_by(blue_channel)

new_image = cv2.merge(
    (
        updated_red_channel,
        updated_green_channel,
        updated_blue_channel
    )
)

cv2.imshow("Image", image)
cv2.imshow("New Image", new_image)

# for index, image in enumerate(images):
#     cv2.imshow(f"Image {index}", image)



cv2.waitKey()
