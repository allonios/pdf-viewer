# -*- coding: utf-8 -*-
import cv2
import numpy as np
from pdf2image import convert_from_path
from screeninfo import get_monitors
import enum
import json
from copy import deepcopy
import evdev
from IrisDetetion02 import detect_eye


class UserActivityMode(enum.Enum):
    book_mark = "BOOK_MARK"
    page_book_mark = "PAGE_BOOK_MARK"
    pass


class Viewer(object):
    def __init__(
        self,
        images,
        padding_height=0,
        window_name="PanZoomWindow",
        window_dim=[200, 300],
        on_left_click_function=None,
        settings=None
    ):
        self.WINDOW_NAME = window_name
        self.H_TRACKBAR_NAME = "y"
        self.V_TRACKBAR_NAME = "x"
        self.image_height = images[0].shape[0]
        self.images = images
        self.padding_height = padding_height
        self.window_dim = window_dim
        self.on_left_click_function = on_left_click_function
        # self.TRACKBAR_TICKS = 1000
        self.document = self.generate_full_document(images) 
        self.document_height = self.document.shape[0]
        self.document_width = self.document.shape[1]
        self.pan_state = PanState(self.document.shape, window_dim, self)
        self.left_button_down_loc = None
        self.mode = UserActivityMode.book_mark.value
        self.settings = settings if settings else {
            "book_marks": [],
            "page_book_marks": []
        }
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_KEEPRATIO)
        self.document = self.redraw_image()
        cv2.setMouseCallback(self.WINDOW_NAME, self.on_mouse)
        cv2.createTrackbar(
            self.H_TRACKBAR_NAME, self.WINDOW_NAME, 0, self.document_height, self.on_h_trackbar_move
        )
        cv2.createTrackbar(
            self.V_TRACKBAR_NAME, self.WINDOW_NAME, 0, self.document_width, self.on_v_trackbar_move
        )
        cv2.createButton(
            "Book Mark Mode", self.on_book_mark_button_click, None, cv2.QT_PUSH_BUTTON, 1)
        cv2.createButton("Page Book Mark Mode",
                         self.on_page_book_mark_button_click, None, cv2.QT_PUSH_BUTTON, 1)
        cv2.createButton("Save Settings", self.on_save_settings,
                         None, cv2.QT_PUSH_BUTTON, 1)

    def update_images(self, images):
        self.images = images
        self.redraw_image()

    def generate_full_document(self,images):
        padding_image = np.zeros((self.padding_height, self.images[0].shape[1], 3), dtype=np.uint8)        
        
        images_with_padding = []
        
        for image in images:
            images_with_padding.append(image)
            images_with_padding.append(padding_image)
        
        return cv2.vconcat(images_with_padding)

    def on_mouse(self, event, x, y, flags, params):
        # print(f"{flags=}")
        # print(f"{event=}")
        if event == cv2.EVENT_LBUTTONDOWN:
            coords_in_displayed_image = np.array([y, x])
            if np.any(coords_in_displayed_image < 0) or np.any(
                coords_in_displayed_image > self.window_dim
            ):
                print("you clicked outside the image area")
            else:
                coordsInFullImage = self.pan_state.ul + coords_in_displayed_image
                if self.mode == UserActivityMode.book_mark.value:
                    self.add_book_mark(
                        coordsInFullImage[0], coordsInFullImage[1])
                if self.mode == UserActivityMode.page_book_mark.value:
                    self.add_page_book_mark(
                        coordsInFullImage[0], coordsInFullImage[1])

                print(
                    f"you clicked on {coords_in_displayed_image} within the displayed image"
                )

                coordsInFullImage = self.pan_state.ul + coords_in_displayed_image

                print(f"this is {coordsInFullImage} in the actual image")

                print(
                    f"this pixel holds {self.document[coordsInFullImage[0], coordsInFullImage[1]]}"
                )

                if self.on_left_click_function is not None:
                    self.on_left_click_function(
                        coordsInFullImage[0], coordsInFullImage[1])
        # elif event == 10:
        #     #sign of the flag shows direction of mousewheel
        #     if flags > 0:
        #         print("Scrolling Top")
        #         self.on_v_trackbar_move(100)
        #     else:
        #         print("Scrolling Down")
        #         self.on_v_trackbar_move(-100)
        #         #scroll down
        # elif event == 11:
        #     #sign of the flag shows direction of mousewheel
        #     if flags > 0:
        #         print("Scrolling Left")
        #         self.on_h_trackbar_move(100)
        #     else:
        #         print("Scrolling Right")
        #         self.on_h_trackbar_move(-100)
        #         #scroll down

    def get_images_coordinates(self, y, x):
        image_index = int(y / (self.image_height + self.padding_height))
        image_coordinates = (x, y - image_index * \
            (self.image_height + self.padding_height))
        is_on_image = image_coordinates[0] <= self.image_height
        print(self.image_height,self.padding_height)
        return is_on_image, image_index, image_coordinates

    def on_v_trackbar_move(self, tick_position):
        self.pan_state.set_y_fraction_offset(
            float(tick_position)/self.document_width)
        # print("Scrolling Heere V ", scroll_value)
        # self.pan_state.set_y_fraction_offset(float(scroll_value))

    def on_h_trackbar_move(self, tick_position):
       self.pan_state.set_x_fraction_offset(
            float(tick_position)/self.document_height)
            
    def redraw_image(self):
        images = deepcopy(self.images)
        for book_mark in self.settings["book_marks"]:
            cv2.circle(images[book_mark[0]], book_mark[1:], 10, [0, 0, 255], 10)

        for page_book_mark in self.settings["page_book_marks"]:
            cv2.line(images[page_book_mark], (0,0), (0, images[page_book_mark].shape[0]), (0,0,255), 5)
            cv2.line(images[page_book_mark], (0,0), (images[page_book_mark].shape[1], 0), (0,0,255), 5)
            cv2.line(images[page_book_mark], (images[page_book_mark].shape[1],0), (images[page_book_mark].shape[1], images[page_book_mark].shape[0]), (0,0,255), 5)
            cv2.line(images[page_book_mark], (0, images[page_book_mark].shape[0]), (images[page_book_mark].shape[1], images[page_book_mark].shape[0]), (0,0,255), 5)

        self.document = self.generate_full_document(images)

        cv2.imshow(
            self.WINDOW_NAME,
            self.document[
                self.pan_state.ul[0]: self.pan_state.ul[0] + self.window_dim[0],
                self.pan_state.ul[1]: self.pan_state.ul[1] + self.window_dim[1],
            ],
        )
        return self.document

    def on_book_mark_button_click(self, _, _1):
        self.mode = UserActivityMode.book_mark.value

    def on_page_book_mark_button_click(self, _, _1):
        self.mode = UserActivityMode.page_book_mark.value

    def on_save_settings(self, _, _1):
        with open("settings.json", "w") as outfile:
            json.dump(self.settings, outfile)

    def add_book_mark(self, x, y):
        is_on_image, image_index, image_coord = self.get_images_coordinates(x, y)
        if is_on_image:
            self.settings["book_marks"].append([int(image_index),int(image_coord[0]), int(image_coord[1])])
            self.redraw_image()

    def add_page_book_mark(self, x, y):
        is_on_image, image_index, image_coord = self.get_images_coordinates(x, y)
        if is_on_image:
            print(list(filter(lambda index: index == image_index,self.settings["page_book_marks"])))
            if(len(list(filter(lambda index: index == image_index,self.settings["page_book_marks"]))) > 0):
                self.settings["page_book_marks"] = list(filter(lambda index: index != image_index, self.settings["page_book_marks"]))
            else:
                self.settings["page_book_marks"].append(int(image_index))
        self.redraw_image()


class PanState(object):
    def __init__(self, im_shape, window_dim, parent_window):
        # upper left of the rectangle (expressed as y,x)
        self.ul = np.array([0, 0])
        self.window_dim = window_dim
        self.document_shape = np.array(im_shape[0:2])
        self.parent_window = parent_window

    def _fix_bounds_and_draw(self):
        """Ensures we didn't scroll outside the image."""
        print(self.ul)
        self.ul = np.maximum(0, np.minimum(
            self.ul, (self.document_shape - self.window_dim)))
        self.parent_window.redraw_image()

    def set_y_fraction_offset(self, fraction):
        self.ul[0] = int(
            round((self.document_shape[0] - self.window_dim[0]) * fraction))
        self._fix_bounds_and_draw()

    def set_x_fraction_offset(self, fraction):
        self.ul[1] = int(
            round((self.document_shape[1] - self.window_dim[1]) * fraction))
        self._fix_bounds_and_draw()


if __name__ == "__main__":
    images = convert_from_path("test_files/resume.pdf")
    json_settings = None

    try:
        with open('settings.json', 'r') as file:
            json_settings = file.read()
    except FileNotFoundError:
        print("File Not Found")

    settings = json.loads(json_settings) if json_settings else None
    print(settings)
    images = list(map(lambda image: np.asarray(image), images))

    images_with_padding = []

    monitor = get_monitors()[0]

    window = Viewer(
        images=images,
        padding_height=50,
        window_name="test window",
        window_dim=[monitor.height - 250, monitor.width - 100],
        settings=settings
    )
    key = -1


    while (
        key != ord("q")
        and key != 27
        and cv2.getWindowProperty(window.WINDOW_NAME, 0) >= 0
    ):
        detect_eye(window)
        key = cv2.waitKey(5)  # User can press 'q' or ESC to exit.
    cv2.destroyAllWindows()
