from tkinter import (
    Tk,
    Frame,
    Button,
    StringVar,
    IntVar,
    Label,
    OptionMenu,
    Entry, Checkbutton
)
from tkinter.filedialog import askopenfilename

import cv2
import os
import json
import numpy as np
from pdf2image import convert_from_path

from image_handlers.base import BaseImageHandler
from image_processors.background_color_processor import (
    BackgroundColorProcessor
)
from image_processors.callback_processor import CallbackProcessor
from image_processors.eye_comfort_processor import EyeComfort
from image_processors.text_color_processor import TextColorProcessor


cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
# cv2.namedWindow("Result1", cv2.WINDOW_NORMAL)

class PDFViewer():
    def __init_tools_frame(self, frame_parent):
        self.tools_frame = Frame(frame_parent)



    def __init__(self):
        with open("run_settings.json", "r") as file:
            self.global_settings = json.loads(
                file.read()
            )

        self.file_dir = ""

        self.window = Tk()
        self.window.title("PDF Viewer")
        self.window.geometry("500x500")

        # styles
        self.font = "Roboto Regular"
        self.fontSize = 10

        self.open_file_button = Button(
            self.window,
            width=25,
            text="Open PDF File",
            command=self.load_pdf_file
        )
        self.open_file_button.pack(expand=True)

        self.window.mainloop()


    def load_pdf_file(self):
        filetypes = (
            ("pdf files", ".pdf"),
        )

        self.file_dir = askopenfilename(
            title="Open a file",
            initialdir="~/PycharmProjects/pdf_viewer/test_files",
            filetypes=filetypes,
        )

        self.images = list(
            map(
                lambda image: cv2.cvtColor(
                    np.asarray(image),
                    cv2.COLOR_RGB2BGR
                ),
                convert_from_path(self.file_dir)
            )
        )
        self.window.destroy()

        self.settings_view = SettingsView(
            self.images,
            os.path.basename(self.file_dir),
            self.global_settings
        )


class SettingsView():
    def __init__(self, images, file_name, settings=None):
        self.images = images
        self.images_numbers_labels = list(
            map(
                lambda number: str(number),
                range(len(images))
            )
        )
        self.images_shape = images[0].shape

        self.file_name = file_name

        self.settings = {}
        if settings:
            self.settings = settings

        if not self.settings.get(file_name, False):
            self.settings[file_name] = []

        print(self.settings)


        # init window
        self.window = Tk()
        self.window.title("PDF Viewer")
        self.window.geometry("500x550")


        # styles
        self.font = "Roboto Regular"
        self.fontSize = 10


        # gui_data
        self.page_number = StringVar()
        self.page_number.set("0")
        self.eye_comfort = IntVar()
        self.text_color = StringVar()
        self.background_color = StringVar()


        # gui elements
        self.select_page_label = Label(
            self.window,
            width=25,
            text="Select Page",
            font=(self.font, self.fontSize),
        )
        self.page_selector = OptionMenu(
            self.window,
            self.page_number,
            *self.images_numbers_labels
        )

        self.eye_comfort_checkbox = Checkbutton(
            self.window,
            text="Eye Comfort",
            width=30,
            font=(self.font, self.fontSize),
            variable=self.eye_comfort
        )

        self.text_color_label = Label(
            self.window,
            width=40,
            text="Text Color (Comma Separated BGR)",
            font=(self.font, self.fontSize),
        )
        self.text_color_entry = Entry(self.window)

        self.background_color_label = Label(
            self.window,
            width=40,
            text="Background Color (Comma Separated BGR)",
            font=(self.font, self.fontSize),
        )
        self.background_color_entry = Entry(self.window)

        self.add_button = Button(
            self.window,
            width=25,
            text="Add",
            command=self.add_settings_element
        )

        self.clear_button = Button(
            self.window,
            width=25,
            text="Clear",
            command=self.clear_settings
        )

        self.show_button = Button(
            self.window,
            width=25,
            text="Show",
            command=self.show
        )

        self.save_button = Button(
            self.window,
            width=25,
            text="Save",
            command=self.save
        )

        # pack gui elements
        self.select_page_label.pack(padx=10, pady=10)
        self.page_selector.pack(padx=10, pady=10)
        self.eye_comfort_checkbox.pack(padx=10, pady=10)
        self.text_color_label.pack(padx=10, pady=10)
        self.text_color_entry.pack(padx=10, pady=10)
        self.background_color_label.pack(padx=10, pady=10)
        self.background_color_entry.pack(padx=10, pady=10)
        self.add_button.pack(padx=10, pady=10)
        self.clear_button.pack(padx=10, pady=10)
        self.show_button.pack(padx=10, pady=10)
        self.save_button.pack(padx=10, pady=10)

        self.window.mainloop()

    def add_settings_element(self):
        page_number = int(self.page_number.get())
        eye_comfort = bool(self.eye_comfort.get())
        text_color = []
        background_color = []

        if self.text_color_entry.get():
            text_color = self.text_color_entry.get().split(",")
        if self.background_color_entry.get():
            background_color = self.background_color_entry.get().split(",")

        self.settings[self.file_name].append(
            {
                "page_number": page_number,
                "eye_comfort": eye_comfort,
                "text_color": text_color,
                "background_color": background_color,
            }
        )
        print(self.settings)

    def clear_settings(self):
        self.settings[self.file_name].clear()

    def get_image_handler(self, settings_element):
        processors = []
        if settings_element["background_color"]:
            processors.append(
                BackgroundColorProcessor(
                    settings_element["background_color"]
                )
            )

        if settings_element["text_color"]:
            processors.append(
                CallbackProcessor(
                    lambda image: cv2.resize(
                        image,
                        None,
                        fx=1.5,
                        fy=1.5,
                        interpolation=cv2.INTER_LANCZOS4
                    )
                ),
            )
            processors.append(
                TextColorProcessor(settings_element["text_color"])
            )
            processors.append(
                CallbackProcessor(
                    lambda image: cv2.resize(
                        image,
                        # None,
                        (self.images_shape[1], self.images_shape[0]),
                        interpolation=cv2.INTER_LANCZOS4
                    )
                ),
            )

        if settings_element["eye_comfort"]:
            processors.append(EyeComfort())

        return BaseImageHandler(
            processors=processors,
            init_image=self.images[settings_element["page_number"]],
            # registered_windows_names=[str(settings_element["page_number"])]
            registered_windows_names=[]
        )

    def show(self):
        show_images = []
        images_to_edit_indexes = set(
            map(
                lambda settings_element: settings_element["page_number"],
                self.settings[self.file_name]
            )
        )

        for image_index, image in enumerate(self.images):
            if image_index in images_to_edit_indexes:
                handler = self.get_image_handler(
                    list(
                        filter(
                            lambda settings_element: (
                                    settings_element[
                                        "page_number"] == image_index
                            ),
                            self.settings[self.file_name]
                        )
                    )[0]
                )
                show_images.append(handler.run())
            # else:
            #     show_images.append(image)

        # for image_index, image in enumerate(show_images):
        #     cv2.imshow(str(image_index), image)

        # cv2.waitKey()

    def save(self):
        with open("run_settings.json", "w") as file:
            file.write(json.dumps(self.settings, indent=4))


if __name__ == "__main__":
    pdf = PDFViewer()
