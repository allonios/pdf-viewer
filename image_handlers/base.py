import cv2


class BaseImageHandler:
    def __init__(
            self,
            processors,
            init_image=None,
            registered_windows_names=("Original Image", "Result")
    ):
        self.processors = processors
        self.current_image = init_image
        self.registered_windows_names = list(registered_windows_names)
        for window_name in self.registered_windows_names:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def register_windows_names(self, windows_names):
        self.registered_windows_names.extend(windows_names)
        for window_name in windows_names:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def run_processors(self):
        for processor in self.processors:
            print(processor)
            self.current_image = processor(self.current_image)

    def run(self):
        # cv2.imshow("Original Image", self.current_image)
        self.run_processors()
        # cv2.imshow("Result", self.current_image)
        # cv2.waitKey()
        return self.current_image
