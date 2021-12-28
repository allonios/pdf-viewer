import cv2


class BaseImageHandler:
    def __init__(self, processors, init_image=None):
        self.processors = processors
        self.current_image = init_image

    def run_processors(self):
        for processor in self.processors:
            print(processor)
            self.current_image = processor(self.current_image)

    def run(self):
        self.run_processors()
        cv2.imshow("Image", self.current_image)
        cv2.waitKey()
