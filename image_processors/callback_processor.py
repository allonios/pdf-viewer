from image_processors.base import BaseProcessor


class CallbackProcessor(BaseProcessor):
    def __init__(self, callback, image=None):
        super().__init__(image)
        self.callback = callback

    def process_image(self, image=None):
        self.image = super().process_image(image)
        return self.callback(self.image)
