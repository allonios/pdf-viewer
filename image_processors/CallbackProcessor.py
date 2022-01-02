from image_processors.base import BaseProcessor


class CallBackProcessor(BaseProcessor):
    def __init__(self, image=None):
        super().__init__(image)