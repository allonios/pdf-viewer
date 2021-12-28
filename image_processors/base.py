from abc import ABCMeta


class BaseProcessor(metaclass=ABCMeta):

    def __init__(self, image=None):
        self.image = image

    def __call__(self, *args, **kwargs):
        return self.process_image(*args, **kwargs)

    def process_image(self, image=None):
        if image is not None:
            self.image = image
        return self.image
