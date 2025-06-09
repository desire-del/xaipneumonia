from src.log import logger


class ImageExplainer:
    def __init__(self, model):
        self.model = model

    def explain(self, image_path):
        logger.error("The explain method must be implemented by subclasses.")
        raise NotImplementedError("This method should be implemented by subclasses.")
