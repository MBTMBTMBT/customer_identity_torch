import numpy as np


class ImageWithMasksAndAttributes:
    def __init__(self, image: np.ndarray, masks: dict[str, np.ndarray], attributes: dict[str, float]):
        self.image: np.ndarray = image
        self.masks: dict[str, np.ndarray] = masks
        self.attributes: dict[str, float] = attributes

