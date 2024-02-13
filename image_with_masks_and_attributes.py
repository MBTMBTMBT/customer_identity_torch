import numpy as np
from categories_and_attributes import CategoriesAndAttributes


def _softmax(x: list[float]) -> list[float]:
    """Compute softmax values for each set of scores in x without using NumPy."""
    # First, compute e^x for each value in x
    exp_values = [_exp(val) for val in x]
    # Compute the sum of all e^x values
    sum_of_exp_values = sum(exp_values)
    # Now compute the softmax value for each original value in x
    softmax_values = [exp_val / sum_of_exp_values for exp_val in exp_values]
    return softmax_values


def _exp(x):
    """Compute e^x for a given x. A simple implementation of the exponential function."""
    return 2.718281828459045 ** x  # Using an approximation of Euler's number e


class ImageWithMasksAndAttributes:
    def __init__(self, image: np.ndarray, masks: dict[str, np.ndarray], attributes: dict[str, float],
                 categories_and_attributes: CategoriesAndAttributes):
        self.image: np.ndarray = image
        self.masks: dict[str, np.ndarray] = masks
        self.attributes: dict[str, float] = attributes
        self.categories_and_attributes: CategoriesAndAttributes = categories_and_attributes

        self.plane_attribute_dict: dict[str, float] = {}
        for attribute in self.categories_and_attributes.plane_attributes:
            self.plane_attribute_dict[attribute] = self.attributes[attribute]

        self.selective_attribute_dict: dict[str, dict[str, float]] = {}
        for category in self.categories_and_attributes.selective_categories.keys():
            self.selective_attribute_dict[category] = {}
            temp_list: list[float] = []
            for attribute in self.categories_and_attributes.selective_categories[category]:
                temp_list.append(self.attributes[attribute])
            softmax_list = _softmax(temp_list)
            for i, attribute in enumerate(self.categories_and_attributes.selective_categories[category]):
                self.selective_attribute_dict[category][attribute] = softmax_list[i]


def _max_value_tuple(some_dict: dict[str, float]) ->tuple[str, float]:
    max_key = max(some_dict, key=some_dict.get)
    return max_key, some_dict[max_key]


class ImageOfPerson(ImageWithMasksAndAttributes):
    def __init__(self, image: np.ndarray, masks: dict[str, np.ndarray], attributes: dict[str, float],
                 categories_and_attributes: CategoriesAndAttributes):
        super().__init__(image, masks, attributes, categories_and_attributes)

    def describe(self, return_rate=False):
        gender = ('man', self.attributes['Male']) if self.attributes['Male'] > self.categories_and_attributes.thresholds_pred['Male'] else ('woman', 1-self.attributes['Male'])
        hair_colour = _max_value_tuple(self.selective_attribute_dict['hair_colour'])
        hair_shape = _max_value_tuple(self.selective_attribute_dict['hair_shape'])
        facial_hair = _max_value_tuple(self.selective_attribute_dict['facial_hair'])
        bangs = (True, self.attributes['Bangs']) if self.attributes['Bangs'] > self.categories_and_attributes.thresholds_pred['Bangs'] else (False, self.attributes['Bangs'])
        glasses = (True, self.attributes['Eyeglasses']) if self.attributes['Eyeglasses'] > self.categories_and_attributes.thresholds_pred['Eyeglasses'] else (False, self.attributes['Eyeglasses'])
        earrings = (True, self.attributes['Wearing_Earrings']) if self.attributes['Wearing_Earrings'] > self.categories_and_attributes.thresholds_pred['Wearing_Earrings'] else (False, self.attributes['Wearing_Earrings'])
        necklace = (True, self.attributes['Wearing_Necklace']) if self.attributes['Wearing_Necklace'] > self.categories_and_attributes.thresholds_pred['Wearing_Necklace'] else (False, self.attributes['Wearing_Necklace'])
        necktie = (True, self.attributes['Wearing_Necktie']) if self.attributes['Wearing_Necktie'] > self.categories_and_attributes.thresholds_pred['Wearing_Necktie'] else (False, self.attributes['Wearing_Necktie'])
        describe_hair = 0

        description = ""
