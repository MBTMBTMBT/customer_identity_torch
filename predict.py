import torch.nn
import os
from PIL import Image

from models import *
from datasets import *
from utils import *

from image_with_masks_and_attributes import ImageWithMasksAndAttributes


def read_images(path: str) -> tuple[list[np.ndarray], list[str]]:
    images_list = []
    path_list = []
    for filename in os.listdir(path):
        if filename.endswith(("jpg", "jpeg")):
            with Image.open(os.path.join(path, filename)) as img:
                rgb_image = np.array(img.convert('RGB'))
                images_list.append(rgb_image)
                path_list.append(os.path.join(path, filename))
    return images_list, path_list


class Predictor:
    def __init__(self, model: torch.nn.Module, device: torch.device, categories_and_attributes: CategoriesAndAttributes):
        self.model = model
        self.device = device
        self.categories_and_attributes = categories_and_attributes

        self._thresholds_mask: list[float] = []
        self._thresholds_pred: list[float] = []
        for key in sorted(list(self.categories_and_attributes.merged_categories.keys())):
            self._thresholds_mask.append(self.categories_and_attributes.thresholds_mask[key])
        for attribute in self.categories_and_attributes.attributes:
            if attribute not in self.categories_and_attributes.avoided_attributes:
                self._thresholds_pred.append(self.categories_and_attributes.thresholds_pred[attribute])

    def predict(self, rgb_image: np.ndarray) -> ImageWithMasksAndAttributes:
        image_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        pred_masks, pred_classes = self.model(image_tensor)
        # Apply binary erosion and dilation to the masks
        pred_masks = binary_erosion_dilation(
            pred_masks, thresholds=self._thresholds_mask,
            erosion_iterations=1, dilation_iterations=1
        )
        pred_masks = pred_masks.detach().squeeze(0).numpy().astype(np.uint8)
        mask_list = [pred_masks[i, :, :] for i in range(pred_masks.shape[0])]
        pred_classes = pred_classes.detach().squeeze(0).numpy()
        class_list = [pred_classes[i].item() for i in range(pred_classes.shape[0])]
        mask_dict = {}
        for i, mask in enumerate(mask_list):
            mask_dict[self.categories_and_attributes.mask_categories[i]] = mask
        attribute_dict = {}
        class_list_iter = class_list.__iter__()
        for attribute in self.categories_and_attributes.attributes:
            if attribute not in self.categories_and_attributes.avoided_attributes:
                attribute_dict[attribute] = class_list_iter.__next__()
        for attribute in self.categories_and_attributes.mask_labels:
            attribute_dict[attribute] = class_list_iter.__next__()
        image_obj = ImageWithMasksAndAttributes(rgb_image, mask_dict, attribute_dict, self.categories_and_attributes)
        return image_obj


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    categories_and_attributes = CelebAMaskHQCategoriesAndAttributes
    cat_layers = CelebAMaskHQCategoriesAndAttributes.merged_categories.keys().__len__()
    segment_model = UNetWithResnet18Encoder(num_classes=cat_layers)
    predictions = len(CelebAMaskHQCategoriesAndAttributes.attributes) - len(
        CelebAMaskHQCategoriesAndAttributes.avoided_attributes) + len(CelebAMaskHQCategoriesAndAttributes.mask_labels)
    predict_model = MultiLabelResNet(num_labels=predictions, input_channels=cat_layers + 3)
    model = CombinedModelNoRegression(segment_model, predict_model, cat_layers=cat_layers)
    model.eval()
    test_path = './test_images'
    p = Predictor(model, device, categories_and_attributes)
    images_list, path_list = read_images(test_path)
    for img, path in zip(images_list, path_list):
        rst = p.predict(img)
        print()
        print(path)
        print(rst.attributes)
