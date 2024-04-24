import os
import random
import imageio.v2 as imageio
import scipy
from PIL import Image, ImageChops, ImageFilter, ImageEnhance, ImageOps, ImageDraw
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
import re
import glob
from image_with_masks_and_attributes import ImageWithMasksAndAttributes
from categories_and_attributes import CategoriesAndAttributes, CelebAMaskHQCategoriesAndAttributes
import json
import matplotlib.patches as patches
from torchvision.transforms import functional as TF
from matplotlib.patches import Rectangle


# class CategoryType(enum.Enum):
#     selective = 0
#     logical = 1


# class DeepFashion2Dataset(Dataset):
#     categories = [
#         'short sleeve top', 'long sleeve top', 'short sleeve outwear',
#         'long sleeve outwear', 'vest', 'sling', 'shorts',
#         'trousers', 'skirt', 'short sleeve dress',
#         'long sleeve dress', 'vest dress', 'sling dress'
#     ]
#     def __init__(self, image_dir, anno_dir, output_size=(300, 400)):
#         self.output_size = (output_size[1], output_size[0])  # input: W, H
#         self.image_dir = image_dir
#         self.anno_dir = anno_dir
#         self.image_filenames = [x for x in os.listdir(image_dir) if x.endswith('.jpg')]
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),  # Converts to Torch tensor and scales to [0, 1]
#         ])
#
#     def __len__(self):
#         return len(self.image_filenames)
#
#     def __getitem__(self, idx):
#         # Load and transform image
#         img_name = self.image_filenames[idx]
#         img_path = os.path.join(self.image_dir, img_name)
#         image = Image.open(img_path).convert('RGB')
#
#         # Determine the crop size based on the aspect ratio of output_size
#         crop_height, crop_width = self.get_max_crop(image.size, self.output_size)
#
#         # Initialize mask array
#         masks = torch.zeros((len(self.categories), *self.output_size), dtype=torch.float32)
#         # labels = torch.zeros(len(self.categories), dtype=torch.float32)
#
#         # Load annotation
#         anno_path = os.path.join(self.anno_dir, img_name.replace('.jpg', '.json'))
#         with open(anno_path, 'r', encoding='utf-8') as file:
#             anno_data = json.load(file)
#
#         # Create masks and determine labels
#         for item_key, item_value in anno_data.items():
#             if item_key.startswith('item'):
#                 category_id = item_value.get('category_id', 0) - 1
#                 if 0 <= category_id < len(self.categories):
#                     mask = Image.new('L', image.size, 0)
#                     draw = ImageDraw.Draw(mask)
#                     for segment in item_value.get('segmentation', []):
#                         draw.polygon(segment, fill=255)
#                     mask = TF.center_crop(mask, (crop_height, crop_width))
#                     mask = TF.resize(mask, self.output_size)
#                     mask_tensor = self.transform(mask).squeeze(0)  # Convert to tensor and remove channel dim
#                     masks[category_id] = torch.max(masks[category_id], mask_tensor)  # Use torch.max to combine masks
#
#         # Crop the image
#         image = TF.center_crop(image, (crop_height, crop_width))
#
#         # Resize image to output size
#         image = TF.resize(image, self.output_size)
#         image_tensor = self.transform(image)
#
#         # Update labels based on mask presence after cropping
#         labels = (torch.sum(masks.reshape(len(self.categories), -1), dim=1) > 0).float()
#
#         return image_tensor, masks, labels
#
#     @staticmethod
#     def get_max_crop(current_size, target_ratio):
#         current_ratio = current_size[0] / current_size[1]
#         target_ratio = target_ratio[0] / target_ratio[1]
#         if current_ratio > target_ratio:
#             # Crop width to fit target ratio
#             return int(current_size[1] * target_ratio), current_size[1]
#         else:
#             # Crop height to fit target ratio
#             return current_size[0], int(current_size[0] / target_ratio)


class DeepFashion2Dataset(Dataset):
    categories = [
        'short sleeve top', 'long sleeve top', 'short sleeve outwear',
        'long sleeve outwear', 'vest', 'sling', 'shorts',
        'trousers', 'skirt', 'short sleeve dress',
        'long sleeve dress', 'vest dress', 'sling dress'
    ]

    def __init__(self, image_dir, anno_dir, output_size=(300, 400)):
        self.output_size = output_size  # input: W, H
        self.image_dir = image_dir
        self.anno_dir = anno_dir
        self.image_filenames = [x for x in os.listdir(image_dir) if x.endswith('.jpg')]
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to Torch tensor and scales to [0, 1]
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load and transform image
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # Determine the crop size based on the aspect ratio of output_size
        crop_width, crop_height = self.get_max_crop(image.size, self.output_size)

        # Load annotation
        anno_path = os.path.join(self.anno_dir, img_name.replace('.jpg', '.json'))
        with open(anno_path, 'r', encoding='utf-8') as file:
            anno_data = json.load(file)

        # Initialize mask array and bbox list
        masks = torch.zeros((len(self.categories), self.output_size[1], self.output_size[0]), dtype=torch.float32)
        bboxes = []

        # Process each item in the annotation
        for item_key, item_value in anno_data.items():
            if item_key.startswith('item'):
                category_id = item_value.get('category_id', 0) - 1
                bbox = item_value.get('bounding_box', [])
                if 0 <= category_id < len(self.categories):
                    # Create mask for current item
                    mask = Image.new('L', image.size, 0)
                    draw = ImageDraw.Draw(mask)
                    for segment in item_value.get('segmentation', []):
                        draw.polygon(segment, fill=255)
                    mask = TF.center_crop(mask, (crop_height, crop_width))
                    mask = TF.resize(mask, (self.output_size[1], self.output_size[0]))
                    mask_tensor = self.transform(mask).squeeze(0)  # Convert to tensor and remove channel dim
                    masks[category_id] = torch.max(masks[category_id], mask_tensor)  # Use torch.max to combine masks

                    # Convert absolute bbox coordinates to relative based on crop and resize
                    rel_bbox = self.convert_bbox(bbox, image.size, (crop_width, crop_height), self.output_size)
                    bboxes.append((category_id, rel_bbox))

        # Crop the image
        image = TF.center_crop(image, (crop_height, crop_width))
        image = TF.resize(image, (self.output_size[1], self.output_size[0]))
        image_tensor = self.transform(image)

        # Update labels based on mask presence after cropping
        labels = (torch.sum(masks.reshape(len(self.categories), -1), dim=1) > 0).float()

        return image_tensor, masks, labels, bboxes

    def convert_bbox(self, bbox, orig_size, crop_size, output_size):
        x1, y1, x2, y2 = bbox
        orig_width, orig_height = orig_size
        crop_width, crop_height = crop_size
        output_width, output_height = output_size

        # Calculate crop offsets (assuming center crop)
        x_offset = (orig_width - crop_width) // 2
        y_offset = (orig_height - crop_height) // 2

        # Adjust bbox coordinates to cropped image
        x1_cropped = x1 - x_offset
        y1_cropped = y1 - y_offset
        x2_cropped = x2 - x_offset
        y2_cropped = y2 - y_offset

        # Scale bbox coordinates to output size
        x1_relative = x1_cropped / crop_width
        y1_relative = y1_cropped / crop_height
        x2_relative = x2_cropped / crop_width
        y2_relative = y2_cropped / crop_height

        return [x1_relative, y1_relative, x2_relative, y2_relative]

    @staticmethod
    def get_max_crop(current_size, target_size):
        current_ratio = current_size[0] / current_size[1]
        target_ratio = target_size[0] / target_size[1]
        if current_ratio > target_ratio:
            # Crop width to fit target ratio
            return int(current_size[1] * target_ratio), current_size[1]
        else:
            # Crop height to fit target ratio
            return current_size[0], int(current_size[0] / target_ratio)


def collate_fn_DeepFashion2(batch):
    images = []
    masks = []
    labels = []
    bboxes = []

    for image, mask, label, bbox in batch:
        images.append(image)
        masks.append(mask)
        labels.append(label)
        bboxes.append(bbox)

    images = torch.stack(images, 0)
    masks = torch.stack(masks, 0)
    labels = torch.stack(labels, 0)

    return images, masks, labels, bboxes


# def show_deepfashion2_image_masks_and_labels(dataset, index):
#     # Get the data from the dataset
#     image, masks, labels = dataset[index]
#
#     # Convert the image tensor to PIL Image for display
#     image_pil = transforms.ToPILImage()(image)
#
#     # Set up the plot
#     num_subplots = len(dataset.categories) + 1
#     fig, axs = plt.subplots(1, num_subplots, figsize=(20, 3))
#
#     # Plot the original image
#     axs[0].imshow(image_pil)
#     axs[0].set_title('Original Image')
#     axs[0].axis('off')
#
#     # Plot each mask
#     for i, mask in enumerate(masks):
#         axs[i + 1].imshow(mask, cmap='gray', interpolation='none')
#         axs[i + 1].set_title(f'{dataset.categories[i]}: {"1" if labels[i] == 1.0 else "0"}')
#         axs[i + 1].axis('off')
#
#     plt.tight_layout()
#     plt.show()


# def show_deepfashion2_image_masks_and_labels(image, masks, labels):
#     categories = [
#         'short sleeve top', 'long sleeve top', 'short sleeve outwear',
#         'long sleeve outwear', 'vest', 'sling', 'shorts',
#         'trousers', 'skirt', 'short sleeve dress',
#         'long sleeve dress', 'vest dress', 'sling dress'
#     ]
#
#     # Convert the image tensor to PIL Image for display
#     image_pil = transforms.ToPILImage()(image)
#
#     # Set up the plot
#     num_subplots = len(labels) + 1
#     fig, axs = plt.subplots(1, num_subplots, figsize=(20, 3))
#
#     # Plot the original image
#     axs[0].imshow(image_pil)
#     axs[0].set_title('Original Image')
#     axs[0].axis('off')
#
#     # Plot each mask
#     for i, mask in enumerate(masks):
#         axs[i + 1].imshow(mask, cmap='gray', interpolation='none')
#         axs[i + 1].set_title(f'{categories[i]}: {"1" if labels[i] == 1.0 else "0"}')
#         axs[i + 1].axis('off')
#
#     plt.tight_layout()
#     plt.show()


def show_deepfashion2_image_masks_and_labels(image, masks, labels, bboxes):
    categories = [
        'short sleeve top', 'long sleeve top', 'short sleeve outwear',
        'long sleeve outwear', 'vest', 'sling', 'shorts',
        'trousers', 'skirt', 'short sleeve dress',
        'long sleeve dress', 'vest dress', 'sling dress'
    ]

    # Convert the image tensor to PIL Image for display
    image_pil = transforms.ToPILImage()(image)
    img_width, img_height = image_pil.size

    # Set up the plot
    num_subplots = len(labels) + 1
    fig, axs = plt.subplots(1, num_subplots, figsize=(20, 3))

    # Plot the original image
    axs[0].imshow(image_pil)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    # Plot each mask
    for i, mask in enumerate(masks):
        axs[i + 1].imshow(image_pil, alpha=0.5)  # Show the underlying image
        axs[i + 1].imshow(mask, cmap='gray', alpha=0.5, interpolation='none')  # Overlay the mask
        axs[i + 1].set_title(f'{categories[i]}: {"1" if labels[i] == 1.0 else "0"}')
        axs[i + 1].axis('off')

        # Add bounding boxes to the plot
        for cat_id, bbox in bboxes:
            if cat_id == i:
                # Convert normalized coordinates to pixel coordinates
                x1, y1, x2, y2 = bbox
                x1_px = max(0, x1 * img_width)
                y1_px = max(0, y1 * img_height)
                x2_px = min(img_width, x2 * img_width)
                y2_px = min(img_height, y2 * img_height)
                rect_width = x2_px - x1_px
                rect_height = y2_px - y1_px
                rect = Rectangle((x1_px, y1_px), rect_width, rect_height, linewidth=1, edgecolor='r', facecolor='none')
                axs[i + 1].add_patch(rect)

    plt.tight_layout()
    plt.show()


def display_deepfashion_image_with_annotations(image_folder, anno_folder):
    for filename in os.listdir(anno_folder):
        if filename.endswith('.json'):
            json_path = os.path.join(anno_folder, filename)
            image_path = os.path.join(image_folder, filename.replace('.json', '.jpg'))
            
            try:
                # Load JSON data
                with open(json_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)

                # Open the corresponding image
                image = Image.open(image_path)
                fig, ax = plt.subplots()
                ax.imshow(image)

                # Prepare label text from JSON data and draw annotations
                labels = []
                for item_key in data:
                    if item_key.startswith('item'):
                        item = data[item_key]
                        category_name = item.get('category_name', 'Unknown')
                        category_id = item.get('category_id', 'Unknown')
                        style = item.get('style', 'Unknown')
                        labels.append(f"Category Name: {category_name}, Category ID: {category_id}, Style: {style}")

                        # Draw bounding box
                        bbox = item.get('bounding_box', [])
                        if bbox:
                            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor='r', facecolor='none')
                            ax.add_patch(rect)

                        # Draw segmentation polygons
                        segments = item.get('segmentation', [])
                        for poly in segments:
                            poly_points = [(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]
                            polygon = patches.Polygon(poly_points, linewidth=1, edgecolor='g', facecolor='none')
                            ax.add_patch(polygon)

                        # Draw landmarks
                        landmarks = item.get('landmarks', [])
                        for i in range(0, len(landmarks), 3):
                            x, y, v = landmarks[i], landmarks[i+1], landmarks[i+2]
                            if v == 2:  # Visible
                                ax.plot(x, y, 'bo')  # Blue dot for visible
                            elif v == 1:  # Occlusion
                                ax.plot(x, y, 'yo')  # Yellow dot for occlusion

                # Show image with labels and annotations
                plt.title('\n'.join(labels))
                plt.axis('off')  # Hide axes
                plt.show()
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")


class CCPDataset(Dataset):
    replay_attributes = []

    def __init__(self, root_dir, output_size=(512, 512),  # replay=10,
                 categories_and_attributes: CategoriesAndAttributes = None, pixel_only=False):
        self.categories_and_attributes = CelebAMaskHQCategoriesAndAttributes() if categories_and_attributes is None else categories_and_attributes
        self.output_size = output_size
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "photos")
        self.mask_dir = os.path.join(root_dir, "annotations/pixel-level/")
        self.label_dir = os.path.join(root_dir, "annotations/image-level/")
        self.mask_path_list = glob.glob(os.path.join(self.mask_dir, '*.mat'))
        self.label_path_list = glob.glob(os.path.join(self.label_dir, '*.mat'))
        self.merged_path_list = self.mask_path_list if pixel_only else self.mask_path_list + self.label_path_list
        # self.replay = replay
        self.original_length = len(self.merged_path_list)

    def __len__(self):
        return self.original_length

    # def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    #     idx_path = self.merged_path_list[idx]
    #     name = os.path.splitext(os.path.basename(idx_path))[0]
    #     image = imageio.imread(f'photos/{name}.jpg')
    #     labels = torch.zeros(len(self.categories_and_attributes.attributes))
    #     labels_str_list = self.categories_and_attributes.attributes
    #     # converte image into a torch tensor and resize it into output size (self.output_size)
    #     # convert it into colour channel first
    #     # make it 0~1
    #     # init groundtruth an integer tensor of zeros same size to image (but has only one channel)
    #     if idx_path in self.mask_dir:
    #         annotation_mat = loadmat('annotations/pixel-level/' + name + '.mat')
    #         groundtruth = annotation_mat['groundtruth']
    #         # converte groundtruth into a torch tensor and resize it into output size
    #         # call get_image_pixel_labels and assign 1s to labels based on its index in labels_str_list
    #         pixel_labels = torch.tensor(True)
    #     elif idx_path in self.label_dir:
    #         # call get_image_labels and assign 1s to labels based on its index in labels_str_list
    #         pixel_labels = torch.tensor(False)
    #         pass

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx_path = self.merged_path_list[idx]
        name = os.path.splitext(os.path.basename(idx_path))[0]
        image = imageio.imread(os.path.join(self.image_dir, f'{name}.jpg'))

        # Calculate the aspect ratio of the output size
        output_aspect_ratio = self.output_size[1] / self.output_size[0]

        # Determine the maximum crop size of the original image that maintains this aspect ratio
        orig_aspect_ratio = image.shape[1] / image.shape[0]
        if orig_aspect_ratio > output_aspect_ratio:
            # Width is too wide for the target aspect ratio
            crop_height = image.shape[0]
            crop_width = int(crop_height * output_aspect_ratio)
        else:
            # Height is too high for the target aspect ratio
            crop_width = image.shape[1]
            crop_height = int(crop_width / output_aspect_ratio)

        # Randomly choose an offset for the crop (i.e., top-left corner of the crop)
        x_offset = random.randint(0, image.shape[1] - crop_width)
        y_offset = random.randint(0, image.shape[0] - crop_height)

        # Apply crop to the original image
        cropped_image = image[y_offset:y_offset + crop_height, x_offset:x_offset + crop_width]

        # Image transformations: resize, convert color channel order, and normalize
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.output_size),
            transforms.ToTensor(),  # Converts to [C, H, W] & scales to [0, 1]
        ])
        image_tensor = transform(cropped_image)

        labels = torch.zeros(len(self.categories_and_attributes.attributes))
        labels_str_list = self.categories_and_attributes.attributes

        # Initialize groundtruth tensor
        groundtruth_tensor = torch.zeros(1, *self.output_size,
                                         dtype=torch.long)  # Assuming you want integer labels for segmentation

        has_pixel_labels = torch.tensor(0)
        if idx_path in self.mask_path_list:
            annotation_mat = loadmat(os.path.join(self.mask_dir, f'{name}.mat'))
            groundtruth = annotation_mat['groundtruth']

            cropped_groundtruth = groundtruth[y_offset:y_offset + crop_height, x_offset:x_offset + crop_width]

            # Convert the numpy array to a PIL Image for resizing
            cropped_groundtruth_image = Image.fromarray(cropped_groundtruth.astype('uint8'))

            # Resize using NEAREST to avoid changing class labels unintentionally
            resized_groundtruth_image = cropped_groundtruth_image.resize((self.output_size[1], self.output_size[0]), resample=Image.NEAREST)

            # Convert the PIL Image back to a numpy array
            resized_groundtruth_array = np.array(resized_groundtruth_image)

            # Finally, convert the numpy array to a PyTorch tensor
            groundtruth_tensor = torch.from_numpy(
                resized_groundtruth_array).long()  # Ensure it's a long tensor for indexing

            # Update labels tensor based on pixel annotations
            pixel_labels = self.get_image_pixel_labels(idx_path)  # Assuming this function returns a list of label names
            for label in pixel_labels:
                if label in labels_str_list:
                    labels[labels_str_list.index(label)] = 1
            has_pixel_labels = torch.tensor(1)

        elif idx_path in self.label_path_list:
            image_labels = self.get_image_labels(idx_path)  # Assuming this function returns a list of label names
            for label in image_labels:
                if label in labels_str_list:
                    labels[labels_str_list.index(label)] = 1
            has_pixel_labels = torch.tensor(0)

        num_classes = len(labels)
        # Ensure groundtruth_tensor is squeezed in case it has an unnecessary channel dimension
        groundtruth_tensor = groundtruth_tensor.squeeze(0)  # Assuming groundtruth_tensor is of shape [1, H, W]
        # Create one-hot encoding
        H, W = groundtruth_tensor.shape
        one_hot_groundtruth = torch.zeros((num_classes, H, W), dtype=torch.float32)
        one_hot_groundtruth.scatter_(0, groundtruth_tensor.unsqueeze(0), 1)

        # # debug:::::
        # # Visualization part
        # fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # Adjust the size as needed
        #
        # # Plot original image
        # axs[0].imshow(image)
        # axs[0].set_title('Original Image')
        # axs[0].axis('off')
        #
        # # Highlight the cropped area in the original image
        # rect = patches.Rectangle((x_offset, y_offset), crop_width, crop_height, linewidth=1, edgecolor='r',
        #                          facecolor='none')
        # axs[0].add_patch(rect)
        #
        # # Plot cropped and resized image
        # axs[1].imshow(transforms.ToPILImage()(image_tensor))
        # axs[1].set_title('Cropped & Resized Image')
        # axs[1].axis('off')
        #
        # # Check and plot ground truth if available
        # if has_pixel_labels == torch.tensor(1):
        #     # Assuming groundtruth_tensor is already in one-hot and resized to self.output_size
        #     # Convert one-hot back to class indices for visualization
        #     groundtruth_indices = torch.argmax(one_hot_groundtruth, dim=0)
        #     axs[2].imshow(groundtruth_indices, cmap='jet')
        #     axs[2].set_title('Ground Truth Annotation')
        #     axs[2].axis('off')
        # else:
        #     # axs[2].set_visible(False)
        #
        #     axs[2].text(0.5, 0.5, 'No pixel-level labels', horizontalalignment='center', verticalalignment='center',
        #                 transform=axs[2].transAxes)
        #     axs[2].axis('off')
        #
        # # Add texture labels based on 'labels' tensor
        # present_labels = [labels_str_list[i] for i, label in enumerate(labels) if label == 1]
        # texture_labels_text = "Texture Labels: " + ", ".join(present_labels)
        # # fig.suptitle(texture_labels_text, fontsize=14)
        # # Adjusting the text to appear like a subtitle below the image in axs[1]
        # axs[1].text(0.5, -0.5, texture_labels_text, ha='center', va='top', transform=axs[1].transAxes, fontsize=14)
        #
        # # axs[1].suptitle(texture_labels_text, fontsize=14)
        #
        # plt.show()

        return image_tensor.to(dtype=torch.float32), one_hot_groundtruth.to(dtype=torch.float32), labels.to(dtype=torch.float32), has_pixel_labels.to(dtype=torch.float32)

    def get_image_labels(self, im_file):
        """
        Returns a list of labels contained in the given image file.
        Parameters:
        - im_file: String. The path to the image file's corresponding annotation file.
        Returns:
        - A list of label names that the image contains.
        """
        # Load the label list
        label_list_data = scipy.io.loadmat(os.path.join(self.root_dir, 'label_list.mat'))['label_list'][0].tolist()
        # Extract the image name from the given file path
        name = os.path.splitext(os.path.basename(im_file))[0]
        # Load the tags from the annotation file
        try:
            tags = scipy.io.loadmat(os.path.join(self.label_dir, f'{name}.mat'))['tags'][0]
        except FileNotFoundError:
            return f"Annotation file for {name} not found."
        # Translate tags to label names
        label_names = [str(label_list_data[tag][0]) for tag in tags]
        return label_names

    def get_image_pixel_labels(self, im_file):
        """
        Returns a list of labels contained in the given image file based on pixel-level annotations.
        Parameters:
        - im_file: String. The file name or path to the image's pixel-level annotation file.
        Returns:
        - A list of label names that the image contains based on pixel-level annotations.
        """
        # Load the label list
        label_list_data = scipy.io.loadmat(os.path.join(self.root_dir, 'label_list.mat'))['label_list'][0].tolist()
        # Extract the base name for the image file
        name = os.path.splitext(os.path.basename(im_file))[0]
        # Load the pixel-level annotation for the given image
        try:
            annotation_mat = scipy.io.loadmat(os.path.join(self.mask_dir, f'{name}.mat'))
            groundtruth = annotation_mat['groundtruth']
        except FileNotFoundError:
            return f"Pixel-level annotation file for {name} not found."
        # Get unique labels from the groundtruth
        cur_labels = np.unique(groundtruth)
        # Translate those labels into human-readable names
        label_names = [str(label_list_data[int(label)][0]) for label in cur_labels if int(label) < len(label_list_data)]
        return label_names


class MergedCCPDataset(CCPDataset):
    def __init__(self, root_dir, output_size=(512, 512),  # replay=10,
                 categories_and_attributes: CategoriesAndAttributes = None, pixel_only=False):
        super(MergedCCPDataset, self).__init__(
            root_dir, output_size, categories_and_attributes=categories_and_attributes, pixel_only=pixel_only,
        )

    def __getitem__(self, idx):
        image, unmerged_masks, attributes, has_pixel_labels = super(MergedCCPDataset, self).__getitem__(idx)

        # Convert Tensor images to PIL for operations
        image = transforms.ToPILImage()(image)
        unmerged_masks = [transforms.ToPILImage()(mask) for mask in unmerged_masks]

        masks = []

        for category in sorted(list(self.categories_and_attributes.merged_categories.keys())):
            combined_mask_np = np.zeros_like(np.array(unmerged_masks[0]))  # Initialize with zeros
            for sub_category in self.categories_and_attributes.merged_categories[category]:
                sub_cat_idx = self.categories_and_attributes.mask_categories.index(sub_category)

                mask_to_merge_np = np.array(unmerged_masks[sub_cat_idx])

                # Use logical or operation to combine masks
                combined_mask_np = np.logical_or(combined_mask_np, mask_to_merge_np).astype(np.uint8)

            masks.append(Image.fromarray(combined_mask_np * 255, 'L'))  # Convert back to PIL and append

        # Convert back to Tensor
        image = transforms.ToTensor()(image)
        masks = torch.stack([transforms.ToTensor()(m) for m in masks], dim=0).squeeze(1)

        return image, masks, attributes, has_pixel_labels

class CelebAMaskHQDataset(Dataset):
    replay_attributes = ['Wearing_Hat', 'Eyeglasses', 'Blond_Hair']

    def __init__(self, root_dir, output_size=(512, 512), replay=10,
                 categories_and_attributes: CategoriesAndAttributes = None):
        self.categories_and_attributes = CelebAMaskHQCategoriesAndAttributes() if categories_and_attributes is None else categories_and_attributes
        self.output_size = output_size
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "CelebA-HQ-img")
        self.mask_dir = os.path.join(root_dir, "CelebAMask-HQ-mask-anno")
        self.replay = replay

        # Load image list
        self.image_list = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])
        self.original_length = len(self.image_list)

        # Build mask dictionary
        self.mask_dict = {}
        for folder in range(15):  # There are folders from 0 to 14
            folder_path = os.path.join(self.mask_dir, str(folder))
            for mask_file in os.listdir(folder_path):
                match = re.match(r"(\d+)_(\w+)\.png", mask_file)
                if match:
                    img_idx, category = match.groups()
                    if img_idx not in self.mask_dict:
                        self.mask_dict[img_idx] = {}
                    full_path = os.path.join(folder_path, mask_file)
                    self.mask_dict[img_idx][category] = full_path

        # Load attribute data
        self.attribute_file = os.path.join(root_dir, "CelebAMask-HQ-attribute-anno.txt")
        with open(self.attribute_file, 'r') as file:
            lines = file.readlines()
            self.attributes = lines[1].split()
            self.attribute_data = {}
            for line in lines[2:]:
                parts = line.strip().split()
                filename = parts[0]
                attrs = {self.attributes[i]: int(parts[i + 1]) for i in range(len(self.attributes))}
                self.attribute_data[filename] = attrs

        # replay data:
        for _ in range(self.replay):
            for atr in self.replay_attributes:
                for i in range(self.original_length):
                    name = self.image_list[i]  # .split('.')[0]
                    if self.has_attribute(name, atr):
                        self.image_list.append(self.image_list[i])

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Load image
        img_name = os.path.join(self.image_dir, self.image_list[idx])
        image = Image.open(img_name)

        # Load masks for this image
        img_idx = self.image_list[idx].split('.')[0].zfill(5)
        masks = []
        for category in self.categories_and_attributes.mask_categories:
            if category in self.mask_dict[img_idx]:
                mask_path = self.mask_dict[img_idx][category]
                mask = Image.open(mask_path).convert('L')  # Convert to grayscale
                # desired_size = image.size[:2]  # Get the size (1024, 1024) from (1024, 1024, 3)
                mask = mask.resize(self.output_size)  # Resize to desired size
                masks.append(mask)

            else:
                # If mask of this category is not available, use a black image
                # print(category)
                masks.append(Image.new('L', image.size).resize(self.output_size))

        image = image.resize(self.output_size)

        attributes = []
        for attribute in self.categories_and_attributes.attributes:
            if attribute in self.categories_and_attributes.avoided_attributes:
                continue
            if self.has_attribute(self.image_list[idx], attribute):
                attributes.append(1.0)
            else:
                attributes.append(0.0)
        for attribute in self.categories_and_attributes.mask_labels:
            mask = masks[sorted(list(self.categories_and_attributes.merged_categories.keys())).index(attribute)]
            label = transforms.ToTensor()(mask).any(dim=-1).any(dim=-1).float()
            attributes.append(label)

        # Convert back to Tensor
        image = transforms.ToTensor()(image)
        masks = torch.stack([transforms.ToTensor()(m) for m in masks], dim=0).squeeze(1)
        attributes = torch.tensor(attributes, dtype=torch.float)

        return image, masks, attributes

    def get_image(self, idx: int):
        image_tensor, masks_tensor, attributes_tensor = self.__getitem__(idx)
        image_np, masks_np, attributes_np = np.array(image_tensor), np.array(masks_tensor), np.array(attributes_tensor)
        masks = {}
        for i, category in enumerate(self.categories_and_attributes.mask_categories):
            masks[category] = masks_np[i]
        attributes = {}
        c = 0
        for i, attribute in enumerate(self.categories_and_attributes.attributes):
            if attribute in self.categories_and_attributes.avoided_attributes:
                c += 1
                continue
            attributes[attribute] = float(attributes_np[i - c])
        return ImageWithMasksAndAttributes(image_np, masks, attributes)

    def __len__(self):
        return len(self.image_list)

    def has_attribute(self, filename, attribute):
        """Check if the specified image has the specified attribute"""
        if filename in self.attribute_data and attribute in self.attribute_data[filename]:
            return self.attribute_data[filename][attribute] == 1
        return False


class SelectedCelebAMaskHQDataset(CelebAMaskHQDataset):
    selected_categories = ['cloth', 'hair', 'hat', 'eye_g', 'skin', ]
    indices = [0, 2, 9, 10, 8, ]

    def __init__(self, root_dir, output_size=(512, 512), replay=10, categories_and_attributes=None):
        super(SelectedCelebAMaskHQDataset, self).__init__(root_dir, output_size, replay=replay,
                                                          categories_and_attributes=categories_and_attributes)
        self.selected_indices = [self.categories_and_attributes.mask_categories.index(cat) for cat in
                                 self.selected_categories]

    def __getitem__(self, idx):
        image, masks_all, attributes = super(SelectedCelebAMaskHQDataset, self).__getitem__(idx)
        masks_selected = masks_all[self.selected_indices]
        return image, masks_selected, attributes


class MergedCelebAMaskHQDataset(CelebAMaskHQDataset):  # 'brow', 'eye', 'mouth', 'nose', ]
    def __init__(self, root_dir, output_size=(512, 512), replay=10, categories_and_attributes=None):
        super(MergedCelebAMaskHQDataset, self).__init__(root_dir, output_size, replay=replay,
                                                        categories_and_attributes=categories_and_attributes)

    def __getitem__(self, idx):
        image, unmerged_masks, attributes = super(MergedCelebAMaskHQDataset, self).__getitem__(idx)

        # Convert Tensor images to PIL for operations
        image = transforms.ToPILImage()(image)
        unmerged_masks = [transforms.ToPILImage()(mask) for mask in unmerged_masks]

        masks = []

        for category in sorted(list(self.categories_and_attributes.merged_categories.keys())):
            combined_mask_np = np.zeros_like(np.array(unmerged_masks[0]))  # Initialize with zeros
            for sub_category in self.categories_and_attributes.merged_categories[category]:
                sub_cat_idx = self.categories_and_attributes.mask_categories.index(sub_category)

                mask_to_merge_np = np.array(unmerged_masks[sub_cat_idx])

                # Use logical or operation to combine masks
                combined_mask_np = np.logical_or(combined_mask_np, mask_to_merge_np).astype(np.uint8)

            masks.append(Image.fromarray(combined_mask_np * 255, 'L'))  # Convert back to PIL and append

        # Convert back to Tensor
        image = transforms.ToTensor()(image)
        masks = torch.stack([transforms.ToTensor()(m) for m in masks], dim=0).squeeze(1)

        return image, masks, attributes

    def get_image(self, idx: int):
        image_tensor, masks_tensor, attributes_tensor = self.__getitem__(idx)
        image_np, masks_np, attributes_np = np.array(image_tensor), np.array(masks_tensor), np.array(attributes_tensor)
        masks = {}
        for i, category in enumerate(sorted(list(self.categories_and_attributes.merged_categories.keys()))):
            masks[category] = masks_np[i]
        attributes = {}
        c = 0
        for i, attribute in enumerate(self.categories_and_attributes.attributes):
            if attribute in self.categories_and_attributes.avoided_attributes:
                c += 1
                continue
            attributes[attribute] = float(attributes_np[i - c])
        return ImageWithMasksAndAttributes(image_np, masks, attributes)


class AugmentedDataset(Dataset):
    def __init__(self, dataset_source: Dataset, flip_prob=0.5, crop_ratio=(0.8, 0.8), scale_factor=(0.5, 2),
                 output_size=(512, 512),
                 noise_level=(0, 10), blur_radius=(0, 2), brightness_factor=(0.75, 1.25), pil=False, seed: int = None):
        self.source = dataset_source

        np.random.seed(seed)

        self.flip_prob = flip_prob
        self.crop_ratio = crop_ratio
        self.output_size = output_size
        self.noise_level = noise_level
        self.blur_radius = blur_radius
        self.brightness_factor = brightness_factor
        self.scale_factor = scale_factor
        self.pil = pil

    def _get_transform_params(self):
        new_crop_width_ratio = np.random.uniform(self.crop_ratio[0], 1)
        new_crop_height_ratio = np.random.uniform(self.crop_ratio[1], 1)
        new_crop_left_ratio = np.random.uniform(0, 1 - new_crop_width_ratio)
        new_crop_top_ratio = np.random.uniform(0, 1 - new_crop_height_ratio)

        params = {
            "do_flip": np.random.rand() < self.flip_prob,
            "crop_width_ratio": new_crop_width_ratio,
            "crop_height_ratio": new_crop_height_ratio,
            "crop_left_ratio": new_crop_left_ratio,
            "crop_top_ratio": new_crop_top_ratio,
            "scale_factor": np.random.uniform(self.scale_factor[0], self.scale_factor[1]),
        }

        if params["scale_factor"] < 1:
            # Generate random padding offsets for scale_factor < 1
            total_padding_width = self.output_size[0] - int(self.output_size[0] * params["scale_factor"])
            total_padding_height = self.output_size[1] - int(self.output_size[1] * params["scale_factor"])
            params["padding_left"] = np.random.randint(0, total_padding_width + 1)
            params["padding_top"] = np.random.randint(0, total_padding_height + 1)
        else:
            # Generate random crop offsets for scale_factor > 1
            extra_width = int(self.output_size[0] * params["scale_factor"]) - self.output_size[0]
            extra_height = int(self.output_size[1] * params["scale_factor"]) - self.output_size[1]
            params["crop_left"] = np.random.randint(0, extra_width + 1)
            params["crop_top"] = np.random.randint(0, extra_height + 1)
        return params

    def _apply_common_transforms(self, image, params):
        # Convert Tensor to PIL Image for transformations
        image = transforms.ToPILImage()(image)

        # Resize to output size
        if not self.pil:
            image = image.resize(self.output_size)

        # Use the stored params instead of generating new ones for flip
        if params["do_flip"]:
            image = ImageOps.mirror(image)

        new_width = int(image.width * params["crop_width_ratio"])
        new_height = int(image.height * params["crop_height_ratio"])
        left = int(image.width * params["crop_left_ratio"])
        top = int(image.height * params["crop_top_ratio"])
        image = image.crop((left, top, left + new_width, top + new_height))

        # Resize to output size (again!) after cropping
        image = image.resize(self.output_size)

        # Apply scale
        scale_factor = params["scale_factor"]
        scaled_size = (int(self.output_size[0] * scale_factor), int(self.output_size[1] * scale_factor))
        image = image.resize(scaled_size, resample=Image.BILINEAR)

        if scale_factor < 1:
            # Calculate padding sizes based on the random offsets generated
            padding_l = params["padding_left"]
            padding_t = params["padding_top"]
            padding_r = self.output_size[0] - scaled_size[0] - padding_l
            padding_b = self.output_size[1] - scaled_size[1] - padding_t

            # Calculate random color for padding
            random_color = self._get_random_colour(image)
            # Apply padding
            image = ImageOps.expand(image, border=(padding_l, padding_t, padding_r, padding_b), fill=random_color)
        elif scale_factor > 1:
            # Calculate the crop positions based on the random offsets generated
            crop_left = params["crop_left"]
            crop_top = params["crop_top"]
            crop_right = crop_left + self.output_size[0]
            crop_bottom = crop_top + self.output_size[1]
            # Crop the image
            image = image.crop((crop_left, crop_top, crop_right, crop_bottom))

        return image

    def _apply_image_transforms(self, image, params):
        image = self._apply_common_transforms(image, params)
        ori_image = image.copy()

        # Add noise
        noise_level = np.random.uniform(self.noise_level[0], self.noise_level[1])
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, (image.height, image.width, 3)).astype(np.uint8)
            noise_image = Image.fromarray(noise, 'RGB')
            image = ImageChops.add(image, noise_image)

        # Add blur
        blur_value = np.random.uniform(self.blur_radius[0], self.blur_radius[1])
        if blur_value > 0:
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_value))

        # Adjust brightness
        brightness_value = np.random.uniform(self.brightness_factor[0], self.brightness_factor[1])
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_value)

        return image, ori_image

    def __getitem__(self, idx):
        # Unpack the first three known returns and capture any additional ones in 'extra_returns'
        image, masks, attributes, *extra_returns = self.source.__getitem__(idx)

        # Generate transform params for this sample
        transform_params = self._get_transform_params()

        # Apply the same augmentations to image and masks using the generated params
        image, ori_image = self._apply_image_transforms(image, transform_params)
        masks = [self._apply_common_transforms(mask, transform_params) for mask in masks]

        image = transforms.ToTensor()(image)
        # ori_image = transforms.ToTensor()(ori_image)  # Commented out as per your code
        masks = torch.stack([transforms.ToTensor()(m) for m in masks], dim=0).squeeze(1)

        # Return the processed image, masks, attributes, and any extra returns
        return image, masks, attributes, *extra_returns

    def __len__(self):
        return len(self.source)

    def _get_random_colour(self, image):
        mode = image.mode
        if mode == 'RGB':
            return tuple(np.random.randint(0, 256, size=3).tolist())
        elif mode == 'RGBA':
            return tuple(np.random.randint(0, 256, size=4).tolist())
        elif mode == 'L':
            # For Gray, ZERO!!!
            return 0


class AugmentedDeepFashion2Dataset(AugmentedDataset):
    def __init__(self, dataset_source, flip_prob=0.5, crop_ratio=(0.8, 0.8), scale_factor=(0.5, 2),
                 output_size=(512, 512), noise_level=(0, 10), blur_radius=(0, 2), brightness_factor=(0.75, 1.25),
                 pil=False, seed=None):
        super().__init__(dataset_source, flip_prob, crop_ratio, scale_factor, output_size,
                         noise_level, blur_radius, brightness_factor, pil, seed)

    def __getitem__(self, idx):
        # Retrieve data from the source dataset
        image_tensor, masks, labels, bboxes = self.source.__getitem__(idx)

        # Generate transform parameters for this sample
        transform_params = self._get_transform_params()

        # Apply transformations to the image
        image = transforms.ToPILImage()(image_tensor)
        transformed_image, _ = self._apply_image_transforms(image_tensor, transform_params)

        # Adjust bounding boxes
        transformed_bboxes = [self._adjust_bbox(bbox, transform_params, image.size) for bbox in bboxes]

        # Convert transformed image back to tensor
        transformed_image_tensor = transforms.ToTensor()(transformed_image)

        # Transform masks if they exist
        transformed_masks = torch.stack(
            [transforms.ToTensor()(self._apply_common_transforms(m, transform_params)) for m in
             masks], dim=0).squeeze(1)

        return transformed_image_tensor, transformed_masks, labels, transformed_bboxes

    def _adjust_bbox(self, bbox, params, orig_size):
        # Extract bounding box coordinates
        category_id, (x1, y1, x2, y2) = bbox

        # Calculate crop dimensions
        crop_width = params['crop_width_ratio'] * orig_size[0]
        crop_height = params['crop_height_ratio'] * orig_size[1]
        crop_x1 = params['crop_left_ratio'] * orig_size[0]
        crop_y1 = params['crop_top_ratio'] * orig_size[1]

        # Adjust bounding box coordinates to cropped image
        x1_cropped = (x1 * orig_size[0] - crop_x1)
        y1_cropped = (y1 * orig_size[1] - crop_y1)
        x2_cropped = (x2 * orig_size[0] - crop_x1)
        y2_cropped = (y2 * orig_size[1] - crop_y1)

        # Normalize to the cropped size to get relative coordinates
        x1_relative = x1_cropped / crop_width
        y1_relative = y1_cropped / crop_height
        x2_relative = x2_cropped / crop_width
        y2_relative = y2_cropped / crop_height

        # Apply flip if necessary
        if params['do_flip']:
            x1_relative = 1 - x1_relative
            x2_relative = 1 - x2_relative
            x1_relative, x2_relative = x2_relative, x1_relative  # Swap values as flip changes their positions

        return (category_id, (x1_relative, y1_relative, x2_relative, y2_relative))


def show_image(image, title=""):
    """Display an image"""
    plt.imshow(image.permute(1, 2, 0).numpy().clip(0, 1))
    plt.title(title)
    plt.axis('off')


def show_masks(masks, categories, mask_indices):
    """Display selected masks on a single figure"""

    num_masks = len(mask_indices)
    cols = 5
    rows = num_masks // cols
    rows += num_masks % cols
    if rows == 0:
        rows = 1
    position = range(1, num_masks + 1)

    fig = plt.figure(figsize=(10, 10))  # adjust the size as needed

    for i, idx in enumerate(mask_indices):
        mask_image = masks[i].numpy()
        ax = fig.add_subplot(rows, cols, position[i])
        ax.imshow(mask_image, cmap="gray")
        ax.set_title(categories[i])
        ax.axis('off')

    plt.tight_layout()


class ATRDataset(Dataset):
    data_map = {
        'background': 0,
        'hat': 1,
        'hair': 2,
        'sunglass': 3,
        'upper-clothes': 4,
        'skirt': 5,
        'pants': 6,
        'dress': 7,
        'belt': 8,
        'left-shoe': 9,
        'right-shoe': 10,
        'face': 11,
        'left-leg': 12,
        'right-leg': 13,
        'left-arm': 14,
        'right-arm': 15,
        'bag': 16,
        'scarf': 17,
    }


class LIPDataset(Dataset):
    data_map = {
        'hat': 1,
        'hair': 2,
        'glove': 3,
        'sunglasses': 4,
        'upperclothes': 5,
        'dress': 6,
        'coat': 7,
        'socks': 8,
        'pants': 9,
        'jumpsuits': 10,
        'scarf': 11,
        'skirt': 12,
        'face': 13,
        'left-arm': 14,
        'right-arm': 15,
        'left-leg': 16,
        'right-leg': 17,
        'left-shoe': 18,
        'right-shoe': 19,
    }
    class_ids = [5, 6, 7, 10, 2, 1, 4, 13]
    class_ids = [2, 1, 4, 13]
    # categories = ['cloth', 'hair', 'hat', 'eye_g', 'skin',]
    categories = ['hair', 'hat', 'eye_g', 'skin', ]

    def __init__(self, root_dir, mode='merge'):
        self.root_dir = root_dir
        self.mode = mode
        assert mode in ['train', 'val', 'merge'], "Mode must be 'train', 'val', or 'merge'"

        # Determine image and mask directories
        train_images_dir = os.path.join(root_dir, 'TrainVal_images', 'train_images')
        val_images_dir = os.path.join(root_dir, 'TrainVal_images', 'val_images')
        train_masks_dir = os.path.join(root_dir, 'TrainVal_parsing_annotations', 'train_segmentations')
        val_masks_dir = os.path.join(root_dir, 'TrainVal_parsing_annotations', 'val_segmentations')
        train_ids_file = os.path.join(root_dir, 'TrainVal_images', 'train_id.txt')
        val_ids_file = os.path.join(root_dir, 'TrainVal_images', 'val_id.txt')

        self.images_paths = []
        self.masks_paths = []

        # Load train ids and paths
        if mode in ['train', 'merge']:
            with open(train_ids_file, 'r') as file:
                train_ids = file.read().splitlines()
            self.images_paths.extend([os.path.join(train_images_dir, image_id + '.jpg') for image_id in train_ids])
            self.masks_paths.extend([os.path.join(train_masks_dir, image_id + '.png') for image_id in train_ids])

        # Load val ids and paths
        if mode in ['val', 'merge']:
            with open(val_ids_file, 'r') as file:
                val_ids = file.read().splitlines()
            self.images_paths.extend([os.path.join(val_images_dir, image_id + '.jpg') for image_id in val_ids])
            self.masks_paths.extend([os.path.join(val_masks_dir, image_id + '.png') for image_id in val_ids])

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image_path = self.images_paths[idx]
        mask_path = self.masks_paths[idx]

        # Read image and mask files as NumPy arrays
        # i = np.array(Image.open(image_path).convert('RGB'))
        # k = np.array(Image.open(mask_path))
        # plt.figure()
        # plt.imshow(k)
        # plt.show()
        # l = self._generate_class_masks(np.array(Image.open(mask_path)).astype(np.uint8), self.class_ids)
        image = transforms.ToTensor()(np.array(Image.open(image_path).convert('RGB')))
        # masks = torch.Tensor(self._generate_class_masks(np.array(Image.open(mask_path)).astype(np.uint8), self.class_ids, [(5,6,7,10)]))
        masks = torch.Tensor(
            self._generate_class_masks(np.array(Image.open(mask_path)).astype(np.uint8), self.class_ids, []))

        return image, masks

    def _generate_class_masks(self, mask_array, class_ids, merge_tuples=[]):
        """
        Generate binary masks for each class ID provided, with specified pairs merged.
        
        :param mask_array: numpy array of shape (x, y) where each pixel corresponds to a class label.
        :param class_ids: list of class IDs to generate binary masks for.
        :param merge_pairs: list of tuples, where each tuple contains two adjacent class IDs that need to be merged.
        :return: numpy array of shape (k, x, y) where k is the number of class IDs after merging, and each slice (k, :, :)
                is a binary mask for a class ID or a merged class ID in class_ids.
        """
        masks = []
        merged_class_ids = set()
        for group in merge_tuples:
            merged_mask = np.zeros_like(mask_array, dtype=np.uint8)
            for class_id in group:
                merged_mask = merged_mask | (mask_array == class_id).astype(np.uint8)
                merged_class_ids.add(class_id)
            masks.append(merged_mask)
        for class_id in class_ids:
            if class_id not in merged_class_ids:
                masks.append((mask_array == class_id).astype(np.uint8))
        masks = np.stack(masks, axis=0)

        return masks


# Usage
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    celebAMaskHQCategoriesAndAttributes = CelebAMaskHQCategoriesAndAttributes()
    # dataset = LIPDataset(root_dir=r"D:\workspace\LIP")
    # dataset = AugmentedDataset(
    #     dataset_source=dataset, output_size=(256,256),
    #     flip_prob=0.5, crop_ratio=(1, 1), scale_factor=(0.8, 1.2),
    #     noise_level=(0, 1), blur_radius=(0, 1), brightness_factor=(0.85, 1.25),
    #     seed=0, pil=True,
    # )
    dataset = MergedCelebAMaskHQDataset(root_dir=r"/home/bentengma/work_space/CelebAMask-HQ", output_size=(256, 256))
    dataset = AugmentedDataset(
        dataset_source=dataset, output_size=(256, 256),
        flip_prob=0.5, crop_ratio=(1, 1), scale_factor=(0.8, 1.2),
        noise_level=(0, 1), blur_radius=(0, 1), brightness_factor=(0.85, 1.25),
        seed=0, pil=True,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Set batch_size to 1 to load one sample at a time
    # dataset = AugmentedDataset(
    #     dataset_source=dataset, output_size=(256,256), 
    #     flip_prob=0.5, crop_ratio=(1, 1), scale_factor=(0.2, 1),
    #     noise_level=(0, 3), blur_radius=(0, 2), brightness_factor=(0.75, 1.35), 
    #     seed=0,
    # )

    for batch_idx, (image_batch, masks_batch, attribute_batch, ori_image_batch) in enumerate(dataloader):
        # Because we are using a batch size of 1, we need to squeeze the batch dimension
        image = image_batch.squeeze(0)
        masks = masks_batch.squeeze(0)
        attribute = attribute_batch.squeeze(0)
        ori_image_batch = ori_image_batch.squeeze(0)

        # Show the original image
        show_image(image, "Image")

        # Show some masks
        # Change these indices to see different masks
        show_masks(masks, sorted(list(celebAMaskHQCategoriesAndAttributes.merged_categories.keys())),
                   range(len(celebAMaskHQCategoriesAndAttributes.merged_categories.keys())))

        plt.show()

        input()
    # dataset = MergedCelebAMaskHQDataset(root_dir=r"/home/bentengma/work_space/CelebAMask-HQ", output_size=(256, 256))
    #
    # for i in range(dataset.__len__()):
    #     image = dataset.get_image(i)
    #     print(image.masks, image.attributes)
    #     input()

    # from torch.utils.data import DataLoader
    # dataset = SelectedCelebAMaskHQDataset(root_dir=r"D:\workspace\CelebAMask-HQ", output_size=(256, 256))
    # dataset = AugmentedDataset(
    #     dataset_source=dataset, output_size=(256,256), 
    #     flip_prob=0.5, crop_ratio=(1, 1), scale_factor=(0.2, 1),
    #     noise_level=(0, 3), blur_radius=(0, 2), brightness_factor=(0.75, 1.35), 
    #     seed=0,
    # )
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Set batch_size to 1 to load one sample at a time

    # for batch_idx, (image_batch, masks_batch) in enumerate(dataloader):
    #     # Because we are using a batch size of 1, we need to squeeze the batch dimension
    #     image = image_batch.squeeze(0)
    #     masks = masks_batch.squeeze(0)

    #     # Show the original image
    #     show_image(image, "Image")

    #     # Show some masks
    #     # Change these indices to see different masks
    #     mask_indices = range(12)
    #     show_masks(masks, SelectedCelebAMaskHQDataset.selected_categories, SelectedCelebAMaskHQDataset.indices)

    #     plt.show()

    # # Let's say we want to show only 5 images for demonstration purposes
    # if batch_idx == 4:
    #     break
