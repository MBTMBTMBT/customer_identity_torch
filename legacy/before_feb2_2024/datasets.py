import os
from PIL import Image, ImageChops, ImageFilter, ImageEnhance, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
import re


class CelebAMaskHQDataset(Dataset):
    categories = ['cloth', 'r_ear', 'hair', 'l_brow', 'l_eye', 'l_lip', 'mouth', 'neck', 'nose', 'r_brow',
                  'r_ear', 'r_eye', 'skin', 'u_lip', 'hat', 'l_ear', 'neck_l', 'eye_g']
    replay_categories = ['Wearing_Hat', 'Eyeglasses',]

    def __init__(self, root_dir, output_size=(512, 512), replay=10):
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
                attrs = {self.attributes[i]: int(parts[i+1]) for i in range(len(self.attributes))}
                self.attribute_data[filename] = attrs

        # replay data:
        for _ in range(self.replay):
            for atr in self.replay_categories:
                for i in range(self.original_length):
                    name = self.image_list[i]  # .split('.')[0]
                    if self.has_attribute(name, atr):
                        self.image_list.append(self.image_list[i])

    def __getitem__(self, idx):
        # Load image
        img_name = os.path.join(self.image_dir, self.image_list[idx])
        image = Image.open(img_name)

        # Load masks for this image
        img_idx = self.image_list[idx].split('.')[0].zfill(5)
        masks = []
        for category in self.categories:
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

        # Convert back to Tensor
        image = transforms.ToTensor()(image)
        masks = torch.stack([transforms.ToTensor()(m) for m in masks], dim=0).squeeze(1)

        return image, masks

    def __len__(self):
        return len(self.image_list)
    
    def has_attribute(self, filename, attribute):
        """Check if the specified image has the specified attribute"""
        if filename in self.attribute_data and attribute in self.attribute_data[filename]:
            return self.attribute_data[filename][attribute] == 1
        return False
    

class SelectedCelebAMaskHQDataset(CelebAMaskHQDataset):
    selected_categories = ['cloth', 'hair', 'hat', 'eye_g', 'skin',]
    indices = [0, 2, 9, 10, 8,]
    def __init__(self, root_dir, output_size=(512, 512)):
        super(SelectedCelebAMaskHQDataset, self).__init__(root_dir, output_size)
        self.selected_indices = [CelebAMaskHQDataset.categories.index(cat) for cat in self.selected_categories]

    def __getitem__(self, idx):
        image, masks_all = super(SelectedCelebAMaskHQDataset, self).__getitem__(idx)
        masks_selected = masks_all[self.selected_indices]
        return image, masks_selected
    

class MergedCelebAMaskHQDataset(CelebAMaskHQDataset):
    # merged_categories = ['cloth', 'ear', 'hair', 'brow', 'eye', 'mouth', 'neck', 'nose', 'skin', 'hat', 'eye_g', 'neck_l']
    merged_categories = ['hair', 'hat', 'eye_g', 'skin',]  # 'brow', 'eye', 'mouth', 'nose', ]
    def __init__(self, root_dir, output_size=(512, 512)):
        super(MergedCelebAMaskHQDataset, self).__init__(root_dir, output_size)
        self.output_size = output_size

    def __getitem__(self, idx):
        image, unmerged_masks = super(MergedCelebAMaskHQDataset, self).__getitem__(idx)

        # Convert Tensor images to PIL for operations
        image = transforms.ToPILImage()(image)
        unmerged_masks = [transforms.ToPILImage()(mask) for mask in unmerged_masks]

        masks = []
        mask_mapping = {
            'ear': ['l_ear', 'r_ear'],
            'brow': ['l_brow', 'r_brow'],
            'eye': ['l_eye', 'r_eye'],
            'mouth': ['l_lip', 'u_lip', 'mouth']
        }

        for category in self.merged_categories:
            if category in mask_mapping:
                combined_mask_np = np.zeros_like(np.array(unmerged_masks[0])) # Initialize with zeros
                for sub_category in mask_mapping[category]:
                    sub_cat_idx = CelebAMaskHQDataset.categories.index(sub_category)
                    
                    mask_to_merge_np = np.array(unmerged_masks[sub_cat_idx])
                    
                    # Use logical or operation to combine masks
                    combined_mask_np = np.logical_or(combined_mask_np, mask_to_merge_np).astype(np.uint8)
                
                masks.append(Image.fromarray(combined_mask_np * 255, 'L'))  # Convert back to PIL and append
            else:
                cat_idx = CelebAMaskHQDataset.categories.index(category)
                masks.append(unmerged_masks[cat_idx])

        # Convert back to Tensor
        image = transforms.ToTensor()(image)
        masks = torch.stack([transforms.ToTensor()(m) for m in masks], dim=0).squeeze(1)

        return image, masks


class AugmentedDataset(Dataset):
    def __init__(self, dataset_source:Dataset, flip_prob=0.5, crop_ratio=(0.8, 0.8), scale_factor=(0.5, 2), output_size=(512, 512), 
                 noise_level=(0, 10), blur_radius=(0, 2), brightness_factor=(0.75, 1.25), pil=False, seed:int=None):
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
            # "noise_level": np.random.uniform(self.noise_level[0], self.noise_level[1]),
            # "blur_value": np.random.uniform(self.blur_radius[0], self.blur_radius[1]),
            # "brightness_value": np.random.uniform(self.brightness_factor[0], self.brightness_factor[1]),
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

        # Calculate random color for padding if necessary
        random_color = self._get_random_color(image)

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
            random_color = self._get_random_color(image)
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
        image, masks = self.source.__getitem__(idx)

        # Generate transform params for this sample
        transform_params = self._get_transform_params()

        # Apply the same augmentations to image and masks using the generated params
        image, ori_image = self._apply_image_transforms(image, transform_params)
        masks = [self._apply_common_transforms(mask, transform_params) for mask in masks]

        image = transforms.ToTensor()(image)
        ori_image = transforms.ToTensor()(ori_image)
        masks = torch.stack([transforms.ToTensor()(m) for m in masks], dim=0).squeeze(1)

        return image, masks, ori_image
    
    def __len__(self):
        return self.source.__len__()
    
    def _get_random_color(self, image):
        mode = image.mode
        if mode == 'RGB':
            return tuple(np.random.randint(0, 256, size=3).tolist())
        elif mode == 'RGBA':
            return tuple(np.random.randint(0, 256, size=4).tolist())
        elif mode == 'L':
            # For Gray, ZERO!!!
            return 0


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
        'background':     0,
        'hat':            1,
        'hair':           2,
        'sunglass':       3,
        'upper-clothes':  4,
        'skirt':          5,
        'pants':          6,
        'dress':          7,
        'belt':           8,
        'left-shoe':      9,
        'right-shoe':    10,
        'face':          11,
        'left-leg':      12,
        'right-leg':     13,
        'left-arm':      14,
        'right-arm':     15,
        'bag':           16,
        'scarf':         17,
    }

class LIPDataset(Dataset):
    data_map = {
        'hat':           1,
        'hair':          2,
        'glove':         3,
        'sunglasses':    4,
        'upperclothes':  5,
        'dress':         6,
        'coat':          7,
        'socks':         8,
        'pants':         9,
        'jumpsuits':    10,
        'scarf':        11,
        'skirt':        12,
        'face':         13,
        'left-arm':     14,
        'right-arm':    15,
        'left-leg':     16,
        'right-leg':    17,
        'left-shoe':    18,
        'right-shoe':   19,
    }
    class_ids = [5, 6, 7, 10, 2, 1, 4, 13]
    class_ids = [2, 1, 4, 13]
    # categories = ['cloth', 'hair', 'hat', 'eye_g', 'skin',]
    categories = ['hair', 'hat', 'eye_g', 'skin',]

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
        masks = torch.Tensor(self._generate_class_masks(np.array(Image.open(mask_path)).astype(np.uint8), self.class_ids, []))

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
    dataset = LIPDataset(root_dir=r"D:\workspace\LIP")
    dataset = AugmentedDataset(
        dataset_source=dataset, output_size=(256,256), 
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

    for batch_idx, (image_batch, masks_batch, _) in enumerate(dataloader):
        # Because we are using a batch size of 1, we need to squeeze the batch dimension
        image = image_batch.squeeze(0)
        masks = masks_batch.squeeze(0)

        # Show the original image
        show_image(image, "Image")

        # Show some masks
        # Change these indices to see different masks
        show_masks(masks, LIPDataset.categories, range(len(LIPDataset.categories)))
        
        plt.show()

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
