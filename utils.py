import torch
import torch.nn.functional as F
import numpy as np
import os
import re
import cv2
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import convolve
from torchvision import transforms

COLOURS = {
    "red": [255, 0, 0],
    "green": [0, 255, 0],
    "blue": [0, 0, 255],
    "white": [255, 255, 255],
    "black": [0, 0, 0],
    "yellow": [255, 255, 0],
    "cyan": [0, 255, 255],
    "magenta": [255, 0, 255],
    "gray": [128, 128, 128],
    "orange": [255, 165, 0],
    "purple": [128, 0, 128],
    "brown": [139, 69, 19],
    "pink": [255, 182, 193],
    "beige": [245, 245, 220],
    "maroon": [128, 0, 0],
    "olive": [128, 128, 0],
    "navy": [0, 0, 128],
    "lime": [50, 205, 50],
    "golden": [255, 223, 0],
    "teal": [0, 128, 128],
    "coral": [255, 127, 80],
    "salmon": [250, 128, 114],
    "turquoise": [64, 224, 208],
    "violet": [238, 130, 238],
    "platinum": [229, 228, 226],
    "ochre": [204, 119, 34],
    "burntsienna": [233, 116, 81],
    "chocolate": [210, 105, 30],
    "tan": [210, 180, 140],
    "ivory": [255, 255, 240],
    "goldenrod": [218, 165, 32],
    "orchid": [218, 112, 214],
    "honey": [238, 220, 130]
}

SPESIFIC_COLOURS = {
    "red": [255, 0, 0],
    "green": [0, 255, 0],
    "blue": [0, 0, 255],
    "white": [255, 255, 255],
    "black": [0, 0, 0],
    "yellow": [255, 255, 0],
    "cyan": [0, 255, 255],
    "magenta": [255, 0, 255],
    "gray": [128, 128, 128],
    "orange": [255, 165, 0],
    "purple": [128, 0, 128],
    "brown": [139, 69, 19],
    "pink": [255, 182, 193],
    "beige": [245, 245, 220],
    "maroon": [128, 0, 0],
    "olive": [128, 128, 0],
    "navy": [0, 0, 128],
    "lime": [50, 205, 50],
    "golden": [255, 223, 0],
    "teal": [0, 128, 128],
    "coral": [255, 127, 80],
    "salmon": [250, 128, 114],
    "turquoise": [64, 224, 208],
    "violet": [238, 130, 238],
    "platinum": [229, 228, 226],
    "ochre": [204, 119, 34],
    "burntsienna": [233, 116, 81],
    "chocolate": [210, 105, 30],
    "tan": [210, 180, 140],
    "ivory": [255, 255, 240],
    "goldenrod": [218, 165, 32],
    "orchid": [218, 112, 214],
    "honey": [238, 220, 130],
    "lavender": [230, 230, 250],
    "mint": [189, 252, 201],
    "peach": [255, 229, 180],
    "ruby": [224, 17, 95],
    "indigo": [75, 0, 130],
    "amber": [255, 191, 0],
    "emerald": [80, 200, 120],
    "sapphire": [15, 82, 186],
    "aquamarine": [127, 255, 212],
    "periwinkle": [204, 204, 255],
    "fuchsia": [255, 0, 255],
    "raspberry": [227, 11, 92],
    "slate": [112, 128, 144],
    "charcoal": [54, 69, 79]
}

DETAILED_COLOURS = {
    "light_red": [255, 204, 204],
    "bright_red": [255, 0, 0],
    "dark_red": [139, 0, 0],
    "light_green": [204, 255, 204],
    "bright_green": [0, 255, 0],
    "dark_green": [0, 100, 0],
    "light_blue": [204, 204, 255],
    "bright_blue": [0, 0, 255],
    "dark_blue": [0, 0, 139],
    "light_yellow": [255, 255, 204],
    "bright_yellow": [255, 255, 0],
    "dark_yellow": [204, 204, 0],
    "light_cyan": [204, 255, 255],
    "bright_cyan": [0, 255, 255],
    "dark_cyan": [0, 139, 139],
    "light_magenta": [255, 204, 255],
    "bright_magenta": [255, 0, 255],
    "dark_magenta": [139, 0, 139],
    "light_orange": [255, 229, 204],
    "bright_orange": [255, 165, 0],
    "dark_orange": [255, 140, 0],
    "light_purple": [229, 204, 255],
    "bright_purple": [128, 0, 128],
    "dark_purple": [102, 0, 102],
    "light_pink": [255, 204, 229],
    "bright_pink": [255, 105, 180],
    "dark_pink": [255, 20, 147],
    "light_brown": [210, 180, 140],
    "medium_brown": [165, 42, 42],
    "dark_brown": [101, 67, 33],
    # ...
}

COLOUR_FAMILIES = {
    "light_reds": [[255, 182, 193], [255, 192, 203], [255, 160, 122]],
    "dark_reds": [[139, 0, 0], [178, 34, 34], [165, 42, 42]],
    "light_blues": [[173, 216, 230], [135, 206, 250], [176, 224, 230]],
    "dark_blues": [[0, 0, 139], [25, 25, 112], [0, 0, 128]],
    "bluish_greens": [[102, 205, 170], [32, 178, 170], [72, 209, 204]],
    "light_greens": [[144, 238, 144], [152, 251, 152], [143, 188, 143]],
    "dark_greens": [[0, 100, 0], [34, 139, 34], [47, 79, 79]],
    "yellows": [[255, 255, 0], [255, 255, 102], [255, 215, 0]],
    "oranges": [[255, 165, 0], [255, 140, 0], [255, 69, 0]],
    "purples": [[128, 0, 128], [147, 112, 219], [138, 43, 226]],
    "pinks": [[255, 192, 203], [255, 182, 193], [255, 105, 180]],
    "browns": [[165, 42, 42], [139, 69, 19], [160, 82, 45]],
    "cyans": [[0, 255, 255], [0, 139, 139], [72, 209, 204]],
    "greys": [[128, 128, 128], [169, 169, 169], [192, 192, 192]],
    # ...
}

SIMPLIFIED_COLOURS = {
    "red": [255, 0, 0],
    "green": [0, 255, 0],
    "blue": [0, 0, 255],
    "white": [255, 255, 255],
    "black": [0, 0, 0],
    "yellow": [255, 255, 0],
    "gray": [128, 128, 128],
    "orange": [255, 165, 0],
    "purple": [128, 0, 128],
    "pink": [255, 182, 193],
    "light blue": [173, 216, 230],
    "dark green": [0, 100, 0],
    "light gray": [211, 211, 211],
    "dark red": [139, 0, 0],
    "beige": [245, 245, 220],
    "navy": [0, 0, 128]
}

HAIR_COLOURS = {
    'midnight black': (9, 8, 6),
    'off black': (44, 34, 43),
    'strong dark brown': (58, 48, 36),
    'medium dark brown': (78, 67, 63),

    'chestnut brown': (106, 78, 66),
    'light chestnut brown': (106, 78, 66),
    'dark golden brown': (95, 72, 56),
    'light golden brown': (167, 133, 106),

    'dark honey blonde': (184, 151, 128),
    'bleached blonde': (220, 208, 186),
    'light ash blonde': (222, 288, 153),
    'light ash brown': (151, 121, 97),

    'lightest blonde': (230, 206, 168),
    'pale golden blonde': (229, 200, 168),
    'strawberry blonde': (165, 137, 70),
    'light auburn': (145, 85, 61),

    'dark auburn': (83, 61, 53),
    'darkest gray': (113, 99, 93),
    'medium gray': (183, 166, 158),
    'light gray': (214, 196, 194),

    'white blonde': (255, 24, 225),
    'platinum blonde': (202, 191, 177),
    'russet red': (145, 74, 67),
    'terra cotta': (181, 82, 57)
}


def save_model(epoch, model, optimizer, best_val_loss, path: str, counter=-1):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'counter': counter,
    }, path)
    print('Saved: epoch {}, counter {}, path {}.'.format(epoch, counter, path))


def load_model(model, optimizer, path="model.pth", cpu_only=False):
    if cpu_only:
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    # print(set(checkpoint.keys()))
    # print('counter' in set(checkpoint.keys()))
    if 'counter' in checkpoint.keys():
        counter = checkpoint['counter']
        return model, optimizer, epoch, best_val_loss, counter
    return model, optimizer, epoch, best_val_loss


def find_latest_checkpoint(model_dir):
    """Find the latest model checkpoint in the given directory."""
    checkpoints = [f for f in os.listdir(model_dir) if f.startswith('model_epoch_') and f.endswith('.pth')]
    if not checkpoints:
        return None

    # Extracting the epoch number from the model filename using regex
    checkpoints.sort(key=lambda x: int(re.search(r'(\d+)', x).group()))
    return os.path.join(model_dir, checkpoints[-1])


def binary_erosion_dilation(tensor, thresholds, erosion_iterations=1, dilation_iterations=1):
    """
    Apply binary threshold, followed by erosion and dilation to a tensor.

    :param tensor: Input tensor (N, C, H, W)
    :param thresholds: List of threshold values for each channel
    :param erosion_iterations: Number of erosion iterations
    :param dilation_iterations: Number of dilation iterations
    :return: Processed tensor
    """

    # Check if the length of thresholds matches the number of channels
    if len(thresholds) != tensor.size(1):
        raise ValueError("Length of thresholds must match the number of channels")

    # Binary thresholding
    for i, threshold in enumerate(thresholds):
        tensor[:, i] = (tensor[:, i] > threshold / 2).float() / 4
        tensor[:, i] += (tensor[:, i] > threshold).float()
        tensor[:, i] /= max(tensor[:, i].clone())

    # Define the 3x3 kernel for erosion and dilation
    kernel = torch.tensor([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Replicate the kernel for each channel
    kernel = kernel.repeat(tensor.size(1), 1, 1, 1).to(tensor.device)

    # Erosion
    for _ in range(erosion_iterations):
        tensor = F.conv2d(tensor, kernel, padding=1, groups=tensor.size(1))  # 3x3 convolution with groups
        tensor = (tensor == 9).float()  # Check if all neighboring pixels are 1

    # Dilation
    for _ in range(dilation_iterations):
        tensor_dilated = F.conv2d(tensor, kernel, padding=1, groups=tensor.size(1))  # 3x3 convolution with groups
        tensor = torch.clamp(tensor + tensor_dilated, 0, 1)  # Combine the original and dilated tensors

    return tensor


def find_contours(mask: np.ndarray) -> list[dict]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_list = []
    for contour in contours:
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        temp_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(temp_mask, [contour], -1, (1), thickness=cv2.FILLED)
        contour_dict = {
            'area': area,
            'centre': (cX, cY),
            'contour': contour,
            'mask': temp_mask,
            'width': w,
            'height': h,
            'bottom': cY + h / 2,
            'top': cY - h / 2,
        }
        contour_list.append(contour_dict)
    sorted_contours = sorted(contour_list, key=lambda x: x['area'], reverse=True)
    return sorted_contours


def merge_masks(mask1: np.ndarray, mask2: np.ndarray):
    merged_mask = np.logical_or(mask1, mask2).astype(np.uint8)
    return merged_mask


def closest_colours(requested_color: np.ndarray, colour_map=COLOURS) -> list[tuple[str, float]]:
    distances = {color: np.linalg.norm(np.array(rgb_val) - requested_color) for color, rgb_val in colour_map.items()}
    sorted_colors = sorted(distances.items(), key=lambda x: x[1])
    top_three_colors = sorted_colors[:3]
    formatted_colors = [(color_name, distance) for color_name, distance in top_three_colors]
    return formatted_colors


def avg_color_float(rgb_image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    avg_colors = torch.zeros((rgb_image.size(0), mask.size(1), rgb_image.size(1)), device=rgb_image.device)
    for i in range(rgb_image.size(0)):
        for j in range(mask.size(1)):
            for k in range(rgb_image.size(1)):
                valid_pixels = torch.masked_select(rgb_image[i, k], mask[i, j])
                avg_color = valid_pixels.float().mean() if valid_pixels.numel() > 0 else torch.tensor(0.0)
                avg_colors[i, j, k] = avg_color

    return avg_colors  # / 255.0


def median_color_float(rgb_image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    median_colors = torch.zeros((rgb_image.size(0), mask.size(1), rgb_image.size(1)), device=rgb_image.device)
    for i in range(rgb_image.size(0)):
        for j in range(mask.size(1)):
            for k in range(rgb_image.size(1)):
                valid_pixels = torch.masked_select(rgb_image[i, k], mask[i, j])
                if valid_pixels.numel() > 0:
                    median_value = valid_pixels.median()
                else:
                    median_value = torch.tensor(0.0, device=rgb_image.device)
                median_colors[i, j, k] = median_value
    return median_colors  # / 255.0


def plot_with_matplotlib(frame, categories, masks, predictions, colours):
    """Generate an image with matplotlib, showing the original frame and masks with titles and color overlays."""
    assert len(masks) == len(categories) == len(predictions), "Length of masks, categories, and predictions must match."

    num_masks = len(masks)
    cols = 3
    rows = (num_masks + 1) // cols + ((num_masks + 1) % cols > 0)  # Adding 1 for the frame
    position = range(1, num_masks + 2)  # +2 to include the frame in the count

    fig = plt.figure(figsize=(15, rows * 3))  # Adjust the size as needed

    # Add the frame as the first image
    ax = fig.add_subplot(rows, cols, 1)
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ax.imshow(frame)
    ax.set_title('Original Frame')
    ax.axis('off')

    # Iterate over the masks
    for i, idx in enumerate(position[1:], start=1):  # Skip 1 for the frame
        ax = fig.add_subplot(rows, cols, idx)

        # Create an RGB image for the colored mask
        colored_mask = np.stack([masks[i - 1]] * 3, axis=-1)  # i-1 because we skip the frame in position

        # Apply color if category is detected and color is provided
        if predictions[i - 1]:
            if (i - 1) < len(colours):
                color = np.array(colours[i - 1], dtype=np.uint8)  # Convert color to uint8
                color_mask = np.zeros_like(colored_mask)  # Initialize color_mask with the same shape as colored_mask
                color_mask[..., 0] = masks[i - 1] * color[0]  # Apply color channel 0
                color_mask[..., 1] = masks[i - 1] * color[1]  # Apply color channel 1
                color_mask[..., 2] = masks[i - 1] * color[2]  # Apply color channel 2
                # Now combine the colored mask with the original grayscale mask
                colored_mask = np.where(masks[i - 1][:, :, None], color_mask, colored_mask).astype(np.uint8)
                # Show the colored mask
                ax.imshow(colored_mask)
                # print(np.max(mask_image))
                # mask_image = masks[i-1]
                # ax.imshow(mask_image, cmap="gray")
            else:
                # If there's no color provided for this category, use white color
                mask_image = masks[i - 1]
                ax.imshow(mask_image, cmap="gray")
        else:
            # If the category is not detected, keep the mask black
            mask_image = masks[i - 1]
            ax.imshow(mask_image, cmap="gray")

        # mask_image = masks[i-1]
        # ax.imshow(mask_image, cmap="gray")

        # Set title with the detection status
        detection_status = 'yes' if predictions[i - 1] else 'no'
        ax.set_title(f"{categories[i - 1]} - {detection_status}")
        ax.axis('off')

    plt.tight_layout()
    fig.canvas.draw()

    # Retrieve buffer and close the plot to avoid memory issues
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return data


def count_colours_in_masked_area(img, mask, colours, filter_size=3, sort=False):
    """
    Counts the number of pixels of each color within the masked area of an image.

    Parameters:
    img (numpy.ndarray): An RGB image, with the shape (height, width, 3).
    mask (numpy.ndarray): A binary mask, with the shape (height, width), where 1 indicates the area of interest.
    colours (dict): A dictionary where keys are color names and values are the corresponding RGB values.
    filter_size (int): The size of the convolution filter to apply for smoothing the image, default is 3.
    sort (bool): Whether to return a sorted list of colors based on pixel count, default is False.

    Returns:
    dict: A dictionary containing the count of pixels for each color in the masked area.
    If sort is True, it also returns a list of tuples, each containing a color name, its proportion in the masked area, and the pixel count. This list is sorted in descending order based on pixel count.

    The function first applies an averaging filter to the image for smoothing. Then, it calculates the Euclidean distance of each pixel in the masked area to the predefined colors. It identifies the closest color for each pixel, counts the occurrences of each color, and creates a dictionary mapping colors to their respective counts. If sorting is requested, it also calculates the proportion of each color and returns a sorted list of colors based on their pixel count.
    """
    avg_filter = np.ones((filter_size, filter_size, 3)) / (filter_size ** 2)
    img_filtered = convolve(img, avg_filter, mode='constant', cval=0.0)
    colours_array = np.array(list(colours.values()))
    masked_img = img_filtered[mask == 1]
    distances = np.linalg.norm(masked_img[:, None] - colours_array, axis=2)
    closest_colours = np.argmin(distances, axis=1)
    unique, counts = np.unique(closest_colours, return_counts=True)
    colour_counts = {list(colours.keys())[i]: count for i, count in zip(unique, counts)}
    if sort:
        total_pixels = sum(counts)
        sorted_colours = sorted(((list(colours.keys())[i], count / total_pixels, count)
                                 for i, count in zip(unique, counts)), key=lambda item: item[2], reverse=True)
        return colour_counts, sorted_colours

    return colour_counts


def average_colours_by_label(labels, colours):
    """
    Computes the average values of colours associated with each label.

    Parameters:
    labels (dict): A dictionary where keys are label names and values are lists of binary values (0 or 1). Each list represents whether a certain feature (labelled by the key) is present (1) or not (0) in a set of instances.
    colours (dict): A dictionary where keys are label names and values are dictionaries. Each inner dictionary maps colour names to lists of values (e.g., pixel counts or intensities) associated with that colour for each instance.

    Returns:
    dict: A dictionary where keys are label names and values are sorted lists of tuples. Each tuple contains a colour name and its average value calculated only from instances where the label is present (1). The tuples are sorted by average values in descending order.

    The function iterates through each label, calculating the average value for each colour only from instances where the label value is 1 (present). It then sorts these average values in descending order for each label and returns this sorted list along with the label name in a dictionary.
    """
    averaged_colours = {}

    for label, label_values in labels.items():
        if label not in colours.keys():
            continue

        colour_values = colours[label]
        averages = {}

        for colour, values in colour_values.items():
            valid_values = [value for value, label_value in zip(values, label_values) if label_value == 1]
            if valid_values:
                averages[colour] = sum(valid_values) / len(valid_values)

        sorted_colours = sorted(averages.items(), key=lambda item: item[1], reverse=True)
        averaged_colours[label] = sorted_colours

    return averaged_colours


def load_images_to_dict(root_dir):
    """
    Load images from a specified directory into a dictionary, removing file extensions from the keys.

    Parameters:
    root_dir (str): The root directory containing the images.

    Returns:
    dict: A dictionary with image names (without extensions) as keys and their corresponding numpy arrays as values.
    """
    image_dict = {}
    for filename in os.listdir(root_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(root_dir, filename)
            # Read the image using OpenCV
            img = cv2.imread(img_path)
            # Convert it from BGR to RGB color space
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Remove the file extension from the filename
            name_without_extension = os.path.splitext(filename)[0]
            image_dict[name_without_extension] = img

    return image_dict


def generate_colour_table(image_dict: dict, colour_map: dict):
    """
    Generates a colour table for each image in the given dictionary, counting the colours in each image.

    Parameters:
    image_dict (dict): A dictionary where keys are image identifiers and values are image arrays in the format (height, width, 3).
    colour_map (dict): A dictionary mapping colour names to their respective RGB values.

    Returns:
    dict: A dictionary where keys are image identifiers and values are colour tables. Each colour table is generated by the 'count_colours_in_masked_area' function and contains a count of how many times each colour (as defined in colour_map) appears in the corresponding image.

    For each image in the image_dict, this function creates a mask that covers the entire image and uses 'count_colours_in_masked_area' to count the occurrences of each colour in the colour_map within the image. The results are stored in a new dictionary, mapping each image identifier to its corresponding colour table.
    """
    colour_table = {}
    for k in image_dict.keys():
        colour_table[k] = count_colours_in_masked_area(image_dict[k],
                                                       np.ones((image_dict[k].shape[0], image_dict[k].shape[1])),
                                                       colour_map, sort=True)
    return colour_table


def compare_colour_distributions(averaged_colours_list, colour_table_dict):
    """
    Compares colour distributions between an averaged colours list and a dictionary of colour tables by calculating the Euclidean distance.

    Parameters:
    averaged_colours_list (list): A list of tuples, where each tuple contains a colour name and its proportion. This is typically the output from 'average_colours_by_label'.
    colour_table_dict (dict): A dictionary where keys are image identifiers and values are colour tables. Each colour table is a list of tuples, each containing a colour name, its proportion in the image, and the pixel count.

    Returns:
    dict: A dictionary where keys are image identifiers and values are the Euclidean distances between the colour distribution in the image and the averaged_colours_list.

    The function iterates over each image's colour table in colour_table_dict. For each image, it calculates the Euclidean distance between the colour proportions in averaged_colours_list and the colour proportions in the image's colour table. The results are stored in a dictionary, mapping each image identifier to the calculated distance.
    """
    distances = {}

    avg_colours_dict = {colour: proportion for colour, proportion in averaged_colours_list}

    for image_name, colour_data in colour_table_dict.items():
        colour_proportions = {colour: proportion for colour, proportion, _ in colour_data[1]}

        common_colours = set(avg_colours_dict.keys()) & set(colour_proportions.keys())
        avg_values = [avg_colours_dict.get(colour, 0) for colour in common_colours]
        prop_values = [colour_proportions.get(colour, 0) for colour in common_colours]

        distances[image_name] = np.linalg.norm(np.array(avg_values) - np.array(prop_values))

    sorted_distances = sorted(distances.items(), key=lambda item: item[1])

    return sorted_distances


# Example usage
# sorted_distances = compare_colour_distributions(averaged_colours, colour_table)

def extract_top_colours_by_threshold(colour_list, threshold):
    """
    Extracts top colours based on a cumulative proportion threshold.

    Parameters:
    colour_list (list): A list of tuples, each being a 2-element (colour, proportion) or 
                        a 3-element (colour, proportion, count) tuple.
    threshold (float): A float between 0 and 1, representing the threshold for the cumulative proportion.

    Returns:
    list: A list of tuples (colour, proportion), sorted by proportion in descending order, 
          whose cumulative proportion just exceeds the threshold.
    """
    # Sort the list by proportion in descending order
    sorted_colours = sorted(colour_list, key=lambda x: x[1], reverse=True)

    # Extract top colours based on the cumulative proportion threshold
    cumulative_proportion = 0.0
    top_colours = []
    for colour in sorted_colours:
        cumulative_proportion += colour[1]
        top_colours.append((colour[0], colour[1]))
        if cumulative_proportion >= threshold:
            break

    return top_colours


def find_nearest_colour_family(colour, colour_families):
    """
    Determines the nearest colour family for a given colour.

    Parameters:
    colour (tuple): The colour in RGB format.
    colour_families (dict): A dictionary where keys are family names and values are lists of representative RGB colours for each family.

    Returns:
    str: The name of the nearest colour family.
    """
    min_distance = float('inf')
    nearest_family = None

    for family, representative_colours in colour_families.items():
        for rep_colour in representative_colours:
            distance = np.linalg.norm(np.array(colour) - np.array(rep_colour))
            if distance < min_distance:
                min_distance = distance
                nearest_family = family

    return nearest_family


def find_nearest_colour_family(colour, colour_families):
    """
    Determines the nearest colour family for a given colour based on the minimum Euclidean distance.

    Parameters:
    colour (tuple): The colour in RGB format.
    colour_families (dict): A dictionary where keys are family names and values are lists of representative RGB colours for each family.

    Returns:
    str: The name of the nearest colour family.
    """
    min_distance = float('inf')
    nearest_family = None

    # Convert colour to numpy array for distance calculation
    colour = np.array(colour)

    for family, family_colours in colour_families.items():
        for rep_colour in family_colours:
            # Calculate the Euclidean distance
            distance = np.linalg.norm(colour - np.array(rep_colour))
            if distance < min_distance:
                min_distance = distance
                nearest_family = family

    return nearest_family


if __name__ == "__main__":
    image_dict = load_images_to_dict('legacy/hair_colours')
    hair_colour_table = generate_colour_table(image_dict, HAIR_COLOURS)
    print(hair_colour_table)


def show_deepfashion2_image_masks_and_labels(image, masks, labels, bboxes, fig=None, axs=None):
    categories = [
        'short sleeve top', 'long sleeve top', 'short sleeve outwear',
        'long sleeve outwear', 'vest', 'sling', 'shorts',
        'trousers', 'skirt', 'short sleeve dress',
        'long sleeve dress', 'vest dress', 'sling dress'
    ]

    # Convert the image tensor to PIL Image for display
    image_pil = transforms.ToPILImage()(image)
    img_width, img_height = image_pil.size

    if fig is None or axs is None:
        fig, axs = plt.subplots(1, len(masks) + 1, figsize=(20, 3))
    else:
        # Clear the previous figures
        for ax in axs:
            ax.clear()

    # Plot the original image
    axs[0].imshow(image_pil)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    # Plot each mask
    for i, mask in enumerate(masks):
        axs[i + 1].imshow(image_pil, alpha=0.5)  # Show the underlying image
        axs[i + 1].imshow(mask, cmap='gray', alpha=0.5, interpolation='none')  # Overlay the mask
        axs[i + 1].set_title(f'{categories[i]}:\n{"%.1f" % labels[i]}')
        axs[i + 1].axis('off')

        # Add bounding boxes to the plot
        for cat_id, bbox in bboxes:
            if cat_id == i:
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
    plt.draw()

    return fig, axs
