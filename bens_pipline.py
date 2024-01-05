import cv2
import torch
import numpy as np
import bodypix
import pyttsx3
from models import *
from utils import *


# setups
face_th_rate = 0.05
thresholds_mask = [
    0.5, 0.75, 0.25, 0.5,  # 0.5, 0.5, 0.5, 0.5,
]
thresholds_pred = [
    0.6, 0.8, 0.1, 0.5,
]
erosion_iterations = 1
dilation_iterations = 1
colour_distance_rate = 1.2
categories = ['hair', 'hat', 'glasses', 'face',]

# Load model 
model_dir = "saved-model"
cat_layers = 4

# CHANGE THESE:
# segment_model = DeepLabV3PlusMobileNetV3(num_classes=4)  # ['hair', 'hat', 'eye_g', 'skin', ] 'brow', 'eye', 'mouth', 'nose', ]
segment_model = UNetWithResnet18Encoder(num_classes=4)
# predict_model = MultiLabelMobileNetV3Large(4, 7)   # 'hair', 'hat', 'glasses', 'face', ; first three with colours, rgb
predict_model = MultiLabelResNet(num_labels=4, input_channels=7)
model = CombinedModel(segment_model, predict_model, cat_layers=cat_layers)
model.eval()

# Check for the latest saved model
latest_checkpoint = find_latest_checkpoint(model_dir)
if latest_checkpoint:
    print(f"Loading model from {latest_checkpoint}")
    model, _, start_epoch, best_val_loss = load_model(model, None, path=latest_checkpoint, cpu_only=True)
    start_epoch += 1
else:
    raise RuntimeError("No save model discovered under %s" % model_dir)

# initialize bodipix detector
bodypix_detector = bodypix.BodyPixDetector()

# prepare hair colour table
image_dict = load_images_to_dict('hair_colours')
hair_colour_table = generate_colour_table(image_dict, SPESIFIC_COLOURS)


def pad_image_to_even_dims(image):
    # Get the current shape of the image
    height, width, _ = image.shape

    # Calculate the padding needed for height and width
    height_pad = 0 if height % 2 == 0 else 1
    width_pad = 0 if width % 2 == 0 else 1

    # Pad the image. Pad the bottom and right side of the image
    padded_image = np.pad(image, ((0, height_pad), (0, width_pad), (0, 0)), mode='constant', constant_values=0)

    return padded_image


def process_head(head_frame, model, thresholds_mask, erosion_iterations, dilation_iterations, thresholds_pred):
    """
    Processes the head frame to extract class counts and color information for head-related classes.

    Args:
    - head_frame (np.ndarray): The head frame extracted by the BodyPix model.
    - model: A PyTorch model instance for classifying and predicting masks for head features.
    - thresholds_mask, erosion_iterations, dilation_iterations: Thresholds and iteration counts for binary erosion and dilation.
    - thresholds_pred: A list of prediction thresholds.

    Returns:
    - Tuple[dict, dict]: A tuple containing two dictionaries:
        - head_class_count: A dictionary with counts for each head-related class.
        - head_class_colours: A dictionary with color information for each head-related class.
    """
    head_class_count = {
        'hair': 0,
        'hat': 0,
        'glasses': 0,
    }
    head_class_colours = {
        'hair': {},
        'hat': {},
        'glasses': {},
    }

    if head_frame is not None:
        try:
            _head_frame_bgr = cv2.cvtColor(head_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('Head Frame', _head_frame_bgr)
        except Exception as ignore:
            pass

        # Convert head frame to PyTorch tensor and normalize
        head_frame_tensor = torch.from_numpy(head_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        masks_batch_pred, pred_classes = model(head_frame_tensor)

        # Apply binary erosion and dilation to the masks
        processed_masks = binary_erosion_dilation(
            masks_batch_pred, thresholds=thresholds_mask, 
            erosion_iterations=erosion_iterations, dilation_iterations=dilation_iterations
        )
        masks = processed_masks.detach().squeeze(0).numpy().astype(np.uint8)
        mask_list = [masks[i,:,:] for i in range(masks.shape[0])]
        pred_classes = pred_classes.detach().squeeze(0).numpy()

        # Determine if each class is present
        class_list = [pred_classes[i].item() > thresholds_pred[i] for i in range(pred_classes.shape[0])]

        # Update class count
        for each_class, k in zip(class_list[0:3], ['hair', 'hat', 'glasses']):
            head_class_count[k] = int(each_class)

        # Update class colours
        for f, each_mask, k, c_map in zip([head_frame, head_frame, head_frame], mask_list[0:2], ['hair', 'hat', 'glasses'], [SPESIFIC_COLOURS, DETAILED_COLOURS, DETAILED_COLOURS]):
            colours = count_colours_in_masked_area(f, each_mask, c_map, sort=True)[1]
            for colour in colours:
                if colour[0] not in head_class_colours[k]:
                    head_class_colours[k][colour[0]] = [colour[1]]
                else:
                    head_class_colours[k][colour[0]].append(colour[1])

    return head_class_count, head_class_colours


def process_cloth(full_frame, torso_mask):
    """
    Processes the full frame with the torso mask to extract class counts and color information for cloth.

    Args:
    - full_frame (np.ndarray): The full original frame from the video source.
    - torso_mask (np.ndarray): The torso mask extracted by the BodyPix model.

    Returns:
    - Tuple[dict, dict]: A tuple containing two dictionaries:
        - cloth_class_count: A dictionary with counts for the cloth class.
        - cloth_class_colours: A dictionary with color information for the cloth class.
    """
    cloth_class_count = {
        'cloth': 0,
    }
    cloth_class_colours = {
        'cloth': {},
    }

    # Check if cloth is detected
    if torso_mask is not None and np.sum(torso_mask) >= 50:
        cloth_class_count['cloth'] = 1

        # Update cloth colours
        colours = count_colours_in_masked_area(full_frame, torso_mask, DETAILED_COLOURS, sort=True)[1]
        for colour in colours:
            if colour[0] not in cloth_class_colours['cloth']:
                cloth_class_colours['cloth'][colour[0]] = [colour[1]]
            else:
                cloth_class_colours['cloth'][colour[0]].append(colour[1])

    return cloth_class_count, cloth_class_colours


# you can use this function directly for prediction.
def predict_frame(head_frame, torso_frame, full_frame, head_mask, torso_mask, model, thresholds_mask, erosion_iterations, dilation_iterations, thresholds_pred):
    """
    Predicts classes and color information for a single processed video frame.

    Args:
    - head_frame (np.ndarray): The head frame extracted by the BodyPix model.
    - full_frame (np.ndarray): The full original frame from the video source.
    - head_mask (np.ndarray): The head mask extracted by the BodyPix model.
    - torso_mask (np.ndarray): The torso mask extracted by the BodyPix model.
    - model: A PyTorch model instance for classifying and predicting masks for head features.
    - thresholds_mask, erosion_iterations, dilation_iterations: Thresholds and iteration counts for binary erosion and dilation.
    - thresholds_pred: A list of prediction thresholds.

    Returns:
    - Tuple[dict, dict]: A tuple containing:
        - class_pred: A dictionary with predicted classes for the single frame.
        - colour_pred: A dictionary with predicted colors for the single frame.
    """
    class_count = {
        'hair': 0,
        'hat': 0,
        'glasses': 0,
        'cloth': 0,
    }
    class_colours = {
        'hair': {},
        'hat': {},
        'glasses': {},
        'cloth': {},
    }

    head_frame = pad_image_to_even_dims(head_frame)
    torso_frame = pad_image_to_even_dims(torso_frame)

    # Process head and cloth separately for the single frame
    head_class_count, head_class_colours = process_head(head_frame, model, thresholds_mask, erosion_iterations, dilation_iterations, thresholds_pred)
    cloth_class_count, cloth_class_colours = process_cloth(full_frame, torso_mask)

    # Update class counts and colours
    for k in head_class_count:
        class_count[k] = head_class_count[k]
        class_colours[k] = head_class_colours[k]

    class_count['cloth'] = cloth_class_count['cloth']
    class_colours['cloth'] = cloth_class_colours['cloth']

    # Compute final class predictions and colors for the single frame
    class_pred = {k: bool(class_count[k]) for k in class_count}
    colour_pred = {k: v for k, v in class_colours.items()}

    return class_pred, colour_pred


# if able to provide multiple frames (see __main__ seciton), then this should work better than the single frame version.
def predict_frames(head_frames, torso_frames, full_frames, head_masks, torso_masks, model, thresholds_mask, erosion_iterations, dilation_iterations, thresholds_pred, SPESIFIC_COLOURS):
    """
    Predicts classes and color information for a sequence of processed video frames.

    Args:
    - head_frames (list[np.ndarray]): List of head frames extracted by the BodyPix model.
    - torso_frames (list[np.ndarray]): List of body frames extracted by the BodyPix model.
    - full_frames (list[np.ndarray]): List of full original frames from the video source.
    - head_masks (list[np.ndarray]): List of head masks extracted by the BodyPix model.
    - torso_masks (list[np.ndarray]): List of torso masks extracted by the BodyPix model.
    - model: A PyTorch model instance for classifying and predicting masks for head features.
    - thresholds_mask, erosion_iterations, dilation_iterations: Thresholds and iteration counts for binary erosion and dilation.
    - thresholds_pred: A list of prediction thresholds.
    - SPESIFIC_COLOURS: A dictionary of specific colors.

    Returns:
    - Tuple[dict, dict]: A tuple containing:
        - class_pred: A dictionary with predicted classes.
        - colour_pred: A dictionary with predicted colors.
    """
    total_class_count = {
        'hair': [],
        'hat': [],
        'glasses': [],
        'cloth': [],
    }
    total_class_colours = {
        'hair': {},
        'hat': {},
        'glasses': {},
        'cloth': {},
    }

    for head_frame, torso_frame, full_frame, head_mask, torso_mask in zip(head_frames, torso_frames, full_frames, head_masks, torso_masks):
        head_frame = pad_image_to_even_dims(head_frame)
        torso_frame = pad_image_to_even_dims(torso_frame)

        # Process head and cloth separately
        head_class_count, head_class_colours = process_head(head_frame, model, thresholds_mask, erosion_iterations, dilation_iterations, thresholds_pred)
        cloth_class_count, cloth_class_colours = process_cloth(full_frame, torso_mask)

        # Accumulate class counts and colours
        for k in head_class_count:
            total_class_count[k].append(head_class_count[k])
            if k in head_class_colours:
                for colour, count in head_class_colours[k].items():
                    if colour not in total_class_colours[k]:
                        total_class_colours[k][colour] = count
                    else:
                        total_class_colours[k][colour].extend(count)

        total_class_count['cloth'].append(cloth_class_count['cloth'])
        for colour, count in cloth_class_colours['cloth'].items():
            if colour not in total_class_colours['cloth']:
                total_class_colours['cloth'][colour] = count
            else:
                total_class_colours['cloth'][colour].extend(count)

    # Compute final class predictions and colors
    class_pred = {k: sum(v) >= len(v) / 2 for k, v in total_class_count.items()}
    colour_pred = average_colours_by_label(total_class_count, total_class_colours)

    return class_pred, colour_pred


def speak(words: str):
    engine = pyttsx3.init()
    engine.say(words)
    engine.runAndWait()


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    count = 0
    _count = 56  # Skip initial frames for camera stabilization

    head_frames, torso_frames, full_frames, head_masks, torso_masks = [], [], [], [], []

    speak("Starting the detection process.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if _count > 0:
            _count -= 1
            continue

        # BodyPix detection
        head_frame, torso_frame, head_mask, torso_mask = bodypix_detector.detect(frame_rgb, return_masks=True)
        head_mask = np.squeeze(head_mask, -1)
        torso_mask = np.squeeze(torso_mask, -1)

        head_frames.append(head_frame)
        torso_frames.append(torso_frame)
        full_frames.append(frame_rgb)
        head_masks.append(head_mask)
        torso_masks.append(torso_mask)

        if count % 7 == 0:
            print("Analyzing frames...")
            speak("Analyzing frames...")

            class_pred, class_colours = predict_frames(head_frames, torso_frames, full_frames, head_masks, torso_masks, model, thresholds_mask, erosion_iterations, dilation_iterations, thresholds_pred, SPESIFIC_COLOURS)
            print("Predicted Classes:", class_pred)

            # Process color distributions for hair
            sorted_distances_hair = compare_colour_distributions(class_colours['hair'], hair_colour_table)[:3]
            print("Top Possible Hair Colours:", sorted_distances_hair)

            # Extract top colours for hat and cloth
            class_colours['hat'] = extract_top_colours_by_threshold(class_colours['hat'], 0.66)
            class_colours['cloth'] = extract_top_colours_by_threshold(class_colours['cloth'], 0.66)
            print("Major Hat Colours:", class_colours['hat'])
            print("Major Cloth Colours:", class_colours['cloth'])

            # Announce and report detected classes and colours
            speak("I can see...")
            _c = 0
            for _, k in enumerate(class_pred.keys()):
                if class_pred[k]:
                    speak(k)
                    _c += 1
            if _c == 0:
                speak("Nothing.")

            if class_pred['hat']:
                hat_colour = str(class_colours['hat'][0][0]) if class_colours['hat'] else "Unknown"
                print("Hat is likely to be:", hat_colour)
                speak(f"Hat is likely to be: {hat_colour}")

            if class_pred['hair']:
                hair_colour = str(sorted_distances_hair[0][0]) if sorted_distances_hair else "Unknown"
                print("Hair is likely to be:", hair_colour)
                speak(f"Hair is likely to be: {hair_colour}")

            if class_pred['cloth']:
                cloth_colour = str(class_colours['cloth'][0][0]) if class_colours['cloth'] else "Unknown"
                print("Cloth is likely to be:", cloth_colour)
                speak(f"Cloth is likely to be: {cloth_colour}")

            # Reset buffers for the next set of frames
            head_frames, full_frames, head_masks, torso_masks = [], [], [], []

        count += 1

    cap.release()
