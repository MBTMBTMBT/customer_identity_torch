from tqdm import tqdm
import random

from utils import *


# def _scale_images_uniformly(images: torch.Tensor, scale_factor: float):
#     """
#     Scale a batch of images with a uniform scale factor, ensuring even dimensions.
#
#     :param images: A torch tensor of shape (batch, channel, h, w)
#     :param scale_factor: Scale factor
#     :return: Scaled images with even dimensions
#     """
#     assert 0 < scale_factor < 1
#
#     # Compute new size and ensure it's even
#     new_h, new_w = int(images.shape[2] * scale_factor), int(images.shape[3] * scale_factor)
#     new_h = new_h + 1 if new_h % 2 != 0 else new_h
#     new_w = new_w + 1 if new_w % 2 != 0 else new_w
#
#     # Resize all images in the batch
#     scaled_images = F.interpolate(images, size=(new_h, new_w), mode='bilinear', align_corners=False)
#
#     return scaled_images


def _scale_images_uniformly(images: torch.Tensor, scale_factor: float):
    """
    Scale a batch of images with a uniform scale factor, ensuring even dimensions.

    :param images: A torch tensor of shape (batch, channel, h, w)
    :param scale_factor: Scale factor
    :return: Scaled images with even dimensions
    """
    # Ensure scale factor is positive
    assert scale_factor > 0, "Scale factor must be positive"

    # Compute new size and ensure it's even
    new_h, new_w = int(images.shape[2] * scale_factor), int(images.shape[3] * scale_factor)
    new_h = new_h + 1 if new_h % 2 != 0 else new_h
    new_w = new_w + 1 if new_w % 2 != 0 else new_w

    # Ensure new dimensions are valid
    assert new_h > 0 and new_w > 0, "New dimensions must be positive integers"

    # Resize all images in the batch
    scaled_images = F.interpolate(images, size=(new_h, new_w), mode='bilinear', align_corners=False)

    return scaled_images


def train_CelebA(model, optimizer, train_loader, criterion_mask, criterion_pred, scale_range, epoch, device, mode=0):
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    running_loss = 0.0
    mask_running_loss = 0.0
    pred_running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
    for i, batch in enumerate(progress_bar):
        inputs, mask_labels, attributes, = batch

        # img = inputs.clone().squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)
        # # _img = _inputs.clone().squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)
        # masks = mask_labels.clone().squeeze(0).numpy().astype(np.uint8)
        # mask_list = [masks[i,:,:] for i in range(masks.shape[0])]
        # _classes = classes.clone().detach().squeeze(0).numpy().astype(np.uint8)
        # class_list = [_classes[i].item() > 0.5 for i in range(_classes.shape[0])]
        # # _colour_labels = (colour_labels.clone().detach().squeeze(0)*255).numpy().astype(np.uint8)
        # colour_list = [np.array([255,255,255], dtype=np.uint8) for i in range(len(class_list))]
        # combined_image = plot_with_matplotlib(
        #     img, 
        #     ['hair', 'hat', 'eye_g', 'skin',],  # 'brow', 'eye', 'mouth', 'nose', ],
        #     mask_list,
        #     class_list + [None for _ in range(len(mask_list) - len(class_list))],
        #     colour_list,
        #     )
        # cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Display', 800, 400)  # adjust the size as needed
        # combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('Display', combined_image)

        attributes = attributes.to(device)
        inputs, mask_labels = inputs.to(device), mask_labels.to(device)

        # Select a uniform scale for the entire batch
        scale_factor = random.uniform(*scale_range)
        inputs, mask_labels = _scale_images_uniformly(inputs, scale_factor), _scale_images_uniformly(mask_labels,
                                                                                                     scale_factor)

        optimizer.zero_grad()

        pred_masks, pred_classes = model(inputs)
        mask_loss = criterion_mask(pred_masks, mask_labels)
        # pred_loss, cl_loss, rg_loss = criterion_pred(pred_classes, classes, pred_colours)  #, colour_labels)
        pred_loss = criterion_pred(pred_classes, attributes)
        loss = mask_loss + pred_loss

        # Assume `predictions` and `classes` are your model's predictions and true class labels respectively
        predictions = pred_classes > 0.5
        # Create a mask for where the true class labels are 1 (positive class)
        positive_class_mask = (attributes == 1)
        # Select predictions and true labels where true labels are 1
        positive_predictions = predictions[positive_class_mask]
        positive_true_labels = attributes[positive_class_mask]
        # Calculate correct predictions for positive class
        correct_positives = (positive_predictions == positive_true_labels).float()
        # Calculate accuracy for positive class
        if correct_positives.numel() > 0:  # Check to make sure we have positive samples
            positive_accuracy = correct_positives.mean()
        else:
            positive_accuracy = torch.tensor(0.0)  # If no positive samples, set accuracy to 0
        # Now `positive_accuracy` will be the accuracy only for the class with label 1
        accuracy = positive_accuracy

        if mode == 1:
            # print('Training segmentation only.')
            model.unfreeze_segment_model()
            mask_loss.backward()
        elif mode == 0:
            # print('Training classification only.')
            model.freeze_segment_model()
            pred_loss.backward()
        else:
            # print('Training whole network.')
            model.unfreeze_segment_model()
            loss.backward()

        optimizer.step()

        running_loss += loss.item()
        mask_running_loss += mask_loss.item()
        pred_running_loss += pred_loss.item()
        # cl_running_loss += cl_loss.item()
        # rg_running_loss += rg_loss.item()

        running_accuracy += accuracy
        progress_bar.set_description(
            f'Train E{epoch}: ML:{mask_running_loss / (i + 1):.3f} PL:{pred_running_loss / (i + 1):.3f} Acc:{running_accuracy / (i + 1):.2f}')

    train_loss = running_loss / len(train_loader)
    mask_train_loss = mask_running_loss / len(train_loader)
    pred_train_loss = pred_running_loss / len(train_loader)
    return train_loss, mask_train_loss, pred_train_loss


def validate_CelebA(model, val_loader, criterion_mask, criterion_pred, epoch, device):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    running_loss = 0.0
    mask_running_loss = 0.0
    pred_running_loss = 0.0
    progress_bar = tqdm(val_loader, desc=f'Validation Epoch {epoch}')
    for i, batch in enumerate(progress_bar):
        inputs, mask_labels, attributes, = batch
        attributes = attributes.to(device)
        # colour_labels = colour_labels.to(device)
        inputs, mask_labels = inputs.to(device), mask_labels.to(device)

        total = len(val_loader)
        scale_factor = i / total * 1.2 + 0.3
        # scale_factor = random.uniform(0.2, 1)
        inputs, mask_labels = _scale_images_uniformly(inputs, scale_factor), _scale_images_uniformly(mask_labels,
                                                                                                     scale_factor)

        pred_masks, pred_classes = model(inputs)
        mask_loss = criterion_mask(pred_masks, mask_labels)
        # pred_loss, cl_loss, rg_loss = criterion_pred(pred_classes, classes, pred_colours)  #, colour_labels)
        pred_loss = criterion_pred(pred_classes, attributes)
        loss = mask_loss + pred_loss

        # Assume `predictions` and `classes` are your model's predictions and true class labels respectively
        predictions = pred_classes > 0.5
        # Create a mask for where the true class labels are 1 (positive class)
        positive_class_mask = (attributes == 1)
        # Select predictions and true labels where true labels are 1
        positive_predictions = predictions[positive_class_mask]
        positive_true_labels = attributes[positive_class_mask]
        # Calculate correct predictions for positive class
        correct_positives = (positive_predictions == positive_true_labels).float()
        # Calculate accuracy for positive class
        if correct_positives.numel() > 0:  # Check to make sure we have positive samples
            positive_accuracy = correct_positives.mean()
        else:
            positive_accuracy = torch.tensor(0.0)  # If no positive samples, set accuracy to 0
        # Now `positive_accuracy` will be the accuracy only for the class with label 1
        accuracy = positive_accuracy

        running_loss += loss.item()
        mask_running_loss += mask_loss.item()
        pred_running_loss += pred_loss.item()
        # cl_running_loss += cl_loss.item()
        # rg_running_loss += rg_loss.item()
        running_accuracy += accuracy
        progress_bar.set_description(
            f'Val E{epoch}:  ML:{mask_running_loss / (i + 1):.3f} PL:{pred_running_loss / (i + 1):.3f} Acc:{running_accuracy / (i + 1):.2f}')

    val_loss = running_loss / len(val_loader)
    mask_val_loss = mask_running_loss / len(val_loader)
    pred_val_loss = pred_running_loss / len(val_loader)
    return val_loss, mask_val_loss, pred_val_loss


def test_CelebA(model, test_loader, criterion_mask, criterion_pred, epoch, device):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    running_loss = 0.0
    mask_running_loss = 0.0
    pred_running_loss = 0.0
    progress_bar = tqdm(test_loader, desc=f'Testing Epoch {epoch}')
    for i, batch in enumerate(progress_bar):
        inputs, mask_labels, attributes = batch
        attributes = attributes.to(device)
        # colour_labels = colour_labels.to(device)
        inputs, mask_labels = inputs.to(device), mask_labels.to(device)

        pred_masks, pred_classes = model(inputs)
        mask_loss = criterion_mask(pred_masks, mask_labels)
        # pred_loss, cl_loss, rg_loss = criterion_pred(pred_classes, classes, pred_colours)  #, colour_labels)
        pred_loss = criterion_pred(pred_classes, attributes)
        loss = mask_loss + pred_loss

        # Assume `predictions` and `classes` are your model's predictions and true class labels respectively
        predictions = pred_classes > 0.5
        # Create a mask for where the true class labels are 1 (positive class)
        positive_class_mask = (attributes == 1)
        # Select predictions and true labels where true labels are 1
        positive_predictions = predictions[positive_class_mask]
        positive_true_labels = attributes[positive_class_mask]
        # Calculate correct predictions for positive class
        correct_positives = (positive_predictions == positive_true_labels).float()
        # Calculate accuracy for positive class
        if correct_positives.numel() > 0:  # Check to make sure we have positive samples
            positive_accuracy = correct_positives.mean()
        else:
            positive_accuracy = torch.tensor(0.0)  # If no positive samples, set accuracy to 0
        # Now `positive_accuracy` will be the accuracy only for the class with label 1
        accuracy = positive_accuracy

        running_loss += loss.item()
        mask_running_loss += mask_loss.item()
        pred_running_loss += pred_loss.item()
        # cl_running_loss += cl_loss.item()
        # rg_running_loss += rg_loss.item()
        running_accuracy += accuracy
        progress_bar.set_description(
            f'Test E{epoch}: ML:{mask_running_loss / (i + 1):.3f} PL:{pred_running_loss / (i + 1):.3f} Acc:{running_accuracy / (i + 1):.2f}')

    test_loss = running_loss / len(test_loader)
    mask_test_loss = mask_running_loss / len(test_loader)
    pred_test_loss = pred_running_loss / len(test_loader)
    # cl_test_loss = cl_running_loss / len(test_loader)
    # rg_test_loss = rg_running_loss / len(test_loader)
    return test_loss, mask_test_loss, pred_test_loss  # , rg_test_loss


def train_CCP(model, optimizer, train_loader, criterion_mask, criterion_pred, scale_range, epoch, device, mode=0):
    model.train()
    running_accuracy = 0.0
    running_loss = 0.0
    mask_running_loss = 0.0
    pred_running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
    for i, batch in enumerate(progress_bar):
        inputs, mask_labels, attributes, has_pixel_labels, = batch

        attributes = attributes.to(device)
        inputs, mask_labels = inputs.to(device), mask_labels.to(device)
        has_pixel_labels = has_pixel_labels.to(device)
        has_pixel_labels = has_pixel_labels.view(-1, 1, 1, 1)

        # Select a uniform scale for the entire batch
        scale_factor = random.uniform(*scale_range)
        inputs, mask_labels = _scale_images_uniformly(inputs, scale_factor), _scale_images_uniformly(mask_labels,
                                                                                                     scale_factor)

        optimizer.zero_grad()

        pred_masks, pred_classes = model(inputs)
        # if there were no pixel labels from the dataset, do not compute there loss values
        pred_masks, mask_labels = torch.mul(pred_masks, has_pixel_labels), torch.mul(mask_labels, has_pixel_labels)
        mask_loss = criterion_mask(pred_masks, mask_labels)  # nn.BCELoss()
        # pred_loss, cl_loss, rg_loss = criterion_pred(pred_classes, classes, pred_colours)  #, colour_labels)
        pred_loss = criterion_pred(pred_classes, attributes)
        loss = mask_loss + pred_loss

        # Assume `predictions` and `classes` are your model's predictions and true class labels respectively
        predictions = pred_classes > 0.5
        # Create a mask for where the true class labels are 1 (positive class)
        positive_class_mask = (attributes == 1)
        # Select predictions and true labels where true labels are 1
        positive_predictions = predictions[positive_class_mask]
        positive_true_labels = attributes[positive_class_mask]
        # Calculate correct predictions for positive class
        correct_positives = (positive_predictions == positive_true_labels).float()
        # Calculate accuracy for positive class
        if correct_positives.numel() > 0:  # Check to make sure we have positive samples
            positive_accuracy = correct_positives.mean()
        else:
            positive_accuracy = torch.tensor(0.0)  # If no positive samples, set accuracy to 0
        # Now `positive_accuracy` will be the accuracy only for the class with label 1
        accuracy = positive_accuracy

        if mode == 1:
            # print('Training segmentation only.')
            model.unfreeze_segment_model()
            mask_loss.backward()
        elif mode == 0:
            # print('Training classification only.')
            model.freeze_segment_model()
            pred_loss.backward()
        else:
            # print('Training whole network.')
            model.unfreeze_segment_model()
            loss.backward()

        optimizer.step()

        running_loss += loss.item()
        mask_running_loss += mask_loss.item()
        pred_running_loss += pred_loss.item()
        # cl_running_loss += cl_loss.item()
        # rg_running_loss += rg_loss.item()

        running_accuracy += accuracy
        progress_bar.set_description(
            f'Train E{epoch}: ML:{mask_running_loss / (i + 1):.3f} PL:{pred_running_loss / (i + 1):.3f} Acc:{running_accuracy / (i + 1):.2f}')

    train_loss = running_loss / len(train_loader)
    mask_train_loss = mask_running_loss / len(train_loader)
    pred_train_loss = pred_running_loss / len(train_loader)
    return train_loss, mask_train_loss, pred_train_loss


def validate_CCP(model, val_loader, criterion_mask, criterion_pred, epoch, device):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    running_loss = 0.0
    mask_running_loss = 0.0
    pred_running_loss = 0.0
    progress_bar = tqdm(val_loader, desc=f'Validation Epoch {epoch}')
    for i, batch in enumerate(progress_bar):
        inputs, mask_labels, attributes, has_pixel_labels, = batch
        attributes = attributes.to(device)
        # colour_labels = colour_labels.to(device)
        inputs, mask_labels = inputs.to(device), mask_labels.to(device)
        has_pixel_labels = has_pixel_labels.to(device)
        has_pixel_labels = has_pixel_labels.view(-1, 1, 1, 1)

        total = len(val_loader)
        scale_factor = 1  # i / total * 1.2 + 0.3
        # scale_factor = random.uniform(0.2, 1)
        inputs, mask_labels = _scale_images_uniformly(inputs, scale_factor), _scale_images_uniformly(mask_labels,
                                                                                                     scale_factor)

        pred_masks, pred_classes = model(inputs)
        # if there were no pixel labels from the dataset, do not compute there loss values
        pred_masks, mask_labels = torch.mul(pred_masks, has_pixel_labels), torch.mul(mask_labels, has_pixel_labels)
        mask_loss = criterion_mask(pred_masks, mask_labels)
        # pred_loss, cl_loss, rg_loss = criterion_pred(pred_classes, classes, pred_colours)  #, colour_labels)
        pred_loss = criterion_pred(pred_classes, attributes)
        loss = mask_loss + pred_loss

        # Assume `predictions` and `classes` are your model's predictions and true class labels respectively
        predictions = pred_classes > 0.5
        # Create a mask for where the true class labels are 1 (positive class)
        positive_class_mask = (attributes == 1)
        # Select predictions and true labels where true labels are 1
        positive_predictions = predictions[positive_class_mask]
        positive_true_labels = attributes[positive_class_mask]
        # Calculate correct predictions for positive class
        correct_positives = (positive_predictions == positive_true_labels).float()
        # Calculate accuracy for positive class
        if correct_positives.numel() > 0:  # Check to make sure we have positive samples
            positive_accuracy = correct_positives.mean()
        else:
            positive_accuracy = torch.tensor(0.0)  # If no positive samples, set accuracy to 0
        # Now `positive_accuracy` will be the accuracy only for the class with label 1
        accuracy = positive_accuracy

        running_loss += loss.item()
        mask_running_loss += mask_loss.item()
        pred_running_loss += pred_loss.item()
        # cl_running_loss += cl_loss.item()
        # rg_running_loss += rg_loss.item()
        running_accuracy += accuracy
        progress_bar.set_description(
            f'Val E{epoch}:  ML:{mask_running_loss / (i + 1):.3f} PL:{pred_running_loss / (i + 1):.3f} Acc:{running_accuracy / (i + 1):.2f}')

    val_loss = running_loss / len(val_loader)
    mask_val_loss = mask_running_loss / len(val_loader)
    pred_val_loss = pred_running_loss / len(val_loader)
    return val_loss, mask_val_loss, pred_val_loss
