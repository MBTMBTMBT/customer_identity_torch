import random

from sklearn.metrics import f1_score
from tqdm import tqdm

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


modes = ['seg', 'pred', 'mix']


def train(model, optimizer, train_loader, criterion_mask, criterion_pred, scale_range, epoch, device, mode='mix', tb_writer=None, counter=-1):
    assert mode in modes
    model.train()
    running_accuracy = 0.0
    running_loss = 0.0
    mask_running_loss = 0.0
    pred_running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
    for i, batch in enumerate(progress_bar):
        inputs, mask_labels, attributes, = batch

        _input, _mask_labels, _attributes = inputs[0].permute(1, 2, 0).cpu().numpy(), mask_labels[0].cpu().numpy(), attributes[0].cpu().numpy()

        # from datasets import show_deepfashion2_image_masks_and_labels
        # show_deepfashion2_image_masks_and_labels(_input, _mask_labels, _attributes)

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

        if mode == 'seg':
            # print('Training segmentation only.')
            model.unfreeze_segment_model()
            mask_loss.backward()
        elif mode == 'pred':
            # print('Training classification only.')
            model.freeze_segment_model()
            pred_loss.backward()
        else:
            # print('Training whole network.')
            try:
                model.unfreeze_segment_model()
            except Exception as ignore:
                pass
            loss.backward()

        optimizer.step()

        running_loss += loss.item()
        mask_running_loss += mask_loss.item()
        pred_running_loss += pred_loss.item()
        # cl_running_loss += cl_loss.item()
        # rg_running_loss += rg_loss.item()

        running_accuracy += accuracy
        progress_bar.set_description(
            f'Train E{epoch}: ML:{mask_loss.item():.4f} PL:{pred_loss.item():.3f} Acc:{accuracy:.2f}')
        if tb_writer is not None and counter > -1:
            tb_writer.add_scalar('Loss/Train', loss.item(), counter)
            tb_writer.add_scalar('LossMask/Train', mask_loss.item(), counter)
            tb_writer.add_scalar('LossPred/Train', pred_loss.item(), counter)
            tb_writer.add_scalar('Accuracy/Train', accuracy, counter)
        if counter > -1:
            counter += 1

    train_loss = running_loss / len(train_loader)
    mask_train_loss = mask_running_loss / len(train_loader)
    pred_train_loss = pred_running_loss / len(train_loader)
    progress_bar.set_description(
        f'Train E{epoch}: ML:{mask_train_loss:.4f} PL:{pred_train_loss:.3f} Acc:{running_accuracy / len(train_loader):.2f}')
    if counter >= -1:
        return train_loss, mask_train_loss, pred_train_loss, running_accuracy / len(train_loader), counter
    return train_loss, mask_train_loss, pred_train_loss, running_accuracy / len(train_loader)


def validate(model, val_loader, criterion_mask, criterion_pred, epoch, device):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    running_loss = 0.0
    mask_running_loss = 0.0
    pred_running_loss = 0.0
    progress_bar = tqdm(val_loader, desc=f'Validation Epoch {epoch}')
    with torch.no_grad():
        for i, batch in enumerate(progress_bar):
            inputs, mask_labels, attributes, = batch
            attributes = attributes.to(device)
            # colour_labels = colour_labels.to(device)
            inputs, mask_labels = inputs.to(device), mask_labels.to(device)

            # total = len(val_loader)
            # scale_factor = i / total * 0.5 + 0.5
            # # scale_factor = random.uniform(0.2, 1)
            # inputs, mask_labels = _scale_images_uniformly(inputs, scale_factor), _scale_images_uniformly(mask_labels,
            #                                                                                              scale_factor)

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
            accuracy = positive_accuracy.detach().cpu().item()

            running_loss += loss.item()
            mask_running_loss += mask_loss.item()
            pred_running_loss += pred_loss.item()
            running_accuracy += accuracy
            progress_bar.set_description(
                f'Val E{epoch}:  ML:{mask_running_loss / (i + 1):.4f} PL:{pred_running_loss / (i + 1):.3f} Acc:{running_accuracy / (i + 1):.2f}')

    val_loss = running_loss / len(val_loader)
    mask_val_loss = mask_running_loss / len(val_loader)
    pred_val_loss = pred_running_loss / len(val_loader)
    return val_loss, mask_val_loss, pred_val_loss, running_accuracy / len(val_loader)


def test(model, test_loader, criterion_mask, criterion_pred, epoch, device):
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


def iou(boxes1, boxes2):
    """
    Compute the intersection over union of two batches of boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    Both boxes1 and boxes2 should be tensors of shape [batch_size, num_boxes, 4].
    """
    # Expand dimensions to [batch_size, num_boxes1, 1, 4] and [batch_size, 1, num_boxes2, 4]
    # to make them [batch_size, num_boxes1, num_boxes2, 4] for broadcasting
    boxes1 = boxes1.unsqueeze(2)
    boxes2 = boxes2.unsqueeze(1)

    # Compute the coordinates of the intersection rectangle
    inter_xmin = torch.max(boxes1[..., 0], boxes2[..., 0])
    inter_ymin = torch.max(boxes1[..., 1], boxes2[..., 1])
    inter_xmax = torch.min(boxes1[..., 2], boxes2[..., 2])
    inter_ymax = torch.min(boxes1[..., 3], boxes2[..., 3])

    # Compute the area of intersection rectangle
    inter_area = torch.clamp(inter_xmax - inter_xmin, min=0) * torch.clamp(inter_ymax - inter_ymin, min=0)

    # Compute the area of both sets of boxes
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # Compute the union
    union_area = area1 + area2 - inter_area

    # Compute the IoU
    iou = inter_area / union_area

    return iou


def calculate_map(pred_boxes, pred_scores, true_boxes, true_labels, iou_threshold=0.5):
    """
    Calculate mean average precision for batches of predicted and true boxes.
    Only considers boxes associated with label 1.

    Parameters:
    - pred_boxes: Tensor of shape [batch_size, num_pred_boxes, 4]
    - pred_scores: Tensor of shape [batch_size, num_pred_boxes]
    - true_boxes: Tensor of shape [batch_size, num_true_boxes, 4]
    - true_labels: Tensor of shape [batch_size, num_pred_boxes], should contain binary labels (0 or 1)
    - iou_threshold: float, threshold for IoU to consider a detection as a true positive.
    """
    batch_size = pred_boxes.size(0)
    aps = []

    for batch_idx in range(batch_size):
        # Filter boxes and scores based on true_labels being 1
        relevant_indices = true_labels[batch_idx] == 1.0
        relevant_pred_boxes = pred_boxes[batch_idx][relevant_indices]
        relevant_pred_scores = pred_scores[batch_idx][relevant_indices]
        relevant_true_boxes = true_boxes[batch_idx][relevant_indices]

        # Sort predictions by scores
        scores, sort_indices = torch.sort(relevant_pred_scores, descending=True)
        sorted_pred_boxes = relevant_pred_boxes[sort_indices]

        # Compute IoUs between sorted pred boxes and true boxes
        ious = iou(sorted_pred_boxes.unsqueeze(0), relevant_true_boxes.unsqueeze(0)).squeeze(0)

        num_true_boxes = relevant_true_boxes.size(0)
        num_pred_boxes = sorted_pred_boxes.size(0)

        if num_true_boxes == 0 or num_pred_boxes == 0:
            aps.append(0.0)
            continue

        used = torch.zeros(num_true_boxes, dtype=torch.bool, device=true_boxes.device)
        tp = torch.zeros(num_pred_boxes, dtype=torch.float32, device=true_boxes.device)
        fp = torch.zeros(num_pred_boxes, dtype=torch.float32, device=true_boxes.device)

        for idx in range(num_pred_boxes):
            max_iou, max_index = torch.max(ious[idx], dim=0)
            if max_iou > iou_threshold and not used[max_index]:
                tp[idx] = 1
                used[max_index] = True
            else:
                fp[idx] = 1

        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)

        recalls = tp_cumsum / (num_true_boxes + 1e-6)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        # Calculate AP for the current batch
        ap = torch.trapz(precisions, recalls)
        aps.append(ap.item())

    # Return the mean AP across all batches
    return sum(aps) / len(aps)


def train_DeepFashion2(model, optimizer, train_loader, criterion_mask, criterion_pred, criterion_bbox, scale_range,
                       epoch, device, tb_writer=None, counter=-1):
    model.train()#
    running_loss = 0.0
    mask_running_loss = 0.0
    pred_running_loss = 0.0
    bbox_running_loss = 0.0
    running_mAP = 0.0
    running_f1 = 0.0

    progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
    for i, batch in enumerate(progress_bar):
        inputs, mask_labels, attributes, bboxes_list = batch

        # _input, _mask_labels, _attributes, _bboxes = inputs[0].permute(1, 2, 0).cpu().numpy(), mask_labels[0].cpu().numpy(), attributes[0].cpu().numpy(), bboxes_list[0]
        # from utils import show_deepfashion2_image_masks_and_labels
        # show_deepfashion2_image_masks_and_labels(_input, _mask_labels, _attributes, _bboxes)
        # plt.show()

        attributes = attributes.to(device)
        inputs, mask_labels = inputs.to(device), mask_labels.to(device)

        # process bboxes, batch dimension being (batch_size, num_bbox_classes, 4)
        bboxes = torch.zeros(len(inputs), model.num_bbox_classes, 4).to(device)
        for b, b_list in enumerate(bboxes_list):
            for id, box in b_list:
                bboxes[b, id, :] = torch.tensor(box).to(device)

        # Select a uniform scale for the entire batch
        scale_factor = random.uniform(*scale_range)
        inputs, mask_labels = _scale_images_uniformly(inputs, scale_factor), _scale_images_uniformly(mask_labels,
                                                                                                     scale_factor)

        optimizer.zero_grad()

        pred_masks, pred_classes, pred_bboxes = model(inputs)
        mask_loss = criterion_mask(pred_masks, mask_labels)
        # pred_loss, cl_loss, rg_loss = criterion_pred(pred_classes, classes, pred_colours)  #, colour_labels)
        pred_loss = criterion_pred(pred_classes, attributes)
        bbox_loss = criterion_bbox(pred_bboxes, bboxes)
        # Mask for the labels, expanded to match losses dimensions
        # x = attributes[:, 0:len(bbox_loss[0])].unsqueeze(-1)
        bbox_loss_mask = attributes[:, 0:len(bbox_loss[0])].unsqueeze(-1).expand_as(
            bbox_loss)  # Make the mask broadcastable to match losses
        # Apply mask by multiplying (non-relevant losses will be zeroed out)
        masked_bbox_loss = bbox_loss * bbox_loss_mask.float()
        # Finally, reduce the loss by summing or averaging only non-zero losses
        final_bbox_loss = masked_bbox_loss.sum() / bbox_loss_mask.float().sum() * 4.0
        loss = mask_loss + pred_loss + final_bbox_loss

        loss.backward()
        optimizer.step()

        # a, b = (attributes > 0.5).cpu().int().numpy().tolist(), (pred_classes > 0.5).cpu().int().numpy().tolist()
        f1 = f1_score((attributes > 0.5).cpu().int().numpy().tolist(),
                      (pred_classes > 0.5).cpu().int().numpy().tolist(), average='samples')
        map_score = calculate_map(pred_bboxes, pred_classes[:, 0:len(bbox_loss[0])], bboxes,
                                  attributes[:, 0:len(bbox_loss[0])])

        running_loss += loss.item()
        mask_running_loss += mask_loss.item()
        pred_running_loss += pred_loss.item()
        bbox_running_loss += final_bbox_loss.item()
        running_f1 += f1
        running_mAP += map_score

        progress_bar.set_description(
            f'TE{epoch}: ML:{mask_loss.item():.3f} PL:{pred_loss.item():.3f} BL:{final_bbox_loss.item():.3f} f1:{f1:.2f} mAP:{map_score:.2f}')

        if tb_writer is not None and counter > -1:
            tb_writer.add_scalar('Loss/Train', loss.item(), counter)
            tb_writer.add_scalar('LossMask/Train', mask_loss.item(), counter)
            tb_writer.add_scalar('LossPred/Train', pred_loss.item(), counter)
            tb_writer.add_scalar('LossBBox/Train', final_bbox_loss.item(), counter)
            tb_writer.add_scalar('MAP/Train', map_score, counter)
            tb_writer.add_scalar('F1/Train', f1, counter)
        if counter > -1:
            counter += 1

    train_loss = running_loss / len(train_loader)
    mask_train_loss = mask_running_loss / len(train_loader)
    pred_train_loss = pred_running_loss / len(train_loader)
    bbox_train_loss = bbox_running_loss / len(train_loader)
    avrg_mAP = running_mAP / len(train_loader)
    avrg_f1 = running_f1 / len(train_loader)
    progress_bar.set_description(
        f'TE{epoch}: ML:{mask_running_loss:.3f} PL:{pred_running_loss:.3f} BL:{bbox_train_loss:.3f} mAP:{avrg_mAP:.2f} f1:{avrg_f1:.2f}')
    if counter >= -1:
        return train_loss, mask_train_loss, pred_train_loss, avrg_mAP, avrg_f1, counter
    return train_loss, mask_train_loss, pred_train_loss, avrg_mAP, avrg_f1


def val_DeepFashion2(model, val_loader, criterion_mask, criterion_pred, criterion_bbox, epoch, device):
    model.train()
    running_loss = 0.0
    mask_running_loss = 0.0
    pred_running_loss = 0.0
    running_mAP = 0.0
    running_f1 = 0.0
    bbox_running_loss = 0.0
    f1_list_a = []
    f1_list_b = []
    f1 = 0.0

    progress_bar = tqdm(val_loader, desc=f'Training Epoch {epoch}')
    with torch.no_grad():
        for i, batch in enumerate(progress_bar):
            inputs, mask_labels, attributes, bboxes_list = batch

            # _input, _mask_labels, _attributes, _bboxes = inputs[0].permute(1, 2, 0).cpu().numpy(), mask_labels[
            #     0].cpu().numpy(), attributes[0].cpu().numpy(), bboxes[0]
            # from datasets import show_deepfashion2_image_masks_and_labels
            # show_deepfashion2_image_masks_and_labels(_input, _mask_labels, _attributes, _bboxes)

            attributes = attributes.to(device)
            inputs, mask_labels = inputs.to(device), mask_labels.to(device)

            # process bboxes, batch dimension being (batch_size, num_bbox_classes, 4)
            bboxes = torch.zeros(len(inputs), model.num_bbox_classes, 4).to(device)
            for b, b_list in enumerate(bboxes_list):
                for id, box in b_list:
                    bboxes[b, id, :] = torch.tensor(box).to(device)

            # Select a uniform scale for the entire batch
            # scale_factor = random.uniform(*scale_range)
            # inputs, mask_labels = _scale_images_uniformly(inputs, scale_factor), _scale_images_uniformly(mask_labels,
            #                                                                                              scale_factor)

            pred_masks, pred_classes, pred_bboxes = model(inputs)
            mask_loss = criterion_mask(pred_masks, mask_labels)
            # pred_loss, cl_loss, rg_loss = criterion_pred(pred_classes, classes, pred_colours)  #, colour_labels)
            pred_loss = criterion_pred(pred_classes, attributes)
            bbox_loss = criterion_bbox(pred_bboxes, bboxes)
            # Mask for the labels, expanded to match losses dimensions
            bbox_loss_mask = attributes[:, 0:len(bbox_loss[0])].unsqueeze(-1).expand_as(
                bbox_loss)  # Make the mask broadcastable to match losses
            # Apply mask by multiplying (non-relevant losses will be zeroed out)
            masked_bbox_loss = bbox_loss * bbox_loss_mask.float()
            # Finally, reduce the loss by summing or averaging only non-zero losses
            final_bbox_loss = masked_bbox_loss.sum() / bbox_loss_mask.float().sum()
            loss = mask_loss + pred_loss + final_bbox_loss

            a = (attributes > 0.5).cpu().int().numpy()[0]
            b = (pred_classes > 0.5).cpu().int().numpy()[0]
            f1_list_a.append(a)
            f1_list_b.append(b)
            if i % 1000 == 0 or i >= len(val_loader) - 1:
                f1 = f1_score(f1_list_a, f1_list_b, average='samples')
            map_score = calculate_map(pred_bboxes, pred_classes[:, 0:len(bbox_loss[0])], bboxes,
                                      attributes[:, 0:len(bbox_loss[0])])

            running_loss += loss.item()
            mask_running_loss += mask_loss.item()
            pred_running_loss += pred_loss.item()
            bbox_running_loss += final_bbox_loss.item()
            running_f1 = f1
            running_mAP += map_score

            progress_bar.set_description(
                f'VE{epoch}: ML:{mask_running_loss / (i + 1):.3f} PL:{pred_running_loss / (i + 1):.3f} BL:{bbox_running_loss / (i + 1):.3f} f1:{running_f1:.2f} mAP:{running_mAP / (i + 1):.2f}')

    val_loss = running_loss / len(val_loader)
    mask_val_loss = mask_running_loss / len(val_loader)
    pred_val_loss = pred_running_loss / len(val_loader)
    bbox_train_loss = bbox_running_loss / len(val_loader)
    avrg_mAP = running_mAP / len(val_loader)
    avrg_f1 = running_f1
    progress_bar.set_description(
        f'VE{epoch}: ML:{mask_val_loss:.3f} PL:{pred_val_loss:.3f} BL:{bbox_train_loss:.3f} f1:{avrg_f1:.2f} mAP:{avrg_mAP:.2f}')
    return val_loss, mask_val_loss, pred_val_loss, avrg_mAP, avrg_f1


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
