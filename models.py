import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import sigmoid
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import sigmoid_focal_loss, nms, box_iou
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from sklearn.metrics import f1_score, jaccard_score, average_precision_score


class Decoder(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        inner_channels = out_channels // 2
        self.up_conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, inner_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x_copy, x):
        x = self.up(x)
        if x.size(2) != x_copy.size(2) or x.size(3) != x_copy.size(3):
            x = F.interpolate(x, size=(x_copy.size(2), x_copy.size(3)), mode='bilinear', align_corners=True)
        x = torch.cat((x_copy, x), dim=1)
        x = self.up_conv(x)
        return x


class UNetWithResnetEncoder(nn.Module):
    def __init__(self, num_classes, in_channels=3, freeze_bn=False, sigmoid=True):
        super(UNetWithResnetEncoder, self).__init__()
        self.sigmoid = sigmoid
        self.resnet = models.resnet34(pretrained=True)  # Initialize with a ResNet model

        if in_channels != 3:
            self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.encoder1 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu)
        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4

        self.up1 = Decoder(512, 256, 256)
        self.up2 = Decoder(256, 128, 128)
        self.up3 = Decoder(128, 64, 64)
        self.up4 = Decoder(64, 64, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self._initialize_weights()

        if freeze_bn:
            self.freeze_bn()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)

        x = self.up1(x4, x5)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        x = F.interpolate(x, size=(x.size(2) * 2, x.size(3) * 2), mode='bilinear', align_corners=True)

        x = self.final_conv(x)

        if self.sigmoid:
            x = torch.sigmoid(x)
        return x

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def unfreeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train()


class MultiLabelResNet(nn.Module):
    def __init__(self, num_labels, input_channels=3, sigmoid=True, pretrained=True, ):
        super(MultiLabelResNet, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        self.sigmoid = sigmoid

        if input_channels != 3:
            self.model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        num_ftrs = self.model.fc.in_features

        self.model.fc = nn.Linear(num_ftrs, num_labels)

    def forward(self, x):
        x = self.model(x)
        if self.sigmoid:
            x = torch.sigmoid(x)
        return x


class CombinedModel(nn.Module):
    def __init__(self, segment_model: nn.Module, predict_model: nn.Module, cat_layers: int = None):
        super(CombinedModel, self).__init__()
        self.segment_model = segment_model
        self.predict_model = predict_model
        self.cat_layers = cat_layers
        self.freeze_seg = False

    def forward(self, x: torch.Tensor):
        seg_masks = self.segment_model(x)
        seg_masks_ = seg_masks.detach()
        if self.cat_layers:
            seg_masks_ = seg_masks_[:, 0:self.cat_layers]
            x = torch.cat((x, seg_masks_), dim=1)
        else:
            x = torch.cat((x, seg_masks_), dim=1)
        logic_outputs = self.predict_model(x)
        return seg_masks, logic_outputs

    def freeze_segment_model(self):
        self.segment_model.eval()

    def unfreeze_segment_model(self):
        self.segment_model.train()


class SegmentPredictor(nn.Module):
    def __init__(self, num_masks, num_labels, in_channels=3, sigmoid=True):
        super(SegmentPredictor, self).__init__()
        self.sigmoid = sigmoid
        self.resnet = models.resnet18(pretrained=True)

        # Adapt ResNet to handle different input channel sizes
        if in_channels != 3:
            self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Encoder layers
        self.encoder1 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu)
        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4

        # Decoder layers
        # resnet18/34
        self.up1 = Decoder(512, 256, 256)
        self.up2 = Decoder(256, 128, 128)
        self.up3 = Decoder(128, 64, 64)
        self.up4 = Decoder(64, 64, 64)

        # resnet50/101/152
        # self.up1 = Decoder(2048, 1024, 1024)
        # self.up2 = Decoder(1024, 512, 512)
        # self.up3 = Decoder(512, 256, 256)
        # self.up4 = Decoder(256, 64, 64)

        # Segmentation head
        self.final_conv = nn.Conv2d(64, num_masks, kernel_size=1)

        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),  # resnet18/34
            # nn.Linear(2048, 256),  # resnet50/101/152
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.5),
            nn.Linear(256, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_labels)
        )

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)

        x = self.up1(x4, x5)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        x = F.interpolate(x, size=(x.size(2) * 2, x.size(3) * 2), mode='bilinear', align_corners=True)

        mask = self.final_conv(x)

        # Predicting the labels using features from the last encoder output
        x_cls = self.global_pool(x5)  # Use the feature map from the last encoder layer
        x_cls = x_cls.view(x_cls.size(0), -1)
        labels = self.classifier(x_cls)

        if self.sigmoid:
            mask = torch.sigmoid(mask)
            labels = torch.sigmoid(labels)

        return mask, labels


class SegmentPredictorBbox(SegmentPredictor):
    def __init__(self, num_masks, num_labels, num_bbox_classes, in_channels=3, sigmoid=True):
        super(SegmentPredictorBbox, self).__init__(num_masks, num_labels, in_channels, sigmoid)
        self.num_bbox_classes = num_bbox_classes
        self.bbox_generator = nn.Sequential(
            nn.Linear(512, 256),  # resnet18/34
            # nn.Linear(2048, 256),  # resnet50/101/152
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, num_bbox_classes * 4)
        )

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)

        x = self.up1(x4, x5)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        x = F.interpolate(x, size=(x.size(2) * 2, x.size(3) * 2), mode='bilinear', align_corners=True)

        mask = self.final_conv(x)

        # Predicting the labels using features from the last encoder output
        x_cls = self.global_pool(x5)  # Use the feature map from the last encoder layer
        x_cls = x_cls.view(x_cls.size(0), -1)
        labels = self.classifier(x_cls)
        bboxes = self.bbox_generator(x_cls).view(-1, self.num_bbox_classes, 4)

        # no sigmoid for bboxes.
        if self.sigmoid:
            mask = torch.sigmoid(mask)
            labels = torch.sigmoid(labels)

        return mask, labels, bboxes


def calc_detection_loss(class_logits, box_regression, labels, boxes):
    classification_loss = sigmoid_focal_loss(class_logits, labels, reduction="mean")
    reg_loss = F.smooth_l1_loss(box_regression, boxes, reduction="mean")
    return classification_loss + reg_loss


def calculate_mAP(detections, ground_truth_boxes, ground_truth_labels):
    # Simplified mAP calculation without considering different IoU thresholds
    y_true = []
    y_scores = []
    for class_id in np.unique(ground_truth_labels):
        gt_indices = (ground_truth_labels == class_id)
        gt_boxes_class = ground_truth_boxes[gt_indices]
        # Dummy predictions: scoring system for the example
        pred_indices = [d[2] for d in detections].index(class_id) if class_id in [d[2] for d in detections] else []
        pred_scores_class = [d[1] for d in detections if d[2] == class_id]
        pred_boxes_class = [d[0] for d in detections if d[2] == class_id]

        # Assume every ground truth has one prediction
        matches = [np.max(box_iou(torch.stack(pred_boxes_class), torch.stack([gt_box]))) > 0.5 for gt_box in
                   gt_boxes_class]
        y_true.extend([1] * len(gt_boxes_class))  # 1 for all ground truths
        y_scores.extend([max(pred_scores_class) if match else 0 for match in matches])

    return average_precision_score(y_true, y_scores)


# class IntegratedModel(nn.Module):
#     def __init__(self, num_classes, num_labels, num_detection_classes, device=torch.device('cpu')):
#         super(IntegratedModel, self).__init__()
#         # Setup the backbone with FPN and last level enhancements
#         self.backbone = resnet_fpn_backbone('resnet50', pretrained=True, extra_blocks=LastLevelP6P7(256, 256))
#
#         # Define anchor generator with specific sizes and aspect ratios
#         anchor_generator = AnchorGenerator(
#             sizes=((32,), (64,), (128,), (256,), (512,), (512,)),  # Tuple of tuples
#             aspect_ratios=((0.5, 1.0, 2.0),) * 6  # Same aspect ratios for each feature map scale
#         )
#
#         # RetinaNet setup using the backbone directly
#         self.retinanet = RetinaNet(self.backbone, num_classes=num_detection_classes, anchor_generator=anchor_generator)
#
#         # U-Net like decoder for segmentation
#         self.up0 = Decoder(2048, 2048, 2048)
#         self.up1 = Decoder(2048, 1024, 512)
#         self.up2 = Decoder(512, 512, 256)
#         self.up3 = Decoder(256, 256, 128)
#         self.up4 = Decoder(128, 64, 64)
#         self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
#
#         # Classification head using features before the FPN
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(2048, num_labels)  # Adjust size according to the last layer of the backbone
#
#         self.segmentation_loss_fn = torch.nn.CrossEntropyLoss()
#         self.classification_loss_fn = torch.nn.BCEWithLogitsLoss()
#
#         self.device = device
#         self.to(self.device)
#
#     def forward(self, x, targets=None):
#         # Backbone through FPN which gives a dict of feature maps
#         features = self.backbone(x)
#
#         # Check if training or evaluation mode
#         if self.training:
#             assert targets is not None, "Targets should not be None when in training mode"
#             detection_output = self.retinanet(x, targets)  # Pass targets during training
#         else:
#             detection_output = self.retinanet(x)  # In eval mode, no targets needed
#
#         # Extract feature maps for segmentation (assumes FPN outputs are ordered or keyed consistently)
#         x0, x1, x2, x3, x4, x5 = [features[k] for k in sorted(features.keys())]
#
#         # U-Net segmentation
#         x = self.up0(x0, self.up1(x4, x5))
#         x = self.up2(x3, x)
#         x = self.up3(x2, x)
#         x = self.up4(x1, x)
#         segmentation_output = self.final_conv(x)
#
#         # Classification
#         x_avg = self.avgpool(x5)  # Use the deepest feature map for classification
#         x_avg = torch.flatten(x_avg, 1)
#         label_output = self.fc(x_avg)
#
#         return segmentation_output, label_output, detection_output
#
#     def prepare_targets(self, bbox_labels):
#         targets = []
#         for image_annotations in bbox_labels:  # Assuming bbox_labels is a list of lists (one per image)
#             bboxes = [torch.tensor(bbox).float().to(self.device) for _, bbox in image_annotations]
#             labels = [torch.tensor(label).long().to(self.device) for label, _ in image_annotations]
#
#             # Handle the case where there are no annotations for an image
#             if not bboxes:
#                 bboxes = torch.empty((0, 4), dtype=torch.float, device=self.device)
#             else:
#                 bboxes = torch.stack(bboxes).to(self.device)
#
#             if not labels:
#                 labels = torch.empty((0,), dtype=torch.long, device=self.device)
#             else:
#                 labels = torch.tensor(labels).to(self.device)  # Convert list of tensors to a single tensor
#
#             targets.append({'boxes': bboxes, 'labels': labels})
#
#         return targets
#
#     def train_batch(self, inputs, masks, attributes, bbox_labels, optimizer):
#         self.train()
#         inputs, masks, attributes = inputs.to(self.device), masks.to(self.device), attributes.to(self.device)
#
#         # Prepare targets for detection
#         targets = self.prepare_targets(bbox_labels)
#
#         optimizer.zero_grad()
#
#         # Forward pass
#         segmentation_output, label_output, loss_dict = self(inputs, targets=targets)
#
#         # Compute losses for segmentation and classification
#         seg_losses = [self.segmentation_loss_fn(seg_out, mask) for seg_out, mask in zip(segmentation_output, masks)]
#         cls_losses = [self.classification_loss_fn(label_out, attr) for label_out, attr in zip(label_output, attributes)]
#         det_loss = sum(loss for loss in loss_dict.values())
#
#         # Sum and average the losses
#         total_loss = sum(seg_losses) / len(seg_losses) + sum(cls_losses) / len(cls_losses) + det_loss
#         total_loss.backward()
#         optimizer.step()
#
#         # Calculate metrics if required
#         self.eval()
#         with torch.no_grad():
#             mAP, f1, iou = self.calculate_metrics(inputs, masks, bbox_labels)
#         self.train()
#
#         return total_loss.item(), sum(seg_losses) / len(seg_losses).item(), sum(cls_losses) / len(
#             cls_losses).item(), det_loss.item(), mAP, f1, iou
#
#     def calculate_metrics(self, inputs, masks, bbox_labels):
#         # Since we need predictions for metrics, we temporarily switch to evaluation mode
#         segmentation_output, label_output, detection_output = self(inputs)  # No targets passed
#
#         # Calculate mAP, F1, and IoU for each image
#         mAP_scores, f1_scores, iou_scores = [], [], []
#         for det_out, label_out, seg_out, mask in zip(detection_output, label_output, segmentation_output, masks):
#             # Extract detection outputs
#             pred_boxes = det_out['boxes']
#             pred_scores = torch.sigmoid(det_out['scores'])
#             pred_labels = det_out['labels']
#
#             # Apply NMS
#             keep = nms(pred_boxes, pred_scores, 0.5)
#             pred_boxes = pred_boxes[keep]
#             pred_scores = pred_scores[keep]
#             pred_labels = pred_labels[keep]
#
#             # Calculate mAP, F1, IoU using your preferred methods or library functions
#             mAP_scores.append(calculate_mAP(pred_boxes, pred_labels, pred_scores, bbox_labels))
#             f1_scores.append(f1_score(masks.cpu().numpy(), seg_out.argmax(dim=1).cpu().numpy(), average='macro'))
#             iou_scores.append(jaccard_score(mask.cpu().numpy(), seg_out.argmax(dim=1).cpu().numpy(), average='macro'))
#
#         # Average the metrics across the batch
#         avg_mAP = sum(mAP_scores) / len(mAP_scores)
#         avg_f1 = sum(f1_scores) / len(f1_scores)
#         avg_iou = sum(iou_scores) / len(iou_scores)
#
#         return avg_mAP, avg_f1, avg_iou
#
#     def eval_batch(self, inputs, masks, attributes, bbox_labels):
#         self.eval()
#         with torch.no_grad():
#             inputs, masks, attributes = inputs.to(self.device), masks.to(self.device), attributes.to(self.device)
#
#             # Prepare targets for detection
#             targets = []
#             for bbox_label in bbox_labels:
#                 bboxes, labels = zip(
#                     *[(torch.tensor(bbox).float().to(self.device), torch.tensor(label).long().to(self.device)) for
#                       label, bbox in bbox_label])
#                 targets.append({'boxes': torch.stack(bboxes), 'labels': torch.stack(labels)})
#
#             segmentation_output, label_output, detection_output = self(inputs, targets=targets)
#
#             seg_loss = self.segmentation_loss_fn(segmentation_output, masks)
#             cls_loss = self.classification_loss_fn(label_output, attributes)
#             det_loss = sum(loss for loss in detection_output.values())
#
#             # Use the shared function to calculate metrics
#             avg_mAP, avg_f1, avg_iou = self.calculate_metrics(inputs, masks, bbox_labels)
#
#             total_loss = seg_loss + cls_loss + det_loss
#
#             return total_loss.item(), seg_loss.item(), cls_loss.item(), det_loss.item(), avg_mAP, avg_f1, avg_iou
#
#     def predict_frame(self, frame):
#         self.eval()
#         with torch.no_grad():
#             frame = frame.to(self.device)
#             segmentation_output, label_output, detection_output = self(frame)
#             # Convert logits to probabilities
#             scores = sigmoid(detection_output['scores'])
#             # Bounding boxes decoding (assuming `detection_output['boxes']` are already decoded)
#             boxes = detection_output['boxes']
#             labels = detection_output['labels']
#
#             # Apply confidence thresholding
#             high_conf_indices = torch.where(scores > 0.5)[0]  # Confidence threshold of 0.5
#             scores = scores[high_conf_indices]
#             boxes = boxes[high_conf_indices]
#             labels = labels[high_conf_indices]
#
#             # Non-Maximum Suppression (NMS)
#             keep_indices = nms(boxes, scores, iou_threshold=0.5)  # IoU threshold for NMS
#             final_boxes = boxes[keep_indices]
#             final_scores = scores[keep_indices]
#             final_labels = labels[keep_indices]
#
#             return final_boxes, final_scores, final_labels, segmentation_output, label_output
#

