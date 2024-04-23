import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import sigmoid
from torch.optim import Adam
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.ops import sigmoid_focal_loss, nms, box_iou
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from sklearn.metrics import f1_score, jaccard_score, average_precision_score


def X2conv(in_channels, out_channels, inner_channels=None):
    inner_channels = out_channels // 2 if inner_channels is None else inner_channels
    down_conv = nn.Sequential(
        nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(inner_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(inner_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))
    return down_conv


class Decoder(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.up_conv = X2conv(out_channels + skip_channels, out_channels)

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


class IntegratedModel(nn.Module):
    def __init__(self, num_classes, num_labels, num_detection_classes, device=torch.device('cpu')):
        super(IntegratedModel, self).__init__()
        self.backbone = resnet_fpn_backbone('resnet50', pretrained=True)  # Using ResNet50 as the shared backbone

        # Modify the first conv layer if using different input size or grayscale images
        # self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.fpn = LastLevelP6P7(2048, 256)  # FPN with extra layers P6 and P7

        # RetinaNet Head setup
        self.retinanet = RetinaNet(self.backbone, num_classes=num_detection_classes,)
        self.retinanet.head = RetinaNetHead(in_channels=256,
                                            num_anchors=self.retinanet.head.classification_head.num_anchors,
                                            num_classes=num_detection_classes)

        # U-Net like decoder for segmentation
        self.up1 = Decoder(2048, 1024, 512)
        self.up2 = Decoder(512, 512, 256)
        self.up3 = Decoder(256, 256, 128)
        self.up4 = Decoder(128, 64, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        # Classification head using features before the FPN
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_labels)  # Adjust size according to the last layer of the backbone

        self.segmentation_loss_fn = torch.nn.CrossEntropyLoss()
        self.classification_loss_fn = torch.nn.BCEWithLogitsLoss()

        self.device = device
        self.to(self.device)

    def forward(self, x):
        # Backbone
        x1 = self.backbone.conv1(x)
        x1 = self.backbone.bn1(x1)
        x1 = self.backbone.relu(x1)
        x1 = self.backbone.maxpool(x1)

        x2 = self.backbone.layer1(x1)
        x3 = self.backbone.layer2(x2)
        x4 = self.backbone.layer3(x3)
        x5 = self.backbone.layer4(x4)

        # RetinaNet detection
        features = self.fpn([x5])  # FPN needs a list of tensors from ResNet layer4
        detection_output = self.retinanet.head(features)

        # U-Net segmentation
        x = self.up1(x4, x5)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        segmentation_output = self.final_conv(x)

        # Classification
        x_avg = self.avgpool(x5)
        x_avg = torch.flatten(x_avg, 1)
        label_output = self.fc(x_avg)

        return segmentation_output, label_output, detection_output

    def train_batch(self, inputs, masks, attributes, bbox_labels, optimizer):
        self.train()
        inputs, masks, attributes = inputs.to(self.device), masks.to(self.device), attributes.to(self.device)

        # Unpack bboxes and labels
        all_bboxes = []
        all_labels = []
        for bbox_label in bbox_labels:
            labels, bboxes = zip(*[(torch.tensor(label).to(self.device), torch.tensor(bbox).to(self.device)) for label, bbox in bbox_label])
            all_bboxes.extend(bboxes)
            all_labels.extend(labels)

        # Convert lists to tensors; ensure your dataset structure allows this concatenation
        bboxes_tensor = torch.cat(all_bboxes, dim=0)
        labels_tensor = torch.cat(all_labels)

        optimizer.zero_grad()
        segmentation_output, label_output, detection_output = self(inputs)

        seg_loss = self.segmentation_loss_fn(segmentation_output, masks)
        cls_loss = self.classification_loss_fn(label_output, attributes)
        det_loss = calc_detection_loss(detection_output['class_logits'], detection_output['box_regression'],
                                       labels_tensor, bboxes_tensor)

        total_loss = seg_loss + cls_loss + det_loss
        total_loss.backward()
        optimizer.step()

        # Post-process outputs for metric calculation; this part is computationally expensive
        scores = torch.sigmoid(detection_output['scores'])
        boxes = detection_output['boxes']
        labels = detection_output['labels']

        final_detections = []
        for class_id in torch.unique(labels_tensor):
            indices = (labels == class_id)
            class_scores = scores[indices]
            class_boxes = boxes[indices]
            high_conf_indices = class_scores > 0.5  # Confidence threshold
            class_scores = class_scores[high_conf_indices]
            class_boxes = class_boxes[high_conf_indices]

            keep_indices = nms(class_boxes, class_scores, iou_threshold=0.5)
            final_detections.extend((class_boxes[i], class_scores[i], class_id) for i in keep_indices)

        # Simulated mAP calculation
        mAP = calculate_mAP(final_detections, bboxes_tensor, labels_tensor)
        f1 = f1_score(labels_tensor.cpu().numpy(), label_output.argmax(dim=1).cpu().numpy(), average='macro')
        iou = jaccard_score(masks.cpu().numpy(), segmentation_output.argmax(dim=1).cpu().numpy(), average='macro')

        return total_loss.item(), seg_loss.item(), cls_loss.item(), det_loss.item(), mAP, f1, iou

    def eval_batch(self, inputs, masks, attributes, bbox_labels):
        self.eval()
        with torch.no_grad():
            inputs, masks, attributes = inputs.to(self.device), masks.to(self.device), attributes.to(self.device)
            # Unpacking bounding boxes and labels assuming bbox_labels is structured correctly
            labels, bboxes = zip(*[(torch.tensor(label).to(self.device), torch.tensor(bbox).to(self.device)) for label, bbox in bbox_label])
            bboxes = torch.stack(bboxes)
            labels = torch.cat(labels)

            segmentation_output, label_output, detection_output = self(inputs)

            seg_loss = self.segmentation_loss_fn(segmentation_output, masks)
            cls_loss = self.classification_loss_fn(label_output, attributes)
            det_loss = self.calc_detection_loss(detection_output['class_logits'], detection_output['box_regression'],
                                                labels, bboxes)
            total_loss = seg_loss + cls_loss + det_loss

            # Calculate performance metrics assuming correct inputs
            scores = torch.sigmoid(detection_output['scores'])
            boxes = detection_output['boxes']

            # Filter and apply NMS
            final_detections = []
            for class_id in torch.unique(labels):
                indices = (labels == class_id)
                class_scores = scores[indices]
                class_boxes = boxes[indices]
                high_conf_indices = class_scores > 0.5  # Confidence threshold
                class_scores = class_scores[high_conf_indices]
                class_boxes = class_boxes[high_conf_indices]

                keep_indices = nms(class_boxes, class_scores, iou_threshold=0.5)
                final_detections.extend((class_boxes[i], class_scores[i], class_id) for i in keep_indices)

            mAP = calculate_mAP(final_detections, bboxes, labels)  # Assuming calculate_mAP is defined correctly
            f1 = f1_score(labels.cpu().numpy(), label_output.argmax(dim=1).cpu().numpy(), average='macro')
            iou = jaccard_score(masks.cpu().numpy(), segmentation_output.argmax(dim=1).cpu().numpy(), average='macro')

        return total_loss.item(), seg_loss.item(), cls_loss.item(), det_loss.item(), mAP, f1, iou

    def predict_frame(self, frame):
        self.eval()
        with torch.no_grad():
            frame = frame.to(self.device)
            segmentation_output, label_output, detection_output = self(frame)
            # Convert logits to probabilities
            scores = sigmoid(detection_output['scores'])
            # Bounding boxes decoding (assuming `detection_output['boxes']` are already decoded)
            boxes = detection_output['boxes']
            labels = detection_output['labels']

            # Apply confidence thresholding
            high_conf_indices = torch.where(scores > 0.5)[0]  # Confidence threshold of 0.5
            scores = scores[high_conf_indices]
            boxes = boxes[high_conf_indices]
            labels = labels[high_conf_indices]

            # Non-Maximum Suppression (NMS)
            keep_indices = nms(boxes, scores, iou_threshold=0.5)  # IoU threshold for NMS
            final_boxes = boxes[keep_indices]
            final_scores = scores[keep_indices]
            final_labels = labels[keep_indices]

            return final_boxes, final_scores, final_labels, segmentation_output, label_output


