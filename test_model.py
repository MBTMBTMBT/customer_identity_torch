import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.ops import MultiScaleRoIAlign, roi_align, nms
from torchvision.ops import boxes as box_ops


# Create the backbone
def create_resnet_backbone(pretrained=True):
    model = resnet50(pretrained=pretrained)
    modules = list(model.children())[:-2]
    backbone = nn.Sequential(*modules)
    backbone.out_channels = 2048
    return backbone


# RPN as previously defined
class RPN(nn.Module):
    def __init__(self, in_channels):
        super(RPN, self).__init__()
        self.conv = nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(512, 3 * 1, kernel_size=1)  # Assuming 3 anchor scales, 1 anchor per location
        self.bbox_pred = nn.Conv2d(512, 3 * 4, kernel_size=1)  # 4 coordinates per box

    def forward(self, x):
        x = self.conv(x)
        logits = self.cls_logits(x)
        bbox_regs = self.bbox_pred(x)
        return logits, bbox_regs


# Fast R-CNN head
class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.fc6 = nn.Linear(in_channels * 7 * 7, 1024)
        self.fc7 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = nn.functional.relu(self.fc6(x))
        x = nn.functional.relu(self.fc7(x))
        cls_score = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_score, bbox_pred


# Faster R-CNN model
class FasterRCNN(nn.Module):
    def __init__(self, backbone, rpn, roi_pooler, box_predictor):
        super(FasterRCNN, self).__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.roi_pooler = roi_pooler
        self.box_predictor = box_predictor

    def forward(self, images):
        features = self.backbone(images)
        rpn_logits, rpn_bbox_regs = self.rpn(features)

        # Convert logits and bbox_regs to proposals
        proposals = self.generate_proposals(features, rpn_logits, rpn_bbox_regs, images.shape[-2:])

        # Apply RoI pooling
        if proposals is not None and proposals.shape[1] > 0:
            pooled_features = self.roi_pooler(features, proposals)
            # Classify proposals
            class_scores, bbox_preds = self.box_predictor(pooled_features)
            return class_scores, bbox_preds, proposals
        else:
            return torch.empty(0), torch.empty(0), torch.empty(0)

    def generate_proposals(self, features, logits, bbox_regs, image_size, nms_thresh=0.7, pre_nms_top_n=1000,
                           post_nms_top_n=1000):
        device = logits.device
        # Assuming logits and bbox_regs are (N, A * 4, H, W) and (N, A * 1, H, W) respectively
        # A is the number of anchors per location
        num_anchors = logits.size(1) // 4
        H, W = logits.shape[-2:]

        # Generate anchor boxes (this should be adapted to your anchor strategy)
        anchors = generate_anchors(num_anchors, image_size).to(device)

        # Convert logits to probabilities (objectness score)
        scores = torch.sigmoid(logits)

        # Convert bbox_regs to boxes
        proposals = apply_deltas_to_anchors(bbox_regs.view(-1, 4), anchors)

        # Clip proposals to image
        proposals = box_ops.clip_boxes_to_image(proposals, image_size)

        # Select the top pre_nms_top_n scores and proposals
        scores, top_idxs = scores.view(-1).topk(pre_nms_top_n)
        proposals = proposals[top_idxs]

        # Apply NMS
        keep = nms(proposals, scores, nms_thresh)
        keep = keep[:post_nms_top_n]  # Keep only top post_nms_top_n results
        proposals = proposals[keep]

        return proposals


# Helper functions for anchors and applying deltas
def generate_anchors(num_anchors_per_location, feature_map_size, anchor_sizes, aspect_ratios):
    # Generate anchor boxes for a single point with all combinations of sizes and aspect ratios
    anchors = []
    for size in anchor_sizes:
        for aspect_ratio in aspect_ratios:
            width = size * aspect_ratio ** 0.5
            height = size / aspect_ratio ** 0.5
            anchors.append([-width / 2, -height / 2, width / 2, height / 2])
    anchors = torch.tensor(anchors, dtype=torch.float32)

    # Replicate anchors to all spatial positions
    grid_height, grid_width = feature_map_size
    shifts_x = torch.arange(0, grid_width) * anchor_stride
    shifts_y = torch.arange(0, grid_height) * anchor_stride
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

    # Add shifts to anchor coordinates
    all_anchors = (anchors.view(-1, 4) + shifts.view(-1, 1, 4)).reshape(-1, 4)
    return all_anchors


def apply_deltas_to_anchors(deltas, anchors):
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights

    dx, dy, dw, dh = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]
    pred_ctr_x = ctr_x + dx * widths
    pred_ctr_y = ctr_y + dy * heights
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    pred_boxes = torch.zeros_like(deltas)
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h
    return pred_boxes


# Instantiate and run the model
backbone = create_resnet_backbone()
rpn = RPN(backbone.out_channels)
roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
box_predictor = FastRCNNPredictor(2048, num_classes=91)  # Number of classes, e.g., COCO dataset

model = FasterRCNN(backbone, rpn, roi_pooler, box_predictor)

# Dummy image input
dummy_image = torch.rand(1, 3, 800, 800)  # (batch_size, channels, H, W)

# Forward pass
class_scores, bbox_preds, proposals = model(dummy_image)

print("Class scores:", class_scores.size())
print("Bounding box predictions:", bbox_preds.size())
print("Proposals:", proposals.size())
