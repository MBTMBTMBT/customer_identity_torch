import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, middle_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(middle_channels + out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, concat_with):
        x = self.up(x)
        x = torch.cat([x, concat_with], dim=1)
        return self.conv(x)


def create_unet_decoder(num_classes):
    # Adjust sizes based on FPN output levels if needed
    return nn.ModuleList([
        UNetDecoderBlock(256, 256, 256),
        UNetDecoderBlock(256, 256, 128),
        UNetDecoderBlock(128, 128, 64),
        nn.Conv2d(128, num_classes, kernel_size=1)  # Ensure the final convolution aligns with the concatenated output
    ])


class MultiTaskResNet50(nn.Module):
    def __init__(self, num_classes_seg, num_classes_clf, num_boxes, pretrained=True):
        super().__init__()
        self.backbone = resnet_fpn_backbone('resnet50', pretrained=pretrained)

        self.decoder = create_unet_decoder(num_classes_seg)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes_clf)

        # Define the RPN and ROI heads separately to integrate them properly
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))
        self.rpn = FasterRCNN(self.backbone, num_classes=num_boxes, rpn_anchor_generator=anchor_generator)

    def forward(self, x):
        features = self.backbone(x)  # FPN features as a dict

        # Start from the deepest layer and move upwards
        x2 = features['2']  # Example FPN layer usage
        x1 = features['1']
        x0 = features['0']

        x = self.decoder[0](x2, x1)
        x = self.decoder[1](x, x0)
        segmentation = self.decoder[2](x)

        pooled_features = self.global_pool(features['0'])
        classification = self.fc(torch.flatten(pooled_features, 1))

        detections = self.rpn(x)

        return segmentation, classification, detections


model = MultiTaskResNet50(num_classes_seg=21, num_classes_clf=10, num_boxes=5)
input_tensor = torch.rand(1, 3, 800, 800)
segmentation, classification, detections = model(input_tensor)
