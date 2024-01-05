# https://github.com/yassouali/pytorch-segmentation/tree/master/models

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
# from itertools import chain
# import torch.utils.model_zoo as model_zoo


# # https://github.com/yassouali/pytorch-segmentation/blob/master/utils/helpers.py
# def initialize_weights(*models):
#     for model in models:
#         for m in model.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1.)
#                 m.bias.data.fill_(1e-4)
#             elif isinstance(m, nn.Linear):
#                 m.weight.data.normal_(0.0, 0.0001)
#                 m.bias.data.zero_()


def set_trainable_attr(m,b):
    m.trainable = b
    for p in m.parameters(): p.requires_grad = b


def x2conv(in_channels, out_channels, inner_channels=None):
    inner_channels = out_channels // 2 if inner_channels is None else inner_channels
    down_conv = nn.Sequential(
        nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(inner_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(inner_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))
    return down_conv


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.down_conv = x2conv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        x = self.down_conv(x)
        x = self.pool(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.up_conv = x2conv(in_channels, out_channels)

    def forward(self, x_copy, x, interpolate=True):
        x = self.up(x)

        if (x.size(2) != x_copy.size(2)) or (x.size(3) != x_copy.size(3)):
            if interpolate:
                # Iterpolating instead of padding
                x = F.interpolate(x, size=(x_copy.size(2), x_copy.size(3)),
                                mode="bilinear", align_corners=True)
            else:
                # Padding in case the incomping volumes are of different sizes
                diffY = x_copy.size()[2] - x.size()[2]
                diffX = x_copy.size()[3] - x.size()[3]
                x = F.pad(x, (diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2))

        # Concatenate
        x = torch.cat([x_copy, x], dim=1)
        x = self.up_conv(x)
        return x


class UNetWithResnet18Encoder(nn.Module):
    class Decoder(nn.Module):
        def __init__(self, in_channels, skip_channels, out_channels):
            super(UNetWithResnet18Encoder.Decoder, self).__init__()
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.up_conv = x2conv(out_channels + skip_channels, out_channels)

        def forward(self, x_copy, x):
            x = self.up(x)
            if x.size(2) != x_copy.size(2) or x.size(3) != x_copy.size(3):
                x = F.interpolate(x, size=(x_copy.size(2), x_copy.size(3)), mode='bilinear', align_corners=True)
            x = torch.cat((x_copy, x), dim=1)
            x = self.up_conv(x)
            return x


    def __init__(self, num_classes, in_channels=3, freeze_bn=False, sigmoid=True):
        super(UNetWithResnet18Encoder, self).__init__()
        self.sigmoid = sigmoid
        resnet18 = models.resnet18(pretrained=True)
        
        if in_channels != 3:
            resnet18.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.encoder1 = nn.Sequential(resnet18.conv1, resnet18.bn1, resnet18.relu)
        self.encoder2 = resnet18.layer1
        self.encoder3 = resnet18.layer2
        self.encoder4 = resnet18.layer3
        self.encoder5 = resnet18.layer4

        self.up1 = UNetWithResnet18Encoder.Decoder(512, 256, 256)
        self.up2 = UNetWithResnet18Encoder.Decoder(256, 128, 128)
        self.up3 = UNetWithResnet18Encoder.Decoder(128, 64, 64)
        self.up4 = UNetWithResnet18Encoder.Decoder(64, 64, 64)

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
        x = F.interpolate(x, size=(x.size(2)*2, x.size(3)*2), mode='bilinear', align_corners=True)

        x = self.final_conv(x)
        
        if self.sigmoid:
            x = torch.sigmoid(x)
        return x

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


class FlexibleResNet(nn.Module):
    def __init__(self, num_logic_outputs, num_regression_groups, regression_size, in_channels=3, sigmoid=True):
        super(FlexibleResNet, self).__init__()
        self.sigmoid = sigmoid

        # suggest num_logic_outputs >= num_regression_groups
        self.num_logic_outputs = num_logic_outputs
        self.num_regression_groups = num_regression_groups
        self.regression_size = regression_size

        # remove the last two layers of resnet, replace with new layers
        self.resnet = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])
        if in_channels != 3:
            self.resnet[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            # nn.Linear(512, 512),
            # nn.BatchNorm1d(512),
            # nn.LeakyReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(512, 512),
            # nn.BatchNorm1d(512),
            # nn.LeakyReLU(),
            # nn.Dropout(0.5),
            nn.Linear(512, num_logic_outputs + num_regression_groups * regression_size)
        )

    def forward(self, x):
        outputs = self.resnet(x)
        outputs = self.global_avg_pool(outputs)
        outputs = outputs.view(outputs.size(0), -1)
        outputs = self.fc(outputs)
        logic_outputs = outputs[:, :self.num_logic_outputs]
        regression_outputs = outputs[:, self.num_logic_outputs:].view(-1, self.num_regression_groups, self.regression_size)
        if self.sigmoid:    
            logic_outputs = F.sigmoid(logic_outputs)
            regression_outputs = F.sigmoid(regression_outputs)
        return logic_outputs, regression_outputs

# model = FlexibleResNet(num_logic_outputs=4, num_regression_groups=3)

# nn.L1Loss
class CustomLoss(nn.Module):
    def __init__(self, logic_weight=1, regression_weight=1, regression_loss_fn=nn.L1Loss(reduction='none'), ):
        super(CustomLoss, self).__init__()
        self.regression_loss_fn = regression_loss_fn
        self.classification_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.logic_weight = logic_weight
        self.regression_weight = regression_weight

    def forward(self, logic_outputs, logic_labels, regression_outputs, regression_targets):
        classification_loss = torch.mean(torch.sum(self.classification_loss_fn(logic_outputs, logic_labels), dim=-1))
        num_groups = regression_outputs.size(1)
        regression_mask = logic_labels[:, :num_groups]
        regression_mask_expanded = logic_labels[:, :num_groups].unsqueeze(-1)  # (n, num_groups, 1)
        regression_mask_expanded = regression_mask_expanded.expand(-1, -1, regression_outputs.size(2))  # (n, num_groups, num_regressions)
        regression_loss_all = self.regression_loss_fn(regression_outputs, regression_targets)
        regression_loss_masked = regression_loss_all * regression_mask_expanded
        regression_loss_masked = torch.sum(torch.mean(regression_loss_masked, dim=-1), dim=-1)
        regression_mask_sum = torch.sum(regression_mask, dim=-1)
        regression_loss = torch.mean(regression_loss_masked / (regression_mask_sum + 1e-6))
        total_loss = self.logic_weight * classification_loss + self.regression_weight * regression_loss
        return total_loss, self.logic_weight * classification_loss, self.regression_weight * regression_loss
    

class CombinedModel(nn.Module):
    def __init__(self, segment_model: nn.Module, predict_model: nn.Module):
        super(CombinedModel, self).__init__()
        self.segment_model = segment_model
        self.predict_model = predict_model

    def forward(self, x: torch.Tensor):
        seg_masks = self.segment_model(x)
        x = torch.cat((x, seg_masks), dim=1)
        logic_outputs, regression_outputs = self.predict_model(x)
        return seg_masks, logic_outputs, regression_outputs
    

class MultiLabelResNet(nn.Module):
    def __init__(self, num_labels, input_channels=3, sigmoid=True, pretrained=True,):
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
    

class CombinedModelNoRegression(nn.Module):
    def __init__(self, segment_model: nn.Module, predict_model: nn.Module, cat_layers:int=None):
        super(CombinedModelNoRegression, self).__init__()
        self.segment_model = segment_model
        self.predict_model = predict_model
        self.cat_layers = cat_layers

    def forward(self, x: torch.Tensor):
        seg_masks = self.segment_model(x)
        if self.cat_layers:
            seg_masks_ = seg_masks[:,0:self.cat_layers]
            x = torch.cat((x, seg_masks_), dim=1)
        else:
            x = torch.cat((x, seg_masks), dim=1)
        logic_outputs = self.predict_model(x)
        return seg_masks, logic_outputs
    
    def freeze_segment_model(self):
        for param in self.segment_model.parameters():
            param.requires_grad = False
        self.segment_model.eval()

    def unfreeze_segment_model(self):
        for param in self.segment_model.parameters():
            param.requires_grad = True
        self.segment_model.train()


class MobileNetSemanticSegmentation(nn.Module):
    class Decoder(nn.Module):
        def __init__(self, in_channels, skip_channels, out_channels):
            super(MobileNetSemanticSegmentation.Decoder, self).__init__()
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.up_conv = self.double_conv(out_channels + skip_channels, out_channels)

        def forward(self, x_copy, x):
            x = self.up(x)
            if x.size(2) != x_copy.size(2) or x.size(3) != x_copy.size(3):
                x = F.interpolate(x, size=(x_copy.size(2), x_copy.size(3)), mode='bilinear', align_corners=True)
            x = torch.cat((x_copy, x), dim=1)
            x = self.up_conv(x)
            return x

        def double_conv(self, in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

    def __init__(self, num_classes, in_channels=3, freeze_bn=False, sigmoid=True):
        super(MobileNetSemanticSegmentation, self).__init__()
        self.sigmoid = sigmoid
        mobilenet_v2 = models.mobilenet_v2(pretrained=True)
        if in_channels != 3:
            mobilenet_v2.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.encoder = mobilenet_v2.features

        self.up1 = MobileNetSemanticSegmentation.Decoder(1280, 96, 96)
        self.up2 = MobileNetSemanticSegmentation.Decoder(96, 32, 32)
        self.up3 = MobileNetSemanticSegmentation.Decoder(32, 24, 24)
        self.up4 = MobileNetSemanticSegmentation.Decoder(24, 16, 16)

        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)
        self._initialize_weights()

        if freeze_bn:
            self.freeze_bn()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        skip_connections = []
        for i, module in enumerate(self.encoder):
            x = module(x)
            if i in {1, 3, 6, 13}:
                skip_connections.append(x)
        
        skip_connections = skip_connections[::-1]  # Reverse the skip connections

        x = self.up1(skip_connections[0], x)
        x = self.up2(skip_connections[1], x)
        x = self.up3(skip_connections[2], x)
        x = self.up4(skip_connections[3], x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        x = self.final_conv(x)
        if self.sigmoid:
            x = torch.sigmoid(x)
        return x

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
    

class MobileNetMultiBinClassifier(nn.Module):
    def __init__(self, num_tasks, input_channels, sigmoid=True):
        super(MobileNetMultiBinClassifier, self).__init__()
        self.sigmoid = sigmoid
        self.shared_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((6, 6))
        )

        self.task_layers = nn.ModuleList([
            nn.Linear(128 * 6 * 6, 1) for _ in range(num_tasks)
        ])

    def forward(self, x):
        x = self.shared_layers(x)
        x = x.view(x.size(0), -1)
        outputs = [task_layer(x) for task_layer in self.task_layers]
        # Concatenate outputs to have them in a single vector
        outputs = torch.cat(outputs, dim=1)
        if self.sigmoid:
            outputs = torch.sigmoid(outputs)
        return outputs


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.atrous_block1 = nn.Conv2d(in_channels, out_channels, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18)
        self.conv_out = nn.Conv2d(out_channels * 4, out_channels, 1, 1)

    def forward(self, x):
        x1 = self.atrous_block1(x)
        x6 = self.atrous_block6(x)
        x12 = self.atrous_block12(x)
        x18 = self.atrous_block18(x)
        x = torch.cat([x1, x6, x12, x18], dim=1)
        return self.conv_out(x)

class DeepLabV3PlusMobileNetV3(nn.Module):
    def __init__(self, num_classes, in_channels=3, sigmoid=True):
        super(DeepLabV3PlusMobileNetV3, self).__init__()
        self.sigmoid = sigmoid
        mobilenet_v3 = models.mobilenet_v3_large(pretrained=True)

        if in_channels != 3:
            mobilenet_v3.features[0][0] = nn.Conv2d(
                in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False
            )

        self.encoder = mobilenet_v3.features

        intermediate_channel = self.encoder[-1].out_channels 
        self.aspp = ASPP(intermediate_channel, 256)

        self.decoder = nn.Sequential(
            nn.Conv2d(256 + in_channels, 256, kernel_size=3, padding=1),  # Concatenated with original input
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        original_input = x 
        x_encoded = self.encoder(x)
        x_aspp = self.aspp(x_encoded)

        x = F.interpolate(x_aspp, size=original_input.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, original_input], dim=1)  # Concatenate with original input
        x = self.decoder(x)

        if self.sigmoid:
            x = torch.sigmoid(x)

        return x
    

class MultiLabelMobileNetV3Small(nn.Module):
    def __init__(self, num_labels, input_channels=3, sigmoid=True, pretrained=True):
        super(MultiLabelMobileNetV3Small, self).__init__()
        mobilenet_v3_small = models.mobilenet_v3_small(pretrained=pretrained)
        self.sigmoid = sigmoid

        if input_channels != 3:
            mobilenet_v3_small.features[0][0] = nn.Conv2d(
                input_channels, 16, kernel_size=3, stride=2, padding=1, bias=False
            )

        self.model = mobilenet_v3_small

        num_ftrs = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(num_ftrs, num_labels)

    def forward(self, x):
        x = self.model(x)
        if self.sigmoid:
            x = torch.sigmoid(x)
        return x


class MultiLabelMobileNetV3Large(nn.Module):
    def __init__(self, num_labels, input_channels=3, sigmoid=True, pretrained=True):
        super(MultiLabelMobileNetV3Large, self).__init__()
        mobilenet_v3_small = models.mobilenet_v3_large(pretrained=pretrained)
        self.sigmoid = sigmoid

        if input_channels != 3:
            mobilenet_v3_small.features[0][0] = nn.Conv2d(
                input_channels, 16, kernel_size=3, stride=2, padding=1, bias=False
            )

        self.model = mobilenet_v3_small

        num_ftrs = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(num_ftrs, num_labels)

    def forward(self, x):
        x = self.model(x)
        if self.sigmoid:
            x = torch.sigmoid(x)
        return x
