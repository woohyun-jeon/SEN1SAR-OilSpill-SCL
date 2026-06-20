import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def get_encoder(encoder_name, pretrained='IMAGENET1K_V1'):
    if encoder_name == 'resnet50':
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    elif encoder_name == 'resnet101':
        weights = models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
    else:
        raise ValueError(f"Encoder {encoder_name} is not supported")

    model = getattr(models, encoder_name)(weights=weights)
    modules = list(model.children())[:-2]

    encoder = nn.Sequential(*modules)
    encoder.name = encoder_name

    return encoder


def get_model(model_name, encoder_name, pretrained='IMAGENET1K_V1', n_classes=1):
    encoder = get_encoder(encoder_name, pretrained)

    if model_name == 'UNet':
        return UNet(encoder, n_classes)
    elif model_name == 'SupConUNet':
        return SupConUNet(encoder, n_classes)
    elif model_name == 'DeepLabV3Plus':
        return DeepLabV3Plus(encoder, n_classes)
    elif model_name == 'SupConDeepLabV3Plus':
        return SupConDeepLabV3Plus(encoder, n_classes)
    else:
        raise ValueError(f"Model {model_name} is not supported")


class UNet(nn.Module):
    def __init__(self, encoder, n_classes=1):
        super(UNet, self).__init__()
        self.encoder = encoder
        self.decoder = UNetDecoder(n_classes)

    def forward(self, x):
        features = []
        x = self.encoder[0](x)
        x = self.encoder[1](x)
        x = self.encoder[2](x)
        features.append(x)
        x = self.encoder[3](x)
        x = self.encoder[4](x)
        features.append(x)
        x = self.encoder[5](x)
        features.append(x)
        x = self.encoder[6](x)
        features.append(x)
        x = self.encoder[7](x)
        features.append(x)
        out = self.decoder(features)

        return out


class UNetDecoder(nn.Module):
    def __init__(self, n_classes):
        super(UNetDecoder, self).__init__()
        self.up1 = UpConv(2048, 1024, 1024)
        self.up2 = UpConv(1024, 512, 512)
        self.up3 = UpConv(512, 256, 256)
        self.up4 = UpConv(256, 64, 64)
        self.up_final = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, features):
        x = self.up1(features[4], features[3])
        x = self.up2(x, features[2])
        x = self.up3(x, features[1])
        x = self.up4(x, features[0])
        x = self.up_final(x)
        x = self.final(x)

        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels=out_channels + skip_channels, out_channels=out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class SupConUNet(nn.Module):
    def __init__(self, encoder, n_classes=1, feature_dim=128):
        super(SupConUNet, self).__init__()
        self.encoder = encoder
        self.decoder = UNetDecoder(n_classes)

        self.pixel_projector = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, feature_dim, 1),
            nn.AdaptiveAvgPool2d((64, 64))
        )

    def forward(self, x):
        features = []

        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in [2, 4, 5, 6, 7]:
                features.append(x)

        segmentation = self.decoder(features)

        pixel_features = self.pixel_projector(features[-1])
        pixel_features = F.normalize(pixel_features, dim=1)

        return segmentation, pixel_features


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.shape[-2:]
        x = self.global_pool(x)
        x = self.conv(x)
        x = self.relu(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super().__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3Plus(nn.Module):
    def __init__(self, encoder, n_classes=1):
        super().__init__()
        self.encoder = encoder
        self.aspp = ASPP(2048, 256, [6, 12, 18])

        self.low_level_conv = nn.Sequential(
            nn.Conv2d(64, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, n_classes, 1)
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        features = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i == 2:
                low_level_feat = x
            if i in [2, 4, 5, 6, 7]:
                features.append(x)

        x = features[-1]
        x = self.aspp(x)
        x = F.interpolate(x, size=low_level_feat.shape[-2:], mode='bilinear', align_corners=False)

        low_level_feat = self.low_level_conv(low_level_feat)
        x = torch.cat([x, low_level_feat], dim=1)

        x = self.decoder(x)

        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        return x


class SupConDeepLabV3Plus(nn.Module):
    def __init__(self, encoder, n_classes=1, feature_dim=128):
        super().__init__()
        self.encoder = encoder
        self.aspp = ASPP(2048, 256, [6, 12, 18])

        self.low_level_conv = nn.Sequential(
            nn.Conv2d(64, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, n_classes, 1)
        )

        self.pixel_projector = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, feature_dim, 1),
            nn.AdaptiveAvgPool2d((64, 64))
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        features = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i == 2:
                low_level_feat = x
            if i in [2, 4, 5, 6, 7]:
                features.append(x)

        aspp_out = self.aspp(features[-1])
        aspp_out = F.interpolate(aspp_out, size=low_level_feat.shape[-2:], mode='bilinear', align_corners=False)

        low_level_feat = self.low_level_conv(low_level_feat)
        x_seg = torch.cat([aspp_out, low_level_feat], dim=1)
        x_seg = self.decoder(x_seg)
        x_seg = F.interpolate(x_seg, size=input_size, mode='bilinear', align_corners=False)

        x_con = self.pixel_projector(features[-1])
        x_con = F.normalize(x_con, dim=1)

        return x_seg, x_con


if __name__ == "__main__":
    batch_size = 2
    input_channels = 3
    input_height = 256
    input_width = 256
    n_classes = 1

    encoder = get_encoder('resnet50', pretrained=None)
    input_tensor = torch.randn(batch_size, input_channels, input_height, input_width)

    model_unet = UNet(encoder, n_classes)
    output_unet = model_unet(input_tensor)
    print(f"UNet output shape: {output_unet.shape}")

    encoder = get_encoder('resnet50', pretrained=None)
    model_deeplabv3plus = DeepLabV3Plus(encoder, n_classes)
    output_deeplabv3plus = model_deeplabv3plus(input_tensor)
    print(f"DeepLabV3+ output shape: {output_deeplabv3plus.shape}")

    encoder = get_encoder('resnet50', pretrained=None)
    model_supcon = SupConDeepLabV3Plus(encoder, n_classes)
    seg_output, con_output = model_supcon(input_tensor)
    print(f"SupConDeepLabV3+ segmentation output shape: {seg_output.shape}")
    print(f"SupConDeepLabV3+ contrastive output shape: {con_output.shape}")
