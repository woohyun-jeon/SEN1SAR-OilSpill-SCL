import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def get_encoder(encoder_name, pretrained='IMAGENET1K_V1'):
    if encoder_name == 'resnet50':
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None
    else:
        raise ValueError(f"Encoder {encoder_name} is not supported")

    model = getattr(models, encoder_name)(weights=weights)
    modules = list(model.children())[:-2]

    encoder = nn.Sequential(*modules)
    encoder.name = encoder_name

    return encoder


class UNet(nn.Module):
    def __init__(self, encoder, n_classes=1):
        super(UNet, self).__init__()
        self.encoder = encoder
        self.decoder = UNetDecoder(n_classes)

    def forward(self, x):
        features = []
        x = self.encoder[0](x)  # Conv1
        x = self.encoder[1](x)  # BatchNorm1
        x = self.encoder[2](x)  # ReLU
        features.append(x)      # features[0]
        x = self.encoder[3](x)  # MaxPool
        x = self.encoder[4](x)  # Layer1
        features.append(x)      # features[1]
        x = self.encoder[5](x)  # Layer2
        features.append(x)      # features[2]
        x = self.encoder[6](x)  # Layer3
        features.append(x)      # features[3]
        x = self.encoder[7](x)  # Layer4
        features.append(x)      # features[4]
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

        # set multi-scale projectors
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 512, 1),
                nn.BatchNorm2d(num_features=512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, feature_dim, 1)
            ) for in_channels in [64, 256, 512, 1024, 2048]  # 각 feature map의 채널 수
        ])

        # set attention module for feature fusion
        self.attention = nn.MultiheadAttention(feature_dim, 8)

    def forward(self, x):
        features = []
        projected_features = []

        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in [2, 4, 5, 6, 7]:
                features.append(x)
                # project each feature map
                proj = self.projectors[len(projected_features)](x)
                # set global average pooling
                proj = F.adaptive_avg_pool2d(proj, 1).squeeze(-1).squeeze(-1)
                projected_features.append(proj)

        # get segmentation output
        segmentation = self.decoder(features)

        # stack projected features and apply attention
        stacked_features = torch.stack(projected_features, dim=0)  # [5, B, feature_dim]
        attended_features, _ = self.attention(stacked_features, stacked_features, stacked_features)

        # combine features with attention weights
        final_features = torch.mean(attended_features, dim=0)  # [B, feature_dim]

        return segmentation, F.normalize(final_features, dim=1)


def get_model(model_name, encoder_name, pretrained='IMAGENET1K_V1', n_classes=1):
    encoder = get_encoder(encoder_name, pretrained)

    if model_name == 'UNet':
        return UNet(encoder, n_classes)
    elif model_name == 'SupConUNet':
        return SupConUNet(encoder, n_classes)
    else:
        raise ValueError(f"Model {model_name} is not supported")


# test code
if __name__ == "__main__":
    batch_size = 1
    input_channels = 3
    input_height = 256
    input_width = 256
    n_classes = 1

    encoder = get_encoder('resnet50', pretrained=None)
    model = UNet(encoder, n_classes)

    input_tensor = torch.randn(batch_size, input_channels, input_height, input_width)

    print("Starting forward pass...")
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")

    print("Model test completed successfully!")