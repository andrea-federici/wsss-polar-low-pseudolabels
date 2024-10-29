
import torch
import torch.nn as nn
from torchinfo import summary
import timm

import config

class XceptionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(XceptionModel, self).__init__()
        
        # Feature Extractor
        self.feature_extractor = timm.create_model('xception', pretrained=True, features_only=True)

        # Inspect feature_info to determine the number of channels. For Xception this is typically 2048.
        feature_info = self.feature_extractor.feature_info
        num_channels = feature_info[-1]['num_chs']

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features=2048, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            nn.Linear(in_features=32, out_features=num_classes)
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        last_features = x[-1] # Get the last feature map (self.feature_extractor(x) is a list)
        pooled_features = last_features.mean(dim=[2, 3]) # Global Average Pooling. Input shape: (batch_size, channels, H, W). This layer reduces each feature map by averaging over the spatial dimensions (H, W), resulting in a tensor of shape (batch_size, channels).
        x = self.classifier(pooled_features)
        return x

    def get_last_conv_layer(self):
        last_conv = None
        for name, layer in reversed(list(self.feature_extractor.named_modules())):
            if isinstance(layer, nn.Conv2d) and layer.kernel_size == (1,1): # We are interested in the last pointwise convolution layer.
                last_conv = layer
                print(f'Selected pointwise convolution layer: {name} - {layer}')
                break
        if last_conv is None:
            raise ValueError("No convolutional layer found in the feature extractor.")
        return last_conv


class ConvModel(nn.Module):
    def __init__(self, input_channels=3, num_classes=2):
        super(ConvModel, self).__init__()

        # Feature Extractor
        self.feature_extractor = nn.Sequential(

            # Block 1
            nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )

        # Global Average Pooling (GAP)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1)) # This layer reduces the spatial dimensions of the feature maps to 1x1

        # Classifier
        self.classifier = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1) # Flatten layer
        x = self.classifier(x)
        return x
    
    def get_last_conv_layer(self):
        for layer in reversed(self.feature_extractor):
            if isinstance(layer, nn.Conv2d):
                return layer
        raise ValueError("No convolutional layer found in the feature extractor.")


# model = XceptionModel()
# model.to(config.device)
# summary(model, input_size=(1, 3, 299, 299), depth=6) # Xception model

