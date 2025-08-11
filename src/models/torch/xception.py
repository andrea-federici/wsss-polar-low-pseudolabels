import timm
import torch.nn as nn
from torchinfo import summary


class Xception(nn.Module):
    def __init__(self, num_classes=2):
        super(Xception, self).__init__()

        # Feature Extractor
        # When pretrained=True, pretrained ImageNet-1K weights are loaded, as specified
        # here: https://huggingface.co/docs/timm/reference/models
        self.feature_extractor = timm.create_model(
            "legacy_xception", pretrained=True, features_only=True
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features=2048, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=32, out_features=num_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)

        # Get the last feature map (self.feature_extractor(x) is a list)
        last_features = x[-1]

        # Global Average Pooling. Input shape: (batch_size, channels, H, W). This layer
        # reduces each feature map by averaging over the spatial dimensions (H, W),
        # resulting in a tensor of shape (batch_size, channels).
        pooled_features = last_features.mean(dim=[2, 3])

        x = self.classifier(pooled_features)
        return x

    def get_last_conv_layer(self):
        last_conv = None
        for _, layer in reversed(list(self.feature_extractor.named_modules())):
            # We are interested in the last pointwise convolution layer
            if isinstance(layer, nn.Conv2d) and layer.kernel_size == (
                1,
                1,
            ):
                last_conv = layer
                break
        if last_conv is None:
            raise ValueError(
                "No pointwise convolutional layer found in the feature extractor."
            )
        return last_conv


# --- TESTING ---

# if __name__ == "__main__":
#     model = Xception()
#     summary(model, input_size=(1, 3, 500, 500))
