
import torch.nn as nn
import timm

class XceptionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(XceptionModel, self).__init__()
        
        # Feature Extractor
        self.feature_extractor = timm.create_model('xception', pretrained=True, features_only=True)

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
