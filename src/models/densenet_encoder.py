"""
DenseNet121 Visual Encoder with CheXpert Pretrain for Entity Detection

Uses torchxrayvision library which includes CheXpert pretrained weights
"""
import torch
import torch.nn as nn

class DenseNet121Encoder(nn.Module):
    """
    DenseNet121 pretrained on CheXpert for chest X-ray entity detection
    
    Better than CLIP variants for pure visual classification tasks.
    """
    def __init__(self, pretrained='chexpert', output_dim=1024):
        super().__init__()
        
        self.output_dim = output_dim
        
        # Try to use torchxrayvision CheXpert model
        try:
            import torchxrayvision as xrv
            print(f"Loading DenseNet121 with CheXpert pretraining...")
            
            # Load model with CheXpert weights
            self.model = xrv.models.DenseNet(weights="densenet121-res224-chex")
            
            # Get feature dimension from DenseNet121 (1024)
            self.feature_dim = 1024
            
            print(f"✓ Loaded DenseNet121-CheXpert (features: {self.feature_dim}d)")
            
        except ImportError:
            print("⚠️  torchxrayvision not found, using torchvision DenseNet121 (ImageNet)")
            
            # Import here to avoid global torchvision issues
            import torchvision.models as models
            
            # Fallback to ImageNet pretrained
            self.model = models.densenet121(pretrained=True)
            
            # Remove classifier to get features
            self.model.classifier = nn.Identity()
            
            self.feature_dim = 1024  # DenseNet121 feature dim
            
            print(f"✓ Loaded DenseNet121-ImageNet as fallback (features: {self.feature_dim}d)")
        
        # Optional: projection layer if output_dim != feature_dim
        if output_dim != self.feature_dim:
            self.projection = nn.Linear(self.feature_dim, output_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] input images (224x224 expected)
        
        Returns:
            features: [B, output_dim] visual features for entity detection
        """
        # torchxrayvision models expect GRAYSCALE (1 channel)
        # MIMIC-CXR images are already grayscale, but loaded as 3-channel RGB (by default in new dataset code)
        
        if hasattr(self.model, 'features'):
            # Torchvision DenseNet (Fallback) - Exepcts 3 channels (RGB)
            # If input is 1 channel, repeat it to make 3 channels
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
        else:
            # torchxrayvision model - Expects 1 channel (Grayscale)
            # If input is 3 channels, convert to grayscale
            if x.shape[1] == 3:
                # Standard RGB to grayscale conversion: 0.299*R + 0.587*G + 0.114*B
                x = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
                # x now [B, 1, H, W]
        
        # For torchxrayvision models, they expect normalized inputs
        # Most models output [B, num_classes] but we want features
        
        if hasattr(self.model, 'features'):
            # Standard DenseNet from torchvision
            features = self.model.features(x)  # [B, 1024, 7, 7]
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))  # [B, 1024, 1, 1]
            features = features.view(features.size(0), -1)  # [B, 1024]
        else:
            # torchxrayvision model 
            # These models have different architecture
            try:
                # Get features before final classifier
                features = self.model.features(x)
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1)
            except:
                # Fallback: use full model and extract from intermediate layer
                features = self.model(x)  # May output predictions, need to hook features
        
        # Project to desired dimension
        features = self.projection(features)
        
        return features


def create_visual_encoder(config):
    """
    Factory function to create visual encoder based on config
    
    Args:
        config: dict with visual_encoder settings
    
    Returns:
        visual_encoder: nn.Module
    """
    encoder_config = config.get('visual_encoder', {})
    encoder_type = encoder_config.get('type', 'resnet50')
    
    if encoder_type == 'densenet121' or encoder_type == 'chexpert':
        # Use DenseNet121 with CheXpert pretraining
        pretrained = encoder_config.get('pretrained_source', 'chexpert')
        output_dim = encoder_config.get('output_dim', 1024)
        
        return DenseNet121Encoder(pretrained=pretrained, output_dim=output_dim)
    
    elif encoder_type == 'resnet50':
        # Fallback to ResNet50
        import torchvision.models as models
        model = models.resnet50(pretrained=True)
        model.fc = nn.Identity()  # Remove classifier
        return model
    
    else:
        raise ValueError(f"Unknown visual encoder type: {encoder_type}")


if __name__ == "__main__":
    # Test encoder
    print("Testing DenseNet121 encoder...")
    
    encoder = DenseNet121Encoder(output_dim=512)
    
    # Dummy input
    x = torch.randn(2, 3, 224, 224)
    
    features = encoder(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Expected: [2, 512]")
    
    assert features.shape == (2, 512), "Shape mismatch!"
    print("✓ Test passed!")
