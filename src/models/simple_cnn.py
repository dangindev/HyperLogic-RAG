"""
Simple CNN visual encoder to replace torchvision ResNet
Avoids torchvision compatibility issues
"""
import torch
import torch.nn as nn

class SimpleCNNEncoder(nn.Module):
    """
    Simple CNN backbone for visual encoding
    Output: [B, 2048, 7, 7] similar to ResNet-50
    """
    def __init__(self):
        super().__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 2048, 3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, 224, 224]
        Returns:
            features: [B, 2048, 7, 7]
        """
        x = self.conv1(x)   # [B, 64, 56, 56]
        x = self.conv2(x)   # [B, 128, 14, 14]
        x = self.conv3(x)   # [B, 256, 7, 7]
        x = self.conv4(x)   # [B, 512, 7, 7]
        x = self.conv5(x)   # [B, 2048, 7, 7]
        return x

if __name__ == "__main__":
    # Test
    model = SimpleCNNEncoder()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"✓ Simple CNN encoder works!")
