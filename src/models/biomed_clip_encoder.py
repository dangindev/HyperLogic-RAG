
import torch
import torch.nn as nn
import sys

class BiomedCLIPEncoder(nn.Module):
    def __init__(self, pretrained=True, freeze=False):
        super().__init__()
        self.output_dim = 512
        
        print("Loading BiomedCLIP encoder...")
        try:
            import open_clip
            # Load BiomedCLIP from HuggingFace
            model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            model, _, _ = open_clip.create_model_and_transforms(model_name)
            self.model = model.visual
            
            if freeze:
                print("🔒 Freezing BiomedCLIP backbone")
                for param in self.model.parameters():
                    param.requires_grad = False
            else:
                print("🔓 BiomedCLIP backbone trainable")

            print("✓ BiomedCLIP loaded successfully")
            
            # Project 768 (ViT-B) -> 512 (HyperLogic exp.)
            self.feature_proj = nn.Linear(768, 512)
            
        except Exception as e:
            print(f"❌ Failed to load BiomedCLIP: {e}")
            raise e
            
    def forward(self, x):
        """
        Args:
            x: [B, 3, 224, 224] images
        Returns:
            features: [B, 512, 14, 14] spatial features
        """
        # 1. Get Patches from TIMM trunk
        # output: [B, 197, 768] (with CLS)
        raw_feat = self.model.trunk.forward_features(x)
        
        # 2. Exclude CLS (index 0)
        patches = raw_feat[:, 1:, :] # [B, 196, 768]
        
        # 3. Project to 512
        patches = self.feature_proj(patches) # [B, 196, 512]
        
        # 4. Reshape to [B, 512, 14, 14] for HyperLogic compatibility
        # N=196 -> 14x14
        B = patches.size(0)
        patches = patches.permute(0, 2, 1).reshape(B, 512, 14, 14)
        
        return patches
