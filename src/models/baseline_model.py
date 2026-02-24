import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class EncoderCNN(nn.Module):
    def __init__(self, embed_dim, encoder_type='resnet50'):
        super(EncoderCNN, self).__init__()
        self.encoder_type = encoder_type
        
        if encoder_type == 'resnet50':
            # Use Hugging Face ResNet
            self.encoder = AutoModel.from_pretrained("microsoft/resnet-50")
            hidden_size = 2048
        elif encoder_type == 'densenet121':
            # Use torchvision DenseNet-121
            import torchvision.models as models
            densenet = models.densenet121(pretrained=True)
            # Remove classifier
            self.encoder = nn.Sequential(*list(densenet.children())[:-1])
            hidden_size = 1024  # DenseNet-121 output
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # Project to embed_dim
        self.fc = nn.Linear(hidden_size, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim, momentum=0.01)

    def forward(self, images):
        # images: [B, 3, H, W]
        
        if self.encoder_type == 'resnet50':
            outputs = self.encoder(images)
            features = outputs.pooler_output.flatten(1)  # [B, 2048]
        elif self.encoder_type == 'densenet121':
            features = self.encoder(images)  # [B, 1024, 7, 7]
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = features.flatten(1)  # [B, 1024]
        
        features = self.fc(features)
        features = self.bn(features)
        return features  # [B, embed_dim]

class DecoderRNN(nn.Module):
    # Using Transformer Decoder for better baseline than LSTM
    def __init__(self, embed_dim, vocab_size, num_heads, num_layers, max_length=100):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = nn.Parameter(torch.zeros(1, max_length, embed_dim))
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.linear = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, features, captions):
        # features: [B, embed_dim] -> [B, 1, embed_dim]
        # captions: [B, seq_len]
        
        embeddings = self.embed(captions) # [B, seq_len, embed_dim]
        # Add simpler visual projection
        visual_features = features.unsqueeze(1) # [B, 1, embed_dim]
        
        # In a standard image captioning transformer, visual features are memory
        output = self.transformer_decoder(tgt=embeddings, memory=visual_features)
        output = self.linear(output)
        return output

class BaselineModel(nn.Module):
    def __init__(self, config):
        super(BaselineModel, self).__init__()
        embed_dim = config['model']['embed_dim']
        encoder_type = config['model'].get('encoder', 'resnet50')
        self.encoder = EncoderCNN(embed_dim, encoder_type=encoder_type)
        self.decoder = DecoderRNN(
            embed_dim=embed_dim,
            vocab_size=config['model']['vocab_size'],
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_layers'],
            max_length=config['dataset']['max_length']
        )
    
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
