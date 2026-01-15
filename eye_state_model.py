#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eye State Classification Model

This module provides a classifier model based on RITnet's DenseNet2D encoder
for binary eye state classification (open/close).
"""

import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class DenseNet2D_down_block(nn.Module):
    """DenseNet down-sampling block (from RITnet)."""
    
    def __init__(self, input_channels, output_channels, down_size, dropout=False, prob=0):
        super(DenseNet2D_down_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))
        self.conv21 = nn.Conv2d(input_channels + output_channels, output_channels, kernel_size=(1, 1), padding=(0, 0))
        self.conv22 = nn.Conv2d(output_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))
        self.conv31 = nn.Conv2d(input_channels + 2 * output_channels, output_channels, kernel_size=(1, 1), padding=(0, 0))
        self.conv32 = nn.Conv2d(output_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))
        self.max_pool = nn.AvgPool2d(kernel_size=down_size) if down_size else None
        
        self.relu = nn.LeakyReLU()
        self.down_size = down_size
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)
        self.dropout3 = nn.Dropout(p=prob)
        self.bn = nn.BatchNorm2d(num_features=output_channels)
    
    def forward(self, x):
        if self.down_size is not None and self.max_pool is not None:
            x = self.max_pool(x)
        
        if self.dropout:
            x1 = self.relu(self.dropout1(self.conv1(x)))
            x21 = torch.cat((x, x1), dim=1)
            x22 = self.relu(self.dropout2(self.conv22(self.conv21(x21))))
            x31 = torch.cat((x21, x22), dim=1)
            out = self.relu(self.dropout3(self.conv32(self.conv31(x31))))
        else:
            x1 = self.relu(self.conv1(x))
            x21 = torch.cat((x, x1), dim=1)
            x22 = self.relu(self.conv22(self.conv21(x21)))
            x31 = torch.cat((x21, x22), dim=1)
            out = self.relu(self.conv32(self.conv31(x31)))
        
        return self.bn(out)


class EyeStateClassifier(nn.Module):
    """
    Eye State Classifier based on DenseNet2D encoder.
    
    This model uses the encoder part of RITnet's DenseNet2D architecture
    followed by global average pooling and fully connected layers for
    binary classification (open/close).
    
    Args:
        in_channels: Number of input channels (1 for grayscale)
        num_classes: Number of output classes (2 for open/close)
        channel_size: Number of channels in dense blocks (default: 32)
        dropout: Whether to use dropout (default: True)
        prob: Dropout probability (default: 0.2)
        pretrained_encoder: Path to pretrained RITnet weights (optional)
    """
    
    def __init__(self, in_channels=1, num_classes=2, channel_size=32, 
                 dropout=True, prob=0.2, pretrained_encoder=None):
        super(EyeStateClassifier, self).__init__()
        
        # Encoder blocks (same as RITnet)
        self.down_block1 = DenseNet2D_down_block(
            input_channels=in_channels, output_channels=channel_size,
            down_size=None, dropout=dropout, prob=prob
        )
        self.down_block2 = DenseNet2D_down_block(
            input_channels=channel_size, output_channels=channel_size,
            down_size=(2, 2), dropout=dropout, prob=prob
        )
        self.down_block3 = DenseNet2D_down_block(
            input_channels=channel_size, output_channels=channel_size,
            down_size=(2, 2), dropout=dropout, prob=prob
        )
        self.down_block4 = DenseNet2D_down_block(
            input_channels=channel_size, output_channels=channel_size,
            down_size=(2, 2), dropout=dropout, prob=prob
        )
        self.down_block5 = DenseNet2D_down_block(
            input_channels=channel_size, output_channels=channel_size,
            down_size=(2, 2), dropout=dropout, prob=prob
        )
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channel_size, 64),
            nn.ReLU(),
            nn.Dropout(p=prob),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Load pretrained encoder if provided
        if pretrained_encoder is not None:
            self._load_pretrained_encoder(pretrained_encoder)
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
    def _load_pretrained_encoder(self, pretrained_path):
        """Load pretrained encoder weights from RITnet model."""
        print(f"Loading pretrained encoder from: {pretrained_path}")
        
        pretrained_dict = torch.load(pretrained_path, map_location='cpu')
        model_dict = self.state_dict()
        
        # Filter encoder weights only (down_block*)
        encoder_keys = [k for k in pretrained_dict.keys() if 'down_block' in k]
        pretrained_encoder = {k: v for k, v in pretrained_dict.items() if k in encoder_keys}
        
        # Check which keys match
        matched_keys = [k for k in pretrained_encoder.keys() if k in model_dict]
        print(f"Matched {len(matched_keys)} / {len(encoder_keys)} encoder keys")
        
        # Update model dict
        model_dict.update(pretrained_encoder)
        self.load_state_dict(model_dict)
        
        print("Pretrained encoder weights loaded successfully")
    
    def forward(self, x):
        # Encoder forward pass
        x1 = self.down_block1(x)
        x2 = self.down_block2(x1)
        x3 = self.down_block3(x2)
        x4 = self.down_block4(x3)
        x5 = self.down_block5(x4)
        
        # Global pooling and classification
        pooled = self.global_pool(x5)
        out = self.classifier(pooled)
        
        return out
    
    def get_features(self, x):
        """Extract features from encoder (for visualization/analysis)."""
        x1 = self.down_block1(x)
        x2 = self.down_block2(x1)
        x3 = self.down_block3(x2)
        x4 = self.down_block4(x3)
        x5 = self.down_block5(x4)
        pooled = self.global_pool(x5)
        return pooled.view(pooled.size(0), -1)


class EyeStateClassifierLite(nn.Module):
    """
    Lightweight Eye State Classifier.
    
    A simpler model with fewer parameters for faster training/inference.
    """
    
    def __init__(self, in_channels=1, num_classes=2, dropout=True, prob=0.3):
        super(EyeStateClassifierLite, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(prob) if dropout else nn.Identity(),
            
            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(prob) if dropout else nn.Identity(),
            
            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(prob) if dropout else nn.Identity(),
            
            # Block 4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(prob) if dropout else nn.Identity(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Model registry
model_dict = {
    'densenet': EyeStateClassifier,
    'lite': EyeStateClassifierLite
}


def get_model(model_name, **kwargs):
    """Get model by name."""
    if model_name not in model_dict:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_dict.keys())}")
    return model_dict[model_name](**kwargs)


def get_nparams(model):
    """Count number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    print("Testing EyeStateClassifier...")
    model = EyeStateClassifier(in_channels=1, num_classes=2)
    print(f"Parameters: {get_nparams(model):,}")
    
    # Test forward pass
    x = torch.randn(2, 1, 480, 640)  # Batch of 2 images
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # Test lite model
    print("\nTesting EyeStateClassifierLite...")
    model_lite = EyeStateClassifierLite(in_channels=1, num_classes=2)
    print(f"Parameters: {get_nparams(model_lite):,}")
    
    out_lite = model_lite(x)
    print(f"Output shape: {out_lite.shape}")
    
    # Test with pretrained weights loading
    import os
    pretrained_path = 'best_model.pkl'
    if os.path.exists(pretrained_path):
        print(f"\nTesting pretrained encoder loading from {pretrained_path}...")
        model_pretrained = EyeStateClassifier(
            in_channels=1, num_classes=2, pretrained_encoder=pretrained_path
        )



















