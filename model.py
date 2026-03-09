import torch
import torch.nn as nn
import torchvision.models as models

class FractalHybridCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(FractalHybridCNN, self).__init__()
        
        # Load pre-trained MobileNetV3 (Lightweight for Edge AI)
        self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        
        # Remove the final classification layer to get the raw features
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Identity()
        
        # Physics-Gate Integration: Add 1 extra feature for the Fractal Dimension
        self.physics_fusion = nn.Sequential(
            nn.Linear(in_features + 1, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, fractal_dim):
        # Extract image features
        img_features = self.backbone(image)
        
        # Ensure fractal_dim is the right shape [batch_size, 1]
        if len(fractal_dim.shape) == 1:
            fractal_dim = fractal_dim.unsqueeze(1)
            
        # Concatenate Deep Learning features with Physics features
        fused_features = torch.cat((img_features, fractal_dim), dim=1)
        
        # Final classification
        out = self.physics_fusion(fused_features)
        return out