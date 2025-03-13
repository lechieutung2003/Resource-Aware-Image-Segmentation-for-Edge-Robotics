import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # Simple decoder for fast edge execution
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, num_classes, kernel_size=1)
        
    def forward(self, x, target_size):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

class MobileNetV3Student(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        
        # Load MobileNetV3 small backbone
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights).features
        
        # MobileNetV3-Small features output 576 channels at the end
        self.backbone = backbone
        self.head = SegmentationHead(in_channels=576, num_classes=num_classes)
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        # Extract features
        features = self.backbone(x)
        # Pass to head
        out = self.head(features, target_size=input_shape)
        # Return a dict to match teacher's DeepLabV3 output format
        return {'out': out}

def get_student_model(config):
    num_classes = config['dataset']['num_classes']
    pretrained = config['student'].get('pretrained', True)
    return MobileNetV3Student(num_classes=num_classes, pretrained=pretrained)

if __name__ == "__main__":
    config = {'dataset': {'num_classes': 21}, 'student': {'pretrained': False}}
    model = get_student_model(config)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)['out']
    print(f"Student Output Shape: {out.shape}")
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
