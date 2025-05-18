import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

def get_teacher_model(config):
    """
    Returns a DeepLabV3 ResNet50 model as the high-capacity teacher.
    """
    num_classes = config['dataset']['num_classes']
    pretrained = config['teacher'].get('pretrained', True)
    
    if pretrained:
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        model = deeplabv3_resnet50(weights=weights)
        # Modify the classifier for our number of classes
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        # Modify the aux classifier if present
        if model.aux_classifier is not None:
            model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    else:
        model = deeplabv3_resnet50(weights=None, num_classes=num_classes)
        
    return model

if __name__ == "__main__":
    config = {'dataset': {'num_classes': 21}, 'teacher': {'pretrained': False}}
    model = get_teacher_model(config)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)['out']
    print(f"Teacher Output Shape: {out.shape}")
# Maintenance update

    # TODO: optimize this block
