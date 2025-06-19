import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image

class SegmentationDataset(Dataset):
    def __init__(self, root_dir=None, split='train', img_size=(224, 224), num_classes=21, synthetic=True, num_samples=100):
        """
        Configurable dataset interface.
        If synthetic=True, generates random shapes to test the pipeline.
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.num_classes = num_classes
        self.synthetic = synthetic
        self.num_samples = num_samples
        
        # If not synthetic, we'd load file paths here.
        if not self.synthetic:
            self.image_paths = []
            self.mask_paths = []
            # ... load paths ...
            pass
            
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        if self.synthetic:
            # Generate random synthetic image
            image = np.random.randint(0, 255, (self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
            # Generate random synthetic mask (classes 0 to num_classes-1)
            mask = np.random.randint(0, self.num_classes, (self.img_size[0], self.img_size[1]), dtype=np.uint8)
            
            # Optionally add a coherent shape to make it learnable
            cv2.circle(image, (self.img_size[1]//2, self.img_size[0]//2), self.img_size[0]//4, (255, 0, 0), -1)
            cv2.circle(mask, (self.img_size[1]//2, self.img_size[0]//2), self.img_size[0]//4, 1, -1)
            
            image = Image.fromarray(image)
            mask = Image.fromarray(mask.astype(np.uint8))
        else:
            # image = Image.open(self.image_paths[idx]).convert('RGB')
            # mask = Image.open(self.mask_paths[idx])
            pass
            
        # Basic Augmentations / Resizing
        image = image.resize(self.img_size, Image.BILINEAR)
        mask = mask.resize(self.img_size, Image.NEAREST)
        
        image = TF.to_tensor(image)
        # Normalize with ImageNet stats
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        
        return image, mask

def get_dataloaders(config):
    img_size = tuple(config['dataset']['img_size'])
    num_classes = config['dataset']['num_classes']
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset'].get('num_workers', 4)
    
    train_dataset = SegmentationDataset(split='train', img_size=img_size, num_classes=num_classes, synthetic=True, num_samples=200)
    val_dataset = SegmentationDataset(split='val', img_size=img_size, num_classes=num_classes, synthetic=True, num_samples=50)
    test_dataset = SegmentationDataset(split='test', img_size=img_size, num_classes=num_classes, synthetic=True, num_samples=50)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader
# Maintenance update
# Maintenance update
# Maintenance update
