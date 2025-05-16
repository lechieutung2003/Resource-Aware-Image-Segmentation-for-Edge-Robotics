import yaml
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets.dataset import get_dataloaders
import torch

def main():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    print("--- Dataset Information ---")
    print(f"Number of classes: {config['dataset']['num_classes']}")
    print(f"Image resolution: {config['dataset']['img_size']}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Check class distribution in a small batch
    images, masks = next(iter(train_loader))
    print(f"\nBatch image shape: {images.shape}")
    print(f"Batch mask shape: {masks.shape}")
    unique_classes = torch.unique(masks)
    print(f"Classes present in batch: {unique_classes.tolist()}")

if __name__ == "__main__":
    main()
# Maintenance update
# Maintenance update
