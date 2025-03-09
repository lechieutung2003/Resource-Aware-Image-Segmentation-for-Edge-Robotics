import yaml
import sys
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets.dataset import get_dataloaders
from src.models.teacher.model import get_teacher_model
from src.models.student.model import get_student_model

def calculate_iou(preds, labels, num_classes):
    # preds: [B, H, W], labels: [B, H, W]
    ious = []
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    return ious

def evaluate(model, dataloader, device, num_classes):
    model.eval()
    total_ious = []
    pixel_acc = 0.0
    total_pixels = 0
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluation"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1)
            
            # Pixel accuracy
            pixel_acc += (preds == masks).sum().item()
            total_pixels += torch.numel(masks)
            
            # IoU
            batch_ious = calculate_iou(preds, masks, num_classes)
            total_ious.append(batch_ious)
            
    # Compute mean metrics
    pixel_acc = pixel_acc / total_pixels
    
    # Calculate mean IoU ignoring NaNs
    valid_ious = []
    for cls in range(num_classes):
        cls_ious = [batch_iou[cls] for batch_iou in total_ious if not torch.isnan(torch.tensor(batch_iou[cls]))]
        if cls_ious:
            valid_ious.append(sum(cls_ious) / len(cls_ious))
            
    mIoU = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
    
    return pixel_acc, mIoU

def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation models")
    parser.add_argument('--model', type=str, choices=['teacher', 'student_baseline', 'student_kd'], required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    _, _, test_loader = get_dataloaders(config)
    num_classes = config['dataset']['num_classes']
    
    if args.model == 'teacher':
        model = get_teacher_model(config)
    else:
        model = get_student_model(config)
        
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print(f"Checkpoint not found at {args.checkpoint}. Exiting.")
        return
        
    model.to(device)
    
    pixel_acc, mIoU = evaluate(model, test_loader, device, num_classes)
    
    print(f"\n--- Evaluation Results for {args.model} ---")
    print(f"Pixel Accuracy: {pixel_acc:.4f}")
    print(f"mIoU: {mIoU:.4f}")

if __name__ == "__main__":
    main()
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
