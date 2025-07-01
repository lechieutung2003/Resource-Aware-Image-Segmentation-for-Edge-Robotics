import yaml
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets.dataset import get_dataloaders
from src.models.teacher.model import get_teacher_model
from src.models.student.model import get_student_model
from src.distillation.loss import get_distillation_loss

def train_one_epoch(student, teacher, dataloader, criterion, optimizer, device):
    student.train()
    teacher.eval() # Teacher is always in eval mode
    
    running_loss = 0.0
    running_ce = 0.0
    running_kd = 0.0
    
    for images, masks in tqdm(dataloader, desc="Distillation Training"):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        with torch.no_grad():
            teacher_outputs = teacher(images)['out']
            
        student_outputs = student(images)['out']
        
        loss, ce_loss, kd_loss = criterion(student_outputs, teacher_outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        running_ce += ce_loss.item() * images.size(0)
        running_kd += kd_loss.item() * images.size(0)
        
    n = len(dataloader.dataset)
    return running_loss / n, running_ce / n, running_kd / n

def validate(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            
            running_loss += loss.item() * images.size(0)
            
    return running_loss / len(dataloader.dataset)

def main():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader, val_loader, _ = get_dataloaders(config)
    
    # Initialize models
    teacher = get_teacher_model(config).to(device)
    student = get_student_model(config).to(device)
    
    # Load teacher weights
    teacher_path = 'models/teacher_best.pth'
    if os.path.exists(teacher_path):
        teacher.load_state_dict(torch.load(teacher_path, map_location=device))
        print("Loaded teacher weights successfully.")
    else:
        print(f"Warning: Teacher weights not found at {teacher_path}. Using uninitialized teacher for demonstration.")
        
    criterion = get_distillation_loss(config)
    optimizer = optim.SGD(
        student.parameters(), 
        lr=config['training']['learning_rate'], 
        momentum=config['training']['momentum'], 
        weight_decay=config['training']['weight_decay']
    )
    
    num_epochs = 1 # Quick demonstration
    
    best_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss, train_ce, train_kd = train_one_epoch(student, teacher, train_loader, criterion, optimizer, device)
        val_loss = validate(student, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f} (CE: {train_ce:.4f}, KD: {train_kd:.4f})")
        print(f"Val Loss: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(student.state_dict(), 'models/student_best.pth')
            print("Saved best distilled student model!")

if __name__ == "__main__":
    main()
# Maintenance update
# Maintenance update
# Maintenance update

    # TODO: optimize this block

    # Edge case handled successfully
