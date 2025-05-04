import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5):
        """
        temperature: Softmax temperature for distillation
        alpha: Weight for the distillation loss (1-alpha is for standard CE)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, student_logits, teacher_logits, targets):
        """
        student_logits: [B, C, H, W]
        teacher_logits: [B, C, H, W]
        targets: [B, H, W] (Class indices)
        """
        # Standard Cross Entropy Loss
        loss_ce = self.ce_loss(student_logits, targets)
        
        # Knowledge Distillation Loss (KL Divergence)
        # Soften probabilities
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # Calculate KL Div
        loss_kd = F.kl_div(
            student_log_probs, 
            teacher_probs, 
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Total loss
        total_loss = (1 - self.alpha) * loss_ce + self.alpha * loss_kd
        
        return total_loss, loss_ce, loss_kd

def get_distillation_loss(config):
    temperature = config['distillation'].get('temperature', 4.0)
    alpha = config['distillation'].get('alpha', 0.5)
    return DistillationLoss(temperature=temperature, alpha=alpha)
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
