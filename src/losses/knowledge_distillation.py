import torch
import torch.nn as nn
import torch.nn.functional as F

class KDLoss(nn.Module):
    def __init__(self, T=2.0):
        super().__init__()
        self.T = T
    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        return F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
