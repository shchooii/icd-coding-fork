import torch
import torch.nn as nn
import torch.nn.functional as F
from src.losses.utils import shot_mAP

class HeadTailBalancerLoss(nn.Module):
    def __init__(self, gamma=2, PFM=None):
        super(HeadTailBalancerLoss, self).__init__()
        self.gamma = gamma
        self.PFM = PFM
        self.eps = 1e-8

    def forward(self, head, tail, balance, labels):
        with torch.no_grad():
            h_acc = self.PFM(head, labels).pow(self.gamma)
            t_acc = self.PFM(tail, labels).pow(self.gamma)
            denom = h_acc + t_acc + self.eps
            k_h, k_t = h_acc / denom, t_acc / denom
            
        p_h = F.softmax(head, dim=-1)
        p_t = F.softmax(tail, dim=-1)
        p_b = F.softmax(balance, dim=-1)

        loss_h = self.PFM(p_h * p_b, labels)            
        loss_t = self.PFM(p_t * p_b, labels)

        loss = (k_h * loss_h + k_t * loss_t).mean()
        return loss