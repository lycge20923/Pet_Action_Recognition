import torch
import torch.nn.functional as F


# Contrastive Loss implementation
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    def forward(self, f1, f2, y):
        dist = F.pairwise_distance(f1, f2)
        loss_pos = (1 - y) * dist.pow(2)
        loss_neg = y * F.relu(self.margin - dist).pow(2)
        return (loss_pos + loss_neg).mean()
