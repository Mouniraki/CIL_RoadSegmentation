import torch
from torch import nn

class TverskyFocalLoss(nn.Module):
    def __init__(self, model: str, alpha=0.7, gamma=1.35):
        super(TverskyFocalLoss, self).__init__()
        self.model = model
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        if self.model == 'segformer':
            output = torch.sigmoid(output)
        
        tp = (output * target).sum(dim=(-1, -2, -3), dtype=torch.float32)
        fn = ((1-output) * target).sum(dim=(-1, -2, -3), dtype=torch.float32)
        fp = (output * (1-target)).sum(dim=(-1, -2, -3), dtype=torch.float32)
        loss = 1 - (tp + 1 / (tp + self.alpha * fn + (1-self.alpha) * fp + 1)).mean()
        return loss
    