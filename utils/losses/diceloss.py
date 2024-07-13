import torch
from torch import nn

class DiceLoss(nn.Module):
    def __init__(self, model: str):
        super(DiceLoss, self).__init__()
        self.model = model

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        if self.model == 'segformer':
            output = torch.sigmoid(output)
        
        pixels_intersection = (output * target).sum(dim=(-1, -2, -3), dtype=torch.float32)
        pixels_union = output.sum(dim=(-1, -2, -3), dtype=torch.float32) + target.sum(dim=(-1, -2, -3), dtype=torch.float32)

        #1 is smooth factor
        loss = 1 - ((2 * pixels_intersection + 1) / (pixels_union + 1)).mean()
        return loss
    