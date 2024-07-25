import torch
from torch import nn


DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


class DiceLoss(nn.Module):
    def __init__(self, model: str):
        super(DiceLoss, self).__init__()
        self.model = model

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        if self.model == 'segformer':
            output = torch.sigmoid(output)
        
        pixels_intersection = (output * target).sum(dim=(-1, -2, -3), dtype=torch.float32)
        pixels_union = output.sum(dim=(-1, -2, -3), dtype=torch.float32) + target.sum(dim=(-1, -2, -3), dtype=torch.float32)

        loss = 1 - (2 * pixels_intersection / pixels_union).mean()
        return loss


class WeightedDiceLoss(nn.Module):
    def __init__(self, model: str, fn_weight: float = 1.0, fp_weight: float = 0.3):
        super(WeightedDiceLoss, self).__init__()
        self.model = model
        self.fn_weight = fn_weight  # Weight for false negatives
        self.fp_weight = fp_weight  # Weight for false positives

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        if self.model == 'segformer':
            output = torch.sigmoid(output)

        smooth = 1e-6  # Small constant to avoid division by zero

        intersection = (output * target).sum(dim=(-1, -2, -3), dtype=torch.float32)
        false_negative = (target * (1 - output)).sum(dim=(-1, -2, -3), dtype=torch.float32)
        false_positive = ((1 - target) * output).sum(dim=(-1, -2, -3), dtype=torch.float32)

        weighted_union = intersection + self.fn_weight * false_negative + self.fp_weight * false_positive

        dice_score = (2 * intersection + smooth) / (weighted_union + smooth)
        loss = 1 - dice_score.mean()
        return loss


class BorderDiceLoss(nn.Module):
    def __init__(self, model: str, device='cpu'):
        super(BorderDiceLoss, self).__init__()
        self.model = model
        self.device = device

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        if self.model == 'segformer':
            output = torch.sigmoid(output)

        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=self.device).unsqueeze(0).unsqueeze(0).to(DEVICE)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=self.device).unsqueeze(0).unsqueeze(0).to(DEVICE)

        edge_x = torch.nn.functional.conv2d(output, sobel_x, padding=0)
        edge_y = torch.nn.functional.conv2d(output, sobel_y, padding=0)
        magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        magnitude = (magnitude - torch.min(magnitude)) / (torch.max(magnitude) - torch.min(magnitude))
        edge_output = magnitude

        edge_x = torch.nn.functional.conv2d(target, sobel_x, padding=0)
        edge_y = torch.nn.functional.conv2d(target, sobel_y, padding=0)
        magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        magnitude = (magnitude - torch.min(magnitude)) / (torch.max(magnitude) - torch.min(magnitude))
        edge_target = (magnitude > 0.5).float()

        pixels_intersection = (edge_output * edge_target).sum(dim=(-1, -2, -3), dtype=torch.float32)
        pixels_union = edge_output.sum(dim=(-1, -2, -3), dtype=torch.float32) + edge_target.sum(dim=(-1, -2, -3), dtype=torch.float32)

        loss = 1 - (2 * pixels_intersection / pixels_union).mean()
        return loss



class CombineLoss(nn.Module):
    def __init__(self, model: str, device='cpu'):
        super(CombineLoss, self).__init__()
        self.model = model
        self.device = device
        self.border_dict_loss = BorderDiceLoss(model=model)
        self.dice_loss = DiceLoss(model=model)

    def forward(self, output: torch.Tensor, target: torch.Tensor):

        return (1-self.dice_loss(output, target))*0.1 + (1-self.border_dict_loss(output, target))*0.9