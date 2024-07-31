import torch
from torch import nn

def dice_loss(model: str, output: torch.Tensor, target: torch.Tensor):
    if model == 'segformer':
        output = torch.sigmoid(output)

    pixels_intersection = (output * target).sum(dim=(-1, -2, -3), dtype=torch.float32)
    pixels_union = output.sum(dim=(-1, -2, -3), dtype=torch.float32) + target.sum(dim=(-1, -2, -3), dtype=torch.float32)

    loss = 1 - (2 * pixels_intersection / pixels_union).mean()
    return loss


def weighted_dice_loss(model: str, output: torch.Tensor, target: torch.Tensor, fn_weight: float = 1.0, fp_weight: float = 0.3):
    if model == 'segformer':
        output = torch.sigmoid(output)

    smooth = 1e-6  # Small constant to avoid division by zero

    intersection = (output * target).sum(dim=(-1, -2, -3), dtype=torch.float32)
    false_negative = (target * (1 - output)).sum(dim=(-1, -2, -3), dtype=torch.float32)
    false_positive = ((1 - target) * output).sum(dim=(-1, -2, -3), dtype=torch.float32)

    weighted_union = intersection + fn_weight * false_negative + fp_weight * false_positive

    dice_score = (2 * intersection + smooth) / (weighted_union + smooth)
    loss = 1 - dice_score.mean()
    return loss




def border_dice_loss(model: str, output: torch.Tensor, target: torch.Tensor, device: str):
    if model == 'segformer':
        output = torch.sigmoid(output)

    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device).unsqueeze(0).unsqueeze(
        0).to(device)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device).unsqueeze(0).unsqueeze(
        0).to(device)

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
    pixels_union = edge_output.sum(dim=(-1, -2, -3), dtype=torch.float32) + edge_target.sum(dim=(-1, -2, -3),
                                                                                            dtype=torch.float32)

    loss = 1 - (2 * pixels_intersection / pixels_union).mean()
    return loss


class DiceLoss(nn.Module):
    def __init__(self, model: str):
        super(DiceLoss, self).__init__()
        self.model = model

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        return dice_loss(self.model, output, target)

class WeightedDiceLoss(nn.Module):
    def __init__(self, model: str, fn_weight: float = 1.0, fp_weight: float = 0.3):
        super(WeightedDiceLoss, self).__init__()
        self.model = model
        self.fn_weight = fn_weight  # Weight for false negatives
        self.fp_weight = fp_weight  # Weight for false positives

    def forward(self, output: torch.Tensor, target: torch.Tensor):

        return weighted_dice_loss(self.model, output, target, fn_weight = self.fn_weight, fp_weight = self.fp_weight)


class BorderDiceLoss(nn.Module):
    def __init__(self, model: str, device='cpu'):
        super(BorderDiceLoss, self).__init__()
        self.model = model
        self.device = device

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        return border_dice_loss(self.model, output, target, self.device)



class CombineLoss(nn.Module):
    def __init__(self, model: str, device: str):
        super(CombineLoss, self).__init__()
        self.model = model
        self.device = device

    def forward(self, output: torch.Tensor, target: torch.Tensor):

        diceLoss = dice_loss(self.model, output, target)
        borderDiceLoss = border_dice_loss(self.model, output, target, self.device)

        return diceLoss/2 + borderDiceLoss/2 # by summing we increase border influence on loss
