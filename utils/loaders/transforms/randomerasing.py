import torch
from torchvision.transforms import v2

# Apply random erasing on data only, not the mask
class RandomErasing:
    def __init__(self, probability : float, scale : float, ratio : float):
        self.p = probability
        self.scale = scale
        self.ratio = ratio
        
    def __call__(self, input: torch.Tensor, target: torch.Tensor | None):
        transform = v2.Compose([
            v2.RandomErasing(p=self.p, scale=self.scale, ratio= self.ratio)
        ])

        input = transform(input)
        return input, target