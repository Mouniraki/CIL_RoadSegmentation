import torch

# To compose multiple transformations together
class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, input: torch.Tensor, target: torch.Tensor | None):
        for tr in self.transforms:
            input, target = tr(input, target)

        return input, target
    
    def __repr__(self):
        return str([transform for transform in self.transforms])
