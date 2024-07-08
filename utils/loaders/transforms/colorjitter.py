import torch
from torchvision.transforms import v2

# Apply color transform only on the input image, not the target mask!
class ColorJitter:
    def __init__(self, brightness: float, contrast: float, saturation: float, hue: float):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, input: torch.Tensor, target: torch.Tensor | None):
        colortransform = v2.Compose([
            v2.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation, hue=self.hue)
        ])

        input = colortransform(input)
        return input, target