import torch
from torchvision.transforms import v2, functional

class RandomResizedCrop:
    def __init__(self, scale: tuple[int, int], ratio: tuple[int, int]):
        self.scale = scale
        self.ratio = ratio

    # Apply transformation to both input images and masks
    def __call__(self, input: torch.Tensor, target: torch.Tensor | None):
        _, out_h, out_w = input.shape
        transform_params = v2.RandomResizedCrop.get_params(img=input, scale=self.scale, ratio=self.ratio)
        input = functional.resized_crop(input, *transform_params, size=(out_h, out_w))

        if target is not None:
            target = functional.resized_crop(target, *transform_params, size=(out_h, out_w))

        return input, target
