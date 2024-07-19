import torch
from torchvision import transforms as T
import torchvision.transforms.functional as F
import math

class Rotation:
    def __init__(self, angle: int, probability: float):
        self.angle = angle
        self.p = probability

    # Apply transformation to both input images and masks
    def __call__(self, input: torch.Tensor, target: torch.Tensor | None):
        # Random rotation
        if torch.rand(1) <= self.p:
        
            angle = T.RandomRotation.get_params([-self.angle, self.angle])
            input = F.rotate(input, angle)
            if target is not None:
                target = F.rotate(target, angle)
            
            # Random horizontal flip
            if torch.rand(1) < 0.5:
                input = F.hflip(input)
                if target is not None:
                    target = F.hflip(target)
            
            # Random vertical flip
            if torch.rand(1) < 0.5:
                input = F.vflip(input)
                if target is not None:
                    target = F.vflip(target)
            input_rot_cropped = crop_around_center(input, *largest_rotated_rect(400, 400, math.radians(angle)))
            target_rot_cropped = crop_around_center(target, *largest_rotated_rect(400, 400, math.radians(angle)))
            transform_resize = T.Resize((400, 400))
        
            return transform_resize(input_rot_cropped), transform_resize(target_rot_cropped)
        else:
            return input, target
        
        
#https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in radians),
    computes the width and height of the largest possible axis-aligned rectangle
    within the rotated rectangle.
    """
    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_h) if (w < h) else math.atan2(bb_h, bb_w)
    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )

def crop_around_center(image, width, height):
    """
    Given a PyTorch image tensor, crops it to the given width and height around its center point.
    """
    image_size = (image.shape[2], image.shape[1])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if width > image_size[0]:
        width = image_size[0]
    if height > image_size[1]:
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[:, y1:y2, x1:x2]