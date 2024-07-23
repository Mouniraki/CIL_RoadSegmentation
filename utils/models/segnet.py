from torch import nn
import torch
from utils.models.refinement_unet import UNet

class SegNet(nn.Module):
    def __init__(self, segformer, path_pretrained):
        super(SegNet, self).__init__()
        self.segformer = segformer
        self.unet = UNet.load_pretrained(UNet(), path_pretrained)

    def forward(self, x):
        x = self.segformer(x)
        x = torch.sigmoid(x)
        x = self.unet(x)
        return x
    
       # Functions to handle model checkpoints
    def save_pretrained(self, path: str):
        return torch.save(self.state_dict(), path)
    
    def load_pretrained(self, checkpoint: str):
        m = SegNet(channels=self.__enc_chs)
        m.load_state_dict(torch.load(checkpoint))
        return m
