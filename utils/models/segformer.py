import torch
from torch import nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation

class SegFormer(nn.Module):
    def __init__(self, non_void_labels: list[str], device: str, checkpoint: str = 'nvidia/mit-b0'):
        super(SegFormer, self).__init__()
        self.__checkpoint = checkpoint
        self.__labels = non_void_labels
        self.__device = device

        self.__model = SegformerForSemanticSegmentation.from_pretrained(self.__checkpoint, 
                                                             num_labels=len(self.__labels), 
                                                             id2label={i: self.__labels[i] for i in range(len(self.__labels))}, 
                                                             label2id={self.__labels[i]: i for i in range(len(self.__labels))}).to(device)

    def forward(self, x):
        _, _, h, w = x.shape # First dimension is the batch size!
        x = self.__model(x).logits
        # Interpolate the logits to have a properly sized segmentation map (since output of transformer is (N_LABELS, H//4, W//4) for each image)
        x = nn.functional.interpolate(x, size=(h, w), mode='bilinear', align_corners=False) # Default is bilinear with align_corners=False
        return x