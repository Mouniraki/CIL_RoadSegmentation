from torch import nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation

class SegFormer(nn.Module):
    def __init__(self, non_void_labels: list[str], checkpoint: str = 'nvidia/mit-b0'):
        super(SegFormer, self).__init__()
        self.__checkpoint = checkpoint
        self.__labels = non_void_labels

        # Initializes the segformer-b5 pre-trained model parameters
        config = SegformerConfig(
            attention_probs_dropout_prob = 0.0,
            classifier_dropout_prob = 0.1,
            decoder_hidden_size = 768,
            depths = [3, 6, 40, 3],
            downsampling_rates = [1, 4, 8, 16],
            drop_path_rate = 0.1,
            hidden_act = "gelu",
            hidden_dropout_prob = 0.0,
            hidden_sizes = [64, 128, 320, 512],
            image_size = 224,
            initializer_range = 0.02,
            layer_norm_eps = 1e-06,
            mlp_ratios = [4, 4, 4, 4],
            num_attention_heads = [1, 2, 5, 8],
            num_channels = 3,
            num_encoder_blocks = 4,
            patch_sizes = [7, 3, 3, 3],
            semantic_loss_ignore_index = 255,
            sr_ratios = [8, 4, 2, 1],
            strides = [4, 2, 2, 2],
            num_labels=len(self.__labels),
            id2label={i: self.__labels[i] for i in range(len(self.__labels))}, 
            label2id={self.__labels[i]: i for i in range(len(self.__labels))}
        )

        self.__model = SegformerForSemanticSegmentation.from_pretrained(self.__checkpoint,
                                                                        config=config)

    def forward(self, x):
        _, _, h, w = x.shape # First dimension is the batch size!
        x = self.__model(x).logits
        # Interpolate the logits to have a properly sized segmentation map (since output of transformer is (N_LABELS, H//4, W//4) for each image)
        x = nn.functional.interpolate(x, size=(h, w), mode='bilinear', align_corners=False) # Default is bilinear with align_corners=False
        return x
    
    # Functions to handle model checkpoints
    def save_pretrained(self, path: str):
        return self.__model.save_pretrained(path)
    
    def load_pretrained(self, checkpoint: str):
        return SegFormer(non_void_labels=self.__labels, checkpoint=checkpoint)