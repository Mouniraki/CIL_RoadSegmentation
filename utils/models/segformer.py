from torch import nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation

class SegFormer(nn.Module):
    def __init__(self):
        super(SegFormer, self).__init__()

        self.config = SegformerConfig(
            num_channels=3,
            num_encoder_blocks=4,
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            hidden_sizes=[32, 64, 160, 256],
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            num_attention_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            classifier_dropout_prob=0.1,
            initializer_range=0.02,
            drop_path_rate=0.1,
            layer_norm_eps=1e-6,
            decoder_hidden_size=256,
            semantic_loss_ignore_index=255
        )

    def forward(self, x):
        return SegformerForSemanticSegmentation(config=self.config)(x)