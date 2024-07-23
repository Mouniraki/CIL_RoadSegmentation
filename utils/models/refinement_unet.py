import torch
from torch import nn

# Implements a block of convolution (i.e. 2*(3x3 convolution + ReLU))
class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

# Implements UNet (following the scheme described in the paper)
class UNet(nn.Module):
    def __init__(self, channels=(1,64,128,256,512,1024)):
        super(UNet, self).__init__()
        self.__enc_chs = channels
        dec_chs = channels[::-1][:-1] # Leave out the last value of the channels after flipping the list, since it is the number of channels of the final output

        # Encoder blocks (using the Block class)
        self.enc_blocks = nn.ModuleList([
            Block(in_ch, out_ch) for in_ch, out_ch in zip(self.__enc_chs[:-1], self.__enc_chs[1:])
        ])
        # 2x2 max pooling (down-sampling the data)
        self.pool = nn.MaxPool2d(kernel_size=2)

        # For the up-convolutions (i.e. fractionally strided convolutions, upsampling the data)
        self.up_convs = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=2, stride=2) for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])
        ])
        # Decoder blocks (using the Block class)
        self.dec_blocks = nn.ModuleList([
            Block(in_ch, out_ch) for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])
        ])
        # 1x1 convolution to produce the output (using the last value of the flipped channel list as input dimension)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=dec_chs[-1], out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        enc_features = [] # To be able to implement the skip connections at decoding

        # Encoder part
        for block in self.enc_blocks[:-1]:
            x = block(x)
            enc_features.append(x) # Keep track of the encoded features at each step for the skip connections
            x = self.pool(x) # Decrease resolution
        x = self.enc_blocks[-1](x) # Latent space encoding

        # Decoder part
        for block, upconv, feature in zip(self.dec_blocks, self.up_convs, enc_features[::-1]):
            x = upconv(x) # Increase resolution
            x = torch.cat([x, feature], dim=1) # Concatenate skip features
            x = block(x)

        return self.head(x) # Reduce to 1 channel
    
    # Functions to handle model checkpoints
    def save_pretrained(self, path: str):
        return torch.save(self.state_dict(), path)
    
    def load_pretrained(self, checkpoint: str):
        m = UNet(channels=self.__enc_chs)
        m.load_state_dict(torch.load(checkpoint))
        return m