import torch
import torch.nn as nn
import torch.nn.functional as F

class AFNOTransformerModel(nn.Module):
    def __init__(self, in_channels=1, embed_dim=32, depth=3, mlp_ratio=4, hidden_dim_afno=64):
       """
       #in_channels -  number of input channels
       #embed_dim -  dimension of the patch embedding/feature extraction
       #depth - number of AFNO transformer blocks stacked in the model
       #mlp_ratio - multiplier for the hidden dimension in the MLP layers inside each transformer block
       #hidden_dim_afno - number of hidden features in block
       """
        super().__init__()
        self.patch_embed = PatchEmbed(in_channels, embed_dim) #Patch embedding
        self.blocks = nn.ModuleList([
            AFNOTransformerBlock(embed_dim, mlp_ratio, hidden_dim_afno) #AFNO transformer blocks
            for _ in range(depth)
        ])
        self.reconstruction_head = nn.Conv2d(embed_dim, 1, kernel_size=1) #Image reconstruction

    def forward(self, x):
        x = self.patch_embed(x) #Patch embedding
        for blk in self.blocks: #Passes data through each AFNO transformer block sequentially
            x = blk(x)
        x = self.reconstruction_head(x) #Reconstructs denoised image
        return x

#-----------------------------Patch embedding-----------------------------------
#Uses a convolution to embed local regions of the image into a higher-dimensional space.
class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)

    def forward(self, x): #input - [B, in_channels, H, W]
        return self.proj(x) #output - [B, embed_dim, H, W]

#------------------------AFNO block---------------------------------------------
#Processes image/feature map in the Fourier space, then returns the result to the real space
class AFNOBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(AFNOBlock, self).__init__()
        #Two fully connected layers for real and imaginary components
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        B, C, H, W = x.shape

        # Fourier Transform
        x_fft = torch.fft.fft2(x, norm='ortho')

        # Stack real and imaginary components
        freq = torch.stack([x_fft.real, x_fft.imag], dim=-1)
        freq = freq.view(B * C * H * W, 2)

        # Tiny MLP in frequency domain - serves as a non-linear Fourier filter
        freq = F.gelu(self.fc1(freq))
        freq = self.fc2(freq) # FREQUENCY DOMAIN MIXING

        # Reshape back
        freq = freq.view(B, C, H, W, 2)
        real, imag = freq[..., 0], freq[..., 1]
        x_fft_filtered = torch.complex(real, imag)

        # Inverse Fourier Transform
        x_out = torch.fft.ifft2(x_fft_filtered, norm='ortho').real

        return x_out

#---------------AFNO Transformer------------------------------------------------
#Transformer-like block, uses frequency mixing similar to self-attention
class AFNOTransformerBlock(nn.Module):
    def __init__(self, dim, mlp_ratio, hidden_dim_afno):
        super().__init__()
        self.afno = AFNOBlock(dim, hidden_dim=hidden_dim_afno)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x):
        B, C, H, W = x.shape

        # Apply AFNO
        x = x + self.afno(x) #applies AFNO to the input, and adds “remixed” output back to the original input

        # MLP
        x_ = self.mlp(x.view(B, H * W, C))
        x_ = x_.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # Residual connection
        x = x + x_

        return x

class MLP(nn.Module): #Small MLP - non -linear filter
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim) #increases the feature dimension from dim to hidden_dim
        self.fc2 = nn.Linear(hidden_dim, dim) #contracts features from hidden_dim to dim

    def forward(self, x): #expansion → non-linear activation → contraction
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x
