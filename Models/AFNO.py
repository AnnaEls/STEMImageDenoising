import torch
import torch.nn as nn
import torch.nn.functional as F

class AFNOBlock(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super(AFNOBlock, self).__init__()
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        B, C, H, W = x.shape

        # Fourier Transform
        x_fft = torch.fft.fft2(x, norm='ortho')

        # Stack real and imag parts
        freq = torch.stack([x_fft.real, x_fft.imag], dim=-1)
        freq = freq.view(B * C * H * W, 2)

        # Tiny MLP in frequency domain
        freq = F.gelu(self.fc1(freq))
        freq = self.fc2(freq)

        # Reshape back
        freq = freq.view(B, C, H, W, 2)
        real, imag = freq[..., 0], freq[..., 1]
        x_fft_filtered = torch.complex(real, imag)

        # Inverse Fourier Transform
        x_out = torch.fft.ifft2(x_fft_filtered, norm='ortho').real

        return x_out

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

class AFNOTransformerBlock(nn.Module):
    def __init__(self, dim, mlp_ratio, hidden_dim_afno):
        super().__init__()
        self.afno = AFNOBlock(dim, hidden_dim=hidden_dim_afno)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Apply AFNO
        x = self.afno(x) 
        
        # MLP
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]   
        x = self.mlp(x.view(B, H * W, C))
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        return x

# --- Patch Embedding ---
class PatchEmbed(nn.Module):
    def __init__(self, in_channels=1, embed_dim=32):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)

    def forward(self, x):
        return self.proj(x)

# --- Full Model ---
class AFNOTransformerModel(nn.Module):
    def __init__(self, in_channels=1, embed_dim=32, depth=3, mlp_ratio=6, hidden_dim_afno=64):
        super().__init__()
        self.patch_embed = PatchEmbed(in_channels, embed_dim)
        self.blocks = nn.ModuleList([
            AFNOTransformerBlock(embed_dim, mlp_ratio, hidden_dim_afno)
            for _ in range(depth)
        ])
        self.reconstruction_head = nn.Conv2d(embed_dim, 1, kernel_size=1)

    def forward(self, x):
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.reconstruction_head(x)
        return x
