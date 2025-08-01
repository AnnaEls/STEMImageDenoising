import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# AFNO Block (Frequency Mixing)
# -------------------------------
class AFNOBlock(nn.Module):
    """
    Efficient AFNO Block:
    - Processes [B, C, H, W] feature maps
    - Frequency mixing in groups
    - Real+Imag concatenation for learnable filtering
    """
    def __init__(self, dim, hidden_dim, num_groups=8):
        super(AFNOBlock, self).__init__()
        self.num_groups = num_groups
        self.group_channels = dim // num_groups
        
        # Frequency domain linear mixing (real+imag)
        self.fc1 = nn.Linear(self.group_channels * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.group_channels * 2)
        self.act = nn.GELU()

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        assert C % self.num_groups == 0, "C must be divisible by num_groups"

        # 1) Forward FFT
        x_fft = torch.fft.fft2(x, norm='ortho')  # [B, C, H, W]

        # 2) Group channels for efficient processing
        x_fft = x_fft.view(B, self.num_groups, self.group_channels, H, W)

        # 3) Real+Imag concat
        freq = torch.cat([x_fft.real, x_fft.imag], dim=2)  # [B, G, 2*Gc, H, W]
        freq = freq.permute(0, 1, 3, 4, 2).contiguous()    # [B, G, H, W, 2*Gc]

        # 4) Frequency MLP
        freq = self.fc1(freq)
        freq = self.act(freq)
        freq = self.fc2(freq)

        # 5) Convert back to complex
        freq = freq.permute(0, 1, 4, 2, 3)                # [B, G, 2*Gc, H, W]
        real, imag = torch.chunk(freq, 2, dim=2)
        x_fft_filtered = torch.complex(real, imag).view(B, C, H, W)

        # 6) Inverse FFT to spatial domain
        x_out = torch.fft.ifft2(x_fft_filtered, norm='ortho').real
        return x_out

# -------------------------------
# Transformer Block
# -------------------------------
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class AFNOTransformerBlock(nn.Module):
    def __init__(self, dim, mlp_ratio, hidden_dim_afno, norm=True, skip_one=True, skip_two=True, num_groups=8):
        super().__init__()
        self.afno = AFNOBlock(dim, hidden_dim=hidden_dim_afno, num_groups=num_groups)
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm = norm

        self.skip_one = skip_one
        self.skip_two = skip_two

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape

        # --- AFNO Residual ---
        skip_1 = x if self.skip_one else None
        if self.norm:
            x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.afno(x)
        if self.skip_one:
            x = x + skip_1

        # --- MLP Residual ---
        x_perm = x.permute(0, 2, 3, 1)           # [B,H,W,C]
        skip_2 = x_perm if self.skip_two else None
        if self.norm:
            x_perm = self.norm2(x_perm)

        x_perm = self.mlp(x_perm.reshape(B, H * W, C))
        x_perm = x_perm.view(B, H, W, C)
        if self.skip_two:
            x_perm = x_perm + skip_2

        return x_perm.permute(0, 3, 1, 2).contiguous()

# -------------------------------
# Patch Embedding
# -------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)

    def forward(self, x):
        return self.proj(x)

# -------------------------------
# Full AFNO Transformer Model
# -------------------------------
class AFNOTransformerModel(nn.Module):
    def __init__(self, in_channels=1, embed_dim=32, depth=3, mlp_ratio=6,
                 hidden_dim_afno=64, num_groups=8, norm=True,
                 skip_one=True, skip_two=True):
        super().__init__()
        self.patch_embed = PatchEmbed(in_channels, embed_dim)
        self.blocks = nn.ModuleList([
            AFNOTransformerBlock(embed_dim, mlp_ratio, hidden_dim_afno,
                                 norm=norm, skip_one=skip_one, skip_two=skip_two,
                                 num_groups=num_groups)
            for _ in range(depth)
        ])
        self.reconstruction_head = nn.Conv2d(embed_dim, in_channels, kernel_size=1)

    def forward(self, x):
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.reconstruction_head(x)
        return x
