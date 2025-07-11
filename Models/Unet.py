import torch
import torch.nn as nn
import torch.nn.functional as F

#Convolutional block
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GELU()
        )
    def forward(self, x):
        return self.conv(x)

#Encoder block
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        skip = self.conv(x)
        x = self.pool(skip)
        return x, skip

#Decoder block
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(2*out_ch, out_ch)
    def forward(self, x1, x2, skip = True): # x1: upsampled, x2: skip
        x1 = self.up(x1)
        
        # Pad x1 if needed (for odd input sizes)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

#U-net
class UNet(nn.Module):
    def __init__(self,in_channels=1,base_ch=32):
        super().__init__()
        # Encoder
        self.down1 = EncoderBlock(in_channels, base_ch) 
        self.down2 = EncoderBlock(base_ch, base_ch * 2) 
        self.down3 = EncoderBlock(base_ch * 2, base_ch * 4) 

        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)

        # Decoder
        self.up3 = DecoderBlock(base_ch * 8, base_ch * 4)            
        self.up2 = DecoderBlock(base_ch * 4, base_ch * 2)            
        self.up1 = DecoderBlock(base_ch * 2, base_ch)                

        self.out_conv = nn.Conv2d(base_ch, in_channels,  1)

    def forward(self, x):
        # Encoder
        x1, skip1 = self.down1(x)
        x2, skip2 = self.down2(x1)
        x3, skip3 = self.down3(x2)     
        
        b = self.bottleneck(x3)
        
        y3 = self.up3(b, skip3)
        y2 = self.up2(y3, skip2)
        y1 = self.up1(y2, skip1)

        y = self.out_conv(y1)

        return y
