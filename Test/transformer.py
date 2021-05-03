import torch
import torch.nn as nn

class TransformerNetwork(nn.Module):
    def __init__(self):
        super(TransformerNetwork,self).__init__()
        self.ConvBlock1 = nn.Sequential(
            ConvLayer(3,32,9,1),
            nn.ReLU(),
            ConvLayer(32,64,3,2),
            nn.ReLU(),
            ConvLayer(64,128,3,2),
            nn.ReLU()
        )
        self.ResidualBlock = nn.Sequential(
            ResidualLayer(128,3),
            ResidualLayer(128,3),
            ResidualLayer(128,3),
            ResidualLayer(128,3),
            ResidualLayer(128,3)
        )
        self.ConvBlock2 = nn.Sequential(
            DeconvLayer(128,64,3,2,1),
            nn.ReLU(),
            DeconvLayer(64,32,3,2,1),
            nn.ReLU(),
            ConvLayer(32,3,9,1,norm="None")
        )

    def forward(self,x):
        x = self.ConvBlock1(x)
        x = self.ResidualBlock(x)
        out = self.ConvBlock2(x)
        return out

class ConvLayer(nn.Module):
    def __init__(self,inC,outC,kernal_size,stride,norm="instance"):
        super(ConvLayer,self).__init__()
        # Padding layers
        padding_size = kernal_size//2
        self.pad = nn.ReflectionPad2d(padding_size)
        
        # Convolution layers
        self.conv_layer = nn.Conv2d(inC,outC,kernal_size,stride)

        #Normalization layers
        self.norm_type = norm
        if (norm=="instance"):
            self.norm_layer = nn.InstanceNorm2d(outC,affine=True)
        elif (norm=="batch"):
            self.norm_layer = nn.BatchNorm2d(outC,affine=True)

    def forward(self,x):
        x = self.pad(x)
        x = self.conv_layer(x)
        out = x if self.norm_type=="None" else self.norm_layer(x)
        return out

class ResidualLayer(nn.Module):
    def __init__(self,channels=128,kernel_size=3):
        super(ResidualLayer,self).__init__()
        self.conv1 = ConvLayer(channels,channels,kernel_size,stride=1)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels,channels,kernel_size,stride=1)
    
    def forward(self,x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + identity
        return out

class DeconvLayer(nn.Module):
    def __init__(self,inC,outC,kernal_size,stride,out_padding,norm="instance"):
        super(DeconvLayer,self).__init__()
        # Padding layers
        padding_size = kernal_size//2
        self.convTran = nn.ConvTranspose2d(inC,outC,kernal_size,stride,padding_size,out_padding)

        #Normalization layers
        self.norm_type = norm
        if (norm=="instance"):
            self.norm_layer = nn.InstanceNorm2d(outC,affine=True)
        elif (norm=="batch"):
            self.norm_layer = nn.BatchNorm2d(outC,affine=True)

    def forward(self,x):
        x = self.convTran(x)
        out = x if self.norm_type=="None" else self.norm_layer(x)
        return out