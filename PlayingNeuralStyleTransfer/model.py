import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

# Alllayers = {'0':'conv1_1',
        #     '1':'relu1_1',
        #     '2':'conv1_2',
        #     '3':'relu1_2',
        #     '4':'max_1',
        #     '5':'conv2_1',
        #     '6':'relu2_1',
        #     '7':'conv2_2',
        #     '8':'relu2_2',
        #     '9':'max_2',
        #     '10':'conv3_1',
        #     '11':'relu3_1',
        #     '12':'conv3_2',
        #     '13':'relu3_2',
        #     '14':'conv3_3',
        #     '15':'relu3_3',
        #     '16':'conv3_4',
        #     '17':'relu3_4',
        #     '18':'max_3',
        #     '19':'conv4_1',
        #     '20':'relu4_1',
        #     '21':'conv4_2',
        #     '22':'relu4_2',
        #     '23':'conv4_3',
        #     '24':'relu4_3',
        #     '25':'conv4_4',
        #     '26':'relu4_4',
        #     '27':'max_4',
        #     '28':'conv5_1',
        #     '29':'relu5_1',
        #     '30':'conv5_2',
        #     '31':'relu5_2',
        #     '32':'conv5_3',
        #     '33':'relu5_3',
        #     '34':'conv5_4',
        #     '35':'relu5_4',
        #     '36':'max_5'}

import os
BASE_PATH = os.getcwd()

class VGG19(nn.Module):
    def __init__(self,vgg19_path=None,pool="max"):
        super(VGG19, self).__init__()
        self.name = 'Model vgg19 using max pooling'
        vgg19_features = models.vgg19(pretrained=True)
        # print(vgg19_path)
        if vgg19_path is not None:
            vgg19_features.load_state_dict(torch.load(vgg19_path),strict=False)
        
        self.features = vgg19_features.features

        for param in self.features.parameters():
            param.requires_grad = False

        if pool=="avg":
            self.name = 'Model vgg19 using average pooling'
            layers = {'4':'max_1','9':'max_2','18':'max_3','27':'max_4','36':'max_5'}
            for name, layer in self.features._modules.items():
                if name in layers: 
                    self.features._modules[name] = nn.AvgPool2d(kernel_size=2, stride=2,padding=0)

    def forward(self, x):
        layers = {'0':'conv1_1','5':'conv2_1','10':'conv3_1','19':'conv4_1','21':'conv4_2','28':'conv5_1'}
        features = {}
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

# gram_matrix function for tensor image  
def gram_matrix(tensor):  
    _,c,h,w = tensor.size()    
    tensor = tensor.view(c,h*w)    
    gram = torch.mm(tensor,tensor.t())  
    return gram

class ContentLoss(nn.Module):
    def __init__(self,target):
        super(ContentLoss,self).__init__()
        self.target = target.detach()

    def forward(self,input):
        self.loss = F.mse_loss(input,self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self,target_feature):
        super(StyleLoss,self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self,input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G,self.target)
        return input

class TVLoss(nn.Module):
    def __init__(self,TVloss_weight=1):
        super(TVLoss,self).__init__()
        self.TVloss_weight = TVloss_weight

    def forward(self,x):
        w_variance = torch.sum(torch.pow(x[:,:,:,:-1]-x[:,:,:,1:],2))
        h_variance = torch.sum(torch.pow(x[:,:,:-1,:]-x[:,:,1:,:],2))
        loss = self.TVloss_weight * (h_variance + w_variance)
        return loss

# use for fast transfer
class VGG19FT(nn.Module):
    def __init__(self,vgg19_path=None,pool="max"):
        super(VGG19, self).__init__()
        self.name = 'Model vgg19 using max pooling'
        vgg19_features = models.vgg19(pretrained=False)
        # print(vgg19_path)
        if vgg19_path is not None:
            vgg19_features.load_state_dict(torch.load(vgg19_path),strict=False)
        
        self.features = vgg19_features.features

        for param in self.features.parameters():
            param.requires_grad = False

        if pool=="avg":
            self.name = 'Model vgg19 using average pooling'
            layers = {'4':'max_1','9':'max_2','18':'max_3','27':'max_4','36':'max_5'}
            for name, layer in self.features._modules.items():
                if name in layers: 
                    self.features._modules[name] = nn.AvgPool2d(kernel_size=2, stride=2,padding=0)

    def forward(self, x):
        layers = {'3':'relu1_2','8':'relu2_2','17':'relu3_4','22':'relu4_2','26':'relu4_4','35':'relu5_4'}
        features = {}
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

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


def set_optimizer(input,optimizer="adam",lr=0.01):
    if optimizer=="adam":
        print("Optimization with Adam")
        optimizer = optim.Adam([input],lr=lr)
    elif optimizer=="lbfgs":
        print("Optimization with LBFGS")
        optimizer = optim.LBFGS([input])
        return optimizer

if __name__=="__main__":
    pass