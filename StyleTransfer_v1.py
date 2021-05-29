#!/usr/bin/env python3
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


import torchvision
import torchvision.transforms as transforms
from torchvision import transforms, models
from torchvision.utils import save_image

import os
import cv2 as cv
import pandas as pd

import requests
from io import BytesIO
import skimage.exposure as exposure
from skimage import io
import time

def load_image(img_path=None,url=None,max_size=400,shape=None):  
# Open the image, convert it into RGB and store in a variable   
    if url is not None:
      response = requests.get(url)
      if response.status_code == 200:
          image = Image.open(BytesIO(response.content))
      else:
          print('An error has occurred.') 
    else:
      image=Image.open(img_path).convert('RGB')  
    # comparing image size with the maximum size   
    if max(image.size)>max_size:  
      size=max_size  
    else:  
      size=max(image.size)  
    # checking for the image shape  
    if shape is not None:  
       size=shape  
    #Applying appropriate transformation to our image such as Resize, ToTensor and Normalization  
    in_transform=transforms.Compose([  
        transforms.Resize(size),  
        transforms.ToTensor(),  
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])  
    #Calling in_transform with our image   
    image=in_transform(image).unsqueeze(0) #unsqueeze(0) is used to add extra layer of dimensionality to the image  
    #Returning image   
    return image

def im_convert(tensor):  
  image=tensor.cpu().clone().detach().numpy()     
  image=image.squeeze()  
  image=image.transpose(1,2,0)  
  image=image*np.array((0.5,0.5,0.5))+np.array((0.5,0.5,0.5))  
  image=image.clip(0,1)  
  return image  

def im_convertT(img, max_size=None):
    # Rescale the image
    if (max_size==None):
        in_transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            # transforms.Lambda(lambda x: x.mul(255))
        ])    
    else:
        H, W, C = img.shape
        image_size = tuple([int((float(max_size) / max([H,W]))*x) for x in [H, W]])
        in_transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            # transforms.Lambda(lambda x: x.mul(255))
        ])
    # Convert image to tensor
    tensor = in_transform(img)
    tensor = tensor.unsqueeze(dim=0)
    return tensor

def resize2d(img, size):
    return (F.adaptive_avg_pool2d(Variable(img), size)).data
    
def showAImg(img,name):
    # plt.title(name)
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(name, bbox_inches='tight', pad_inches=0)
    plt.show()

def showStyleContentTarget(style,content,target,title):
    fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(18,5))  
    fig.suptitle(title, fontsize=16)   
    #Plotting content image   
    ax1.imshow(im_convert(content))  
    ax1.axis('off')  
    #Plotting style image  
    ax2.imshow(im_convert(style))  
    ax2.axis('off')  
    #Plotting target image  
    ax3.imshow(im_convert(target))  
    ax3.axis('off') 
    plt.show() 

def get_features(image,feature_layers,model):
    features = {}
    for name,layer in model._modules.items():
        image = layer(image)
        if name in feature_layers:
            features[feature_layers[name]]=image
    return  features

def gram_matrix(tensor):  
   #Unwrapping the tensor dimensions into respective variables i.e. batch size, distance, height and width   
  _,d,h,w=tensor.size()   
  #Reshaping data into a two dimensional of array or two dimensional of tensor  
  tensor=tensor.view(d,h*w)  
  #Multiplying the original tensor with its own transpose using torch.mm   
  #tensor.t() will return the transpose of original tensor  
  gram=torch.mm(tensor,tensor.t())  
  #Returning gram matrix   
  return gram  

def train(content_image,content_weight,style_image,style_weight,model,steps,device):
    feature_layers = {'0':'conv1_1','5':'conv2_1','10':'conv3_1','19':'conv4_1','21':'conv4_2','28':'conv5_1'}
    content_features=get_features(content_image,feature_layers,model)
    print("======> content_features")
    print(content_features)
    showAImg(content_features[0][0])
    style_features=get_features(style_image,feature_layers,model)
    print("======> style_features")
    print(style_features)
    style_grams={layer:gram_matrix(style_features[layer]) for layer in style_features}  

    #Initializing style_weights dictionary  
    style_weights={'conv1_1':1.,      #Key 1 with max value 1  
                'conv2_1':0.75,  #Key 2 with max value 0.75  
                'conv3_1':0.2,    #Key 3 with max value 0.2  
                'conv4_1':0.2,   #Key 4 with max value 0.2  
                'conv5_1':0.2}   #Key 5 with max value 0.2  

    # show_every=15000
    show_every = steps

    target=content_image.clone().requires_grad_(True).to(device)
    # target=torch.randn(content_image.size()).type_as(content_image.data).requires_grad_(True).to(device) #random init

    optimizer=optim.Adam([target],lr=0.03)  
    result = []

    start_time = time.time()
    for ii in range(1,steps+1):
        target_features = get_features(target,feature_layers,model)
        content_loss = torch.mean((target_features['conv4_2']-content_features['conv4_2'])**2)  
        style_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
            _, d, h, w = target_feature.shape
            style_loss += layer_style_loss / (d * h * w)
        total_loss = content_weight*content_loss+style_weight*style_loss

        #Using the optimizer to update parameters within our target image
        optimizer.zero_grad()  
        total_loss.backward()  
        optimizer.step()  

        #Comparing the iteration variable with our show every    
        if ii % show_every==0:
            result.append([ii,style_loss.item(),content_loss.item(),total_loss.item(),target])
            # print('Iteration ',ii,' / content loss = ',content_loss.item(),'  style loss = ',style_loss.item(),' => total loss = ',total_loss.item())
            # plt.imshow(im_convert(target))  
            # plt.axis('off')  
            # plt.show()
        #print('Iteration ',ii,' / content loss = ',content_loss.item(),'  style loss = ',style_loss.item(),' => total loss = ',total_loss.item())
    stop_time = time.time()
    # Show Result
    print("Iteration : {} , Training time : {} seconds".format(steps,stop_time-start_time))
    return target,result

# save image(numpy)
def saveImg(img,img_path):
    if type(img).__module__=='numpy':
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = img.astype(np.uint8)
        io.imsave(img_path, img)
    else:
        print('Image should be numpy class.')
        save_image(img, img_path)

# color preservation : histogram matching, luminance only transfer
class ColorPreservation(object):
    def __init__(self):
        print('Color preservation...')
    
    def match(name,src,ref):
        if name == 'his':
            print('Using method Histogram Matching')
            img = ColorPreservation.histogramMatching(src, ref)
        elif name == 'lumi':
            print('Using method Luminance Only Transfer')
            img = ColorPreservation.luminanceOnlyTransfer(src, ref)
        return img

    # color histogram matching by put source(numpy) and reference(numpy)
    def histogramMatching(src, ref):
        multi = True if src.shape[-1] > 1 else False
        matched = exposure.match_histograms(src, ref, multichannel = multi)
        return matched

    # find mean, std for luminance transfer
    def mean_std(image):
        mean = [
            np.mean(image[:, :, 0]),
            np.mean(image[:, :, 1]),
            np.mean(image[:, :, 2])
        ]
        std = [
            np.std(image[:, :, 0]),
            np.std(image[:, :, 1]),
            np.std(image[:, :, 2])
        ]
        return mean,std

    # color luminance transfer by put source(numpy) and reference(numpy)
    def luminanceOnlyTransfer(src, ref):
        src = cv.cvtColor(src,cv.COLOR_BGR2LAB)
        ref = cv.cvtColor(ref,cv.COLOR_BGR2LAB)
        Ms,SDs = ColorPreservation.mean_std(src)
        Mr,SDr = ColorPreservation.mean_std(ref)
        H,W,D = src.shape
        for h in range(0,H):
            for w in range(0,W):
                for d in range(0,D):
                    luminance_px = src[h,w,d]
                    luminance_px = (SDr[d]/SDs[d])*(luminance_px-Ms[d])+Mr[d]
                    luminance_px = 0 if luminance_px<0 else 255 if luminance_px>255 else luminance_px
                    src[h,w,d] = luminance_px
        src = cv.cvtColor(src,cv.COLOR_LAB2BGR)
        return src

def main(method,image_type,style,style_weight,content,content_weight,pool,iteration):
    #importing model features   
    if pool == 'max':
        print('VGG19 using max pooling')
        vgg=models.vgg19(pretrained=True).features  
        for param in vgg.parameters():  
            param.requires_grad_(False)  
    elif pool == 'avg':
        print('VGG19 using average pooling')
        vgg = models.vgg19(pretrained=True).features
        for param in vgg.parameters():  
            param.requires_grad_(False) 
        layers = {'4':'max_1',
                '9':'max_2',
                '18':'max_3',
                '27':'max_4',
                '36':'max_5'}
        for name, layer in vgg._modules.items():
            if name in layers: 
                vgg._modules[name] = nn.AvgPool2d(kernel_size=2, stride=2,padding=0)
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using ',device,' to process')
    vgg.to(device)   

    # load image 
    if image_type=='path':
        style_image = load_image(style).to(device)
        # showAImg(im_convert(style_image),'style image')
        content_image = load_image(content).to(device)
        # showAImg(im_convert(content_image), 'content image')
    elif image_type=='url':
        style_image = load_image(url=style).to(device)
        # showAImg(im_convert(style_image),'style image')
        content_image = load_image(url=content).to(device)
        # showAImg(im_convert(content_image), 'content image')
    style_image = resize2d(style_image, (224,224))
    content_image = resize2d(content_image, (224,224))

    if method == 'before his':
        print('before his')
        # histogram
        style_image = im_convert(style_image)
        content_im = im_convert(content_image)
        style_his_img = ColorPreservation.match('his',style_image,content_im)
        style_image = im_convertT(style_his_img).type(torch.cuda.FloatTensor)
    elif method == 'before lumi':
        print('before lumi')
        # luminance
        style_image = im_convert(style_image)
        content_im = im_convert(content_image)
        style_lumi_img = ColorPreservation.match('lumi',style_image,content_im)
        style_image = im_convertT(style_lumi_img).type(torch.cuda.FloatTensor)
    elif method == 'before style':
        print('before style')
        style_image = style_image

    print('start train')
    target,result = train(content_image,content_weight,style_image,style_weight,vgg,iteration,device)
    print('finish train')
    title = 'Iteration '+str(result[0][0])+' content loss : {:2f}'.format(result[0][2]) +' style loss : {:2f}'.format(result[0][1]) +' total loss : {:2f}'.format(result[0][3])
    showStyleContentTarget(style_image, content_image,target,title)
    
    
    target = im_convert(target)
    showAImg(target,'target.jpg')
    # saveImg(target,'target.jpg')
    content = im_convert(content_image)
    # histogram matching
    
    if method == 'after':
        print('after')
        his_img = ColorPreservation.match('his',target,content)
        # saveImg(his_img,'after_his_img.jpg')
        showAImg(his_img,'after_his_img.jpg')

        # lumi_img = ColorPreservation.match('lumi',target,content)
        # saveImg(lumi_img,'after_lumi_img.jpg')
        # showAImg(lumi_img,name='Image')


if __name__ == "__main__":
    # get_ipython().run_line_magic('matplotlib', 'inline')
    # IMAGE_TYPE = 'url'
    # STYLE_IMG = 'https://github.com/nawaritlk/Senior_project/blob/frong/Model/StyleImage/Chakrabhan/0001.jpg?raw=true'
    # CONTENT_IMG = 'https://github.com/nawaritlk/Senior_project/blob/frong/Model/Test/ContentImg.jpg?raw=true'
    # or
    # IMAGE_TYPE = 'path'
    # STYLE_IMG = r'.\Test\StyleImage\Chakrabhan\0001.jpg'
    # CONTENT_IMG = r'.\Test\ContentImg.jpg'

    #use in colab
    IMAGE_TYPE = 'path'
    STYLE_IMG = r'./Test/StyleImage/Chakrabhan/0001.jpg'
    CONTENT_IMG = r'./Test/ContentImage/animals/Abyssinian_13.jpg'

    ITERATION = 5000
    CONTENT_WEIGHT = 1e-2
    STYLE_WEIGHT = 1e6

    MODEL_POOLING = 'max' # or 'avg'

    METHOD = 'before his' # 'before style', 'before lumi', 'before his', 'after'
    main(METHOD,IMAGE_TYPE,STYLE_IMG,STYLE_WEIGHT,CONTENT_IMG,CONTENT_WEIGHT,MODEL_POOLING,ITERATION)

    METHOD = 'before style' # 'before style', 'before lumi', 'before his', 'after'
    main(METHOD,IMAGE_TYPE,STYLE_IMG,STYLE_WEIGHT,CONTENT_IMG,CONTENT_WEIGHT,MODEL_POOLING,ITERATION)

    # METHOD = 'before lumi' # 'before style', 'before lumi', 'before his', 'after'
    # main(METHOD,IMAGE_TYPE,STYLE_IMG,STYLE_WEIGHT,CONTENT_IMG,CONTENT_WEIGHT,MODEL_POOLING,ITERATION)

    METHOD = 'after' # 'before style', 'before lumi', 'before his', 'after'
    main(METHOD,IMAGE_TYPE,STYLE_IMG,STYLE_WEIGHT,CONTENT_IMG,CONTENT_WEIGHT,MODEL_POOLING,ITERATION)

    MODEL_POOLING = 'avg'

    METHOD = 'before his' # 'before style', 'before lumi', 'before his', 'after'
    main(METHOD,IMAGE_TYPE,STYLE_IMG,STYLE_WEIGHT,CONTENT_IMG,CONTENT_WEIGHT,MODEL_POOLING,ITERATION)

    METHOD = 'before style' # 'before style', 'before lumi', 'before his', 'after'
    main(METHOD,IMAGE_TYPE,STYLE_IMG,STYLE_WEIGHT,CONTENT_IMG,CONTENT_WEIGHT,MODEL_POOLING,ITERATION)

    # METHOD = 'before lumi' # 'before style', 'before lumi', 'before his', 'after'
    # main(METHOD,IMAGE_TYPE,STYLE_IMG,STYLE_WEIGHT,CONTENT_IMG,CONTENT_WEIGHT,MODEL_POOLING,ITERATION)

    METHOD = 'after' # 'before style', 'before lumi', 'before his', 'after'
    main(METHOD,IMAGE_TYPE,STYLE_IMG,STYLE_WEIGHT,CONTENT_IMG,CONTENT_WEIGHT,MODEL_POOLING,ITERATION)