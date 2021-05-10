from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


import torchvision
import torchvision.transforms as transforms
from torchvision import models

import os
import cv2 as cv
import pandas as pd
import time

import utils as fn
import model

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('=> Using ',device,' to process')
    if device.type == 'cuda':
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:',torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)
    return device

DEVICE = get_device()

def train(VGG,NUM_EPOCHS,ADAM_LR,style,STYLE_WEIGHT,content,CONTENT_WEIGHT,target):
    style_features = VGG(style)
    style_gram = {}
    for key, value in style_features.items():
        style_gram[key] = model.gram_matrix(value)

    # Optimizer
    optimizer = optim.Adam([target],lr=ADAM_LR)

    content_loss_history = []
    style_loss_history = []
    total_loss_history = []

    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        # print("Epoch : {}/{} running..........".format(epoch+1,NUM_EPOCHS))
        
        torch.cuda.empty_cache()

        optimizer.zero_grad()

        content_features = VGG(content)
        generated_features = VGG(target)

        # content loss
        MSELoss = nn.MSELoss().to(DEVICE)
        content_loss = CONTENT_WEIGHT * MSELoss(generated_features['conv4_2'],content_features['conv4_2'])

        # style loss
        style_loss = 0
        for key,value in generated_features.items():
            s_loss = MSELoss(model.gram_matrix(value),style_gram[key])
            style_loss += s_loss
        style_loss *= STYLE_WEIGHT

        # total loss
        total_loss = CONTENT_WEIGHT*content_loss + STYLE_WEIGHT*style_loss

        total_loss.backward()
        optimizer.step()

        content_loss_history.append(content_loss.item())
        style_loss_history.append(style_loss.item())
        total_loss_history.append(total_loss.item())

    stop_time = time.time()
    
    # Show Result
    print("Training time : {} seconds".format(stop_time-start_time))
    print("=> Content Loss ",content_loss_history[-1])
    print("=> Style Loss ",style_loss_history[-1])
    print("=> Total Loss ",total_loss_history[-1])
    fn.plotLoss(content_loss_history, style_loss_history, total_loss_history)
    
    target_img = fn.im_convert(target)
    fn.show3Image(fn.im_convert(content), fn.im_convert(style), target_img)
    # target(numpy)
    return  target_img

def main(VGG,size,stylePath,contentPath,method,color,NUM_EPOCHS,ADAM_LR,STYLE_WEIGHT,CONTENT_WEIGHT):
    # main(pool,size,stylePath,contentPath,method,color,NUM_EPOCHS,ADAM_LR,STYLE_WEIGHT,CONTENT_WEIGHT)
    # device = get_device()

    # VGG = model.VGG19(pool=pool).to(DEVICE)
    # print(VGG.name) # VGG.features

    # load style image
    Style = fn.AImage(stylePath)
    Content = fn.AImage(contentPath)
    # fn.show2Image(Style.Img,Content.Img)
    
    if method == 'before':
        print('=> Before style transfer')
        if color == 'histogram':
            style_his = fn.ColorPreservation.histogramMatching(Style.Img, Content.Img)
            fn.show2Image(Style.Img,style_his,'Original Style','Style(histogram matching)')
            style_tensor = fn.im_convertT(style_his)
            style = fn.FImg.resize(style_tensor, size).to(DEVICE)
        elif color == 'luminance':
            style_lumi = fn.ColorPreservation.luminanceOnlyTransfer(Style.Img, Content.Img)
            fn.show2Image(Style.Img,style_lumi,'Original Style','Style(luminance only transfer)')
            style_tensor = fn.im_convertT(style_lumi)
            style = fn.FImg.resize(style_tensor, size).to(DEVICE)
        else:
            print('Do not use color preservation')
            style = fn.FImg.resize(Style.Tensor, size).to(DEVICE)
        content = fn.FImg.resize(Content.Tensor, size).to(DEVICE)

    target=content.clone().requires_grad_(True).to(DEVICE)
    # target=torch.randn(content.size()).type_as(content.data).requires_grad_(True).to(device) #random init

    generate_img = train(VGG,NUM_EPOCHS,ADAM_LR,style,STYLE_WEIGHT,content,CONTENT_WEIGHT,target)
    
    if method == 'after':
        print('=> After style transfer')
        if color == 'histogram':
            generate_his = fn.ColorPreservation.histogramMatching(generate_img, Content.Img)
            fn.show2Image(generate_img,generate_his,'Original Generated','Generated(histogram matching)')
            return generate_his
        elif color == 'luminance':
            generate_lumi = fn.ColorPreservation.luminanceOnlyTransfer(generate_img, Content.Img)
            fn.show2Image(generate_img,generate_lumi,'Original Generated','Generated(luminance only transfer)')
            return generate_lumi
        else:
            print('Do not use color preservation')
            return generate_img

if __name__=="__main__":
    get_ipython().run_line_magic('matplotlib', 'inline')
    STYLE_IMG = r'./StyleImage/Chakrabhan/0001.jpg'
    CONTENT_IMG = r'./ContentImage/animals/Abyssinian_14.jpg'
    POOL = 'max' # or 'avg'
    METHOD = 'before' # or 'after'
    COLOR = None # or 'histogram' or 'luminance'

    NUM_EPOCHS = 5000
    ADAM_LR = 0.03
    STYLE_WEIGHT = 1e2
    CONTENT_WEIGHT = 1e-2
    IMG_SIZE = (224,224)
    
    VGG = model.VGG19(POOL).to(DEVICE)
    print(VGG.name) # VGG.features

    BASE_PATH = os.getcwd()
    a = main(VGG,IMG_SIZE,STYLE_IMG,CONTENT_IMG,METHOD,COLOR,NUM_EPOCHS,ADAM_LR,STYLE_WEIGHT,CONTENT_WEIGHT)
    fn.FImg.save(a, BASE_PATH+'/output/max_before_no.jpg')

    b = main(VGG,IMG_SIZE,STYLE_IMG,CONTENT_IMG,METHOD,'histogram',NUM_EPOCHS,ADAM_LR,STYLE_WEIGHT,CONTENT_WEIGHT)
    fn.FImg.save(a, BASE_PATH+'/output/max_before_no.jpg')

    c = main(VGG,IMG_SIZE,STYLE_IMG,CONTENT_IMG,METHOD,'luminance',NUM_EPOCHS,ADAM_LR,STYLE_WEIGHT,CONTENT_WEIGHT)
    fn.FImg.save(c, BASE_PATH+'/output/max_before_lumi.jpg')

    d = main(VGG,IMG_SIZE,STYLE_IMG,CONTENT_IMG,'after',COLOR,NUM_EPOCHS,ADAM_LR,STYLE_WEIGHT,CONTENT_WEIGHT)
    fn.FImg.save(d, BASE_PATH+'/output/max_after_no.jpg')

    e = main(VGG,IMG_SIZE,STYLE_IMG,CONTENT_IMG,'after','histogram',NUM_EPOCHS,ADAM_LR,STYLE_WEIGHT,CONTENT_WEIGHT)
    fn.FImg.save(e, BASE_PATH+'/output/max_after_his.jpg')

    f = main(VGG,IMG_SIZE,STYLE_IMG,CONTENT_IMG,'after','luminance',NUM_EPOCHS,ADAM_LR,STYLE_WEIGHT,CONTENT_WEIGHT)
    fn.FImg.save(f, BASE_PATH+'/output/max_after_lumi.jpg')

    POOL = 'avg'
    VGG = model.VGG19(POOL).to(DEVICE)
    print(VGG.name)

    g = main(VGG,IMG_SIZE,STYLE_IMG,CONTENT_IMG,METHOD,COLOR,NUM_EPOCHS,ADAM_LR,STYLE_WEIGHT,CONTENT_WEIGHT)
    fn.FImg.save(g, BASE_PATH+'/output/avg_before_no.jpg')

    h = main(VGG,IMG_SIZE,STYLE_IMG,CONTENT_IMG,METHOD,'histogram',NUM_EPOCHS,ADAM_LR,STYLE_WEIGHT,CONTENT_WEIGHT)
    fn.FImg.save(h, BASE_PATH+'/output/avg_before_his.jpg')

    i = main(VGG,IMG_SIZE,STYLE_IMG,CONTENT_IMG,METHOD,'luminance',NUM_EPOCHS,ADAM_LR,STYLE_WEIGHT,CONTENT_WEIGHT)
    fn.FImg.save(i, BASE_PATH+'/output/avg_before_lumi.jpg')

    j = main(VGG,IMG_SIZE,STYLE_IMG,CONTENT_IMG,'after',COLOR,NUM_EPOCHS,ADAM_LR,STYLE_WEIGHT,CONTENT_WEIGHT)
    fn.FImg.save(j, BASE_PATH+'/output/avg_after_no.jpg')

    k = main(VGG,IMG_SIZE,STYLE_IMG,CONTENT_IMG,'after','histogram',NUM_EPOCHS,ADAM_LR,STYLE_WEIGHT,CONTENT_WEIGHT)
    fn.FImg.save(k, BASE_PATH+'/output/avg_after_his.jpg')

    l = main(VGG,IMG_SIZE,STYLE_IMG,CONTENT_IMG,'after','luminance',NUM_EPOCHS,ADAM_LR,STYLE_WEIGHT,CONTENT_WEIGHT)
    fn.FImg.save(l, BASE_PATH+'/output/avg_after_lumi.jpg')

    
    # fn.FImg.save(a, BASE_PATH+'/output/max_before_no.jpg')
    # fn.FImg.save(b, BASE_PATH+'/output/max_before_his.jpg')
    # fn.FImg.save(c, BASE_PATH+'/output/max_before_lumi.jpg')
    # fn.FImg.save(d, BASE_PATH+'/output/max_after_no.jpg')
    # fn.FImg.save(e, BASE_PATH+'/output/max_after_his.jpg')
    # fn.FImg.save(f, BASE_PATH+'/output/max_after_lumi.jpg')
    # fn.FImg.save(g, BASE_PATH+'/output/max_before_no.jpg')
    # fn.FImg.save(h, BASE_PATH+'/output/max_before_his.jpg')
    # fn.FImg.save(i, BASE_PATH+'/output/max_before_lumi.jpg')
    # fn.FImg.save(j, BASE_PATH+'/output/max_after_no.jpg')
    # fn.FImg.save(k, BASE_PATH+'/output/max_after_his.jpg')
    # fn.FImg.save(l, BASE_PATH+'/output/max_after_lumi.jpg')
