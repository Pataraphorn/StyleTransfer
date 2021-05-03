import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import time

import model
import function
import transformer
import os

# SETTINGS
IMAGE_SIZE = [224,224]
NUM_EPOCHS = 5000
# BATCH_SIZE = 4
CONTENT_WEIGHT = 17
STYLE_WEIGHT = 50
ADAM_LEARNING_RATE = 0.001

BASE_PATH = os.getcwd()
# SAVE OUTPUT
SAVE_EVERY = 1000
SAVE_MODEL_PATH = BASE_PATH+r'/outputs/transfer/models'
SAVE_IMAGE_PATH = BASE_PATH+r'/outputs/transfer/generated_image'

DATASET_PATH = r'/ContentImage'

# SEED = 35
PLOT_LOSS = 1

# PATH IMAGE
STYLE_IMG_PATH = BASE_PATH+r'/StyleImage/Chakrabhan/0001.jpg'
CONTENT_IMG_PATH = BASE_PATH+r'/ContentImg.jpg'

def train():
    # check device
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print("using ",device)
    # if device=="cuda":
    #     print('__CUDNN VERSION:', torch.backends.cudnn.version())
    #     print('__Number CUDA Devices:', torch.cuda.device_count())
    #     print('__CUDA Device Name:',torch.cuda.get_device_name(0))
    #     print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

    # Image
    style_image = function.OneImage(STYLE_IMG_PATH)
    style_img = style_image.Tensor.to(device)

    content_image = function.OneImage(CONTENT_IMG_PATH)
    content_img = content_image.Tensor.to(device)

    # Target is clone content
    generated_img = content_img.clone().requires_grad_(True).to(device)
    # function.showImg(function.im_convert(generated_img))
    
    # # Target is noise
    # generated_img = torch.randn(content_img.size()).type_as(content_img.data).requires_grad_(True).to(device) #random init
    # function.showImg(function.im_convert(generated_img))

    function.show3Image(style_image.img, content_image.img, function.im_convert(generated_img))

    # Model
    VGG = model.VGG19().to(device)
    # AVGVGG = model.VGG19(pool="avg").to(device)
    # print(VGG)

    # style feature
    style_features = VGG(style_img)
    style_gram = {}
    for key, value in style_features.items():
        style_gram[key] = function.gram_matrix(value)

    # Optimizer
    optimizer = optim.Adam([generated_img],lr=ADAM_LEARNING_RATE)

    # history loss
    content_loss_history = []
    style_loss_history = []
    total_loss_history = []

    # Training
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        print("Epoch : {}/{}".format(epoch+1,NUM_EPOCHS))
        torch.cuda.empty_cache()

        generated_features = VGG(generated_img)
        content_features = VGG(content_img)
        style_features = VGG(style_img)

        optimizer.zero_grad()

        # content loss
        MSELoss = nn.MSELoss().to(device)
        content_loss = CONTENT_WEIGHT * MSELoss(generated_features['relu2_2'],content_features['relu2_2'])
        
        # style loss
        style_loss = 0
        for key,value in generated_features.items():
            s_loss = MSELoss(function.gram_matrix(value),style_gram[key])
            style_loss += s_loss
        style_loss *= STYLE_WEIGHT

        # total loss
        total_loss = content_loss+style_loss

        total_loss.backward()
        optimizer.step()

        print("\tContent Loss : {:2f}".format(content_loss))
        print("\tStyle Loss : {:2f}".format(style_loss))
        print("\tTotal Loss : {:2f}".format(total_loss))
        print("\t=> Time elapsed : {} seconds".format(time.time()-start_time))
        function.showImg(function.im_convert(generated_img))

        content_loss_history.append(content_loss)
        style_loss_history.append(style_loss)
        total_loss_history.append(total_loss)

        # # save model
        if ((epoch-1)%SAVE_EVERY==0):
            generated_image = function.im_convert(generated_img)
            generated_path = SAVE_IMAGE_PATH+"/sample_"+str(epoch-1)+'.jpg'
            # function.saveImg(sample_image,sample_path)
            print(type(generated_image))
            function.saveImg(generated_image ,generated_path)
            function.showImg(generated_image)
            print("Save generated image at {}".format(generated_path))
        
    stop_time = time.time()

    # Show Result
    print("Training time : {} seconds".format(stop_time-start_time))
    print("=> Iteration : ",NUM_EPOCHS)
    print("=> Content Loss ",content_loss_history)
    print("=> Style Loss ",style_loss_history)
    print("=> Total Loss ",total_loss_history)

    VGG.eval()
    VGG.cpu()
    final_path = SAVE_MODEL_PATH+"_vgg_weight.pt"
    print("Save VGG weight at {}".format(final_path))
    torch.save(VGG.state_dict(),final_path)
    print("Save final model")

    if(PLOT_LOSS):
        function.plotLoss(content_loss_history,style_loss_history,total_loss_history)