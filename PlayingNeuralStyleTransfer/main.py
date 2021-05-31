import torch
import torch.nn as nn
import torch.optim as optim

import time

import utils as fn
import model

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('=> Using ',device,' to process')
    # if device.type == 'cuda':
    #     print('__CUDNN VERSION:', torch.backends.cudnn.version())
    #     print('__Number CUDA Devices:', torch.cuda.device_count())
    #     print('__CUDA Device Name:',torch.cuda.get_device_name(0))
    #     print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)
    return device

def train(DEVICE,VGG,NUM_EPOCHS,ADAM_LR,style,STYLE_WEIGHT,content,CONTENT_WEIGHT,target):
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
    print("Running..........")
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
        total_loss = content_loss + style_loss

        total_loss.backward()
        optimizer.step()

        # if (epoch+1)%100==0:
        #     fn.show3Image(fn.im_convert(content), fn.im_convert(style), fn.im_convert(target))
            # fn.FImg.save(fn.im_convert(target), 'main_it_'+str(epoch+1)+'.jpg')

        content_loss_history.append(content_loss.item())
        style_loss_history.append(style_loss.item())
        total_loss_history.append(total_loss.item())

    stop_time = time.time()
    
    # Show Result
    print("Iteration : {} , Training time : {} seconds".format(NUM_EPOCHS,stop_time-start_time))
    print("=> Content Loss ",content_loss_history[-1])
    print("=> Style Loss ",style_loss_history[-1])
    print("=> Total Loss ",total_loss_history[-1])
    fn.plotLoss(content_loss_history, style_loss_history, total_loss_history)
    
    target_img = fn.im_convert(target)
    # fn.show3Image(fn.im_convert(content), fn.im_convert(style), target_img)
    # target(numpy)
    return  target_img

def main(VGG,size,stylePath,contentPath,method,color,NUM_EPOCHS,ADAM_LR,STYLE_WEIGHT,CONTENT_WEIGHT):
    # main(pool,size,stylePath,contentPath,method,color,NUM_EPOCHS,ADAM_LR,STYLE_WEIGHT,CONTENT_WEIGHT)
    DEVICE = get_device()

    # VGG = model.VGG19(pool=pool).to(DEVICE)
    # print(VGG.name) # VGG.features
    VGG = VGG.to(DEVICE)

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
            fn.show2Image(Style.Img,style_lumi,'Original Style','Style(luminance matching)')
            style_tensor = fn.im_convertT(style_lumi)
            style = fn.FImg.resize(style_tensor, size).to(DEVICE)
        else:
            # print('Do not use color preservation')
            style = fn.FImg.resize(Style.Tensor, size).to(DEVICE)
        content = fn.FImg.resize(Content.Tensor, size).to(DEVICE)
    else:
        style = fn.FImg.resize(Style.Tensor, size).to(DEVICE)
        content = fn.FImg.resize(Content.Tensor, size).to(DEVICE)

    target=content.clone().requires_grad_(True).to(DEVICE)
    # target=torch.randn(content.size()).type_as(content.data).requires_grad_(True).to(device) #random init

    generate_img = train(DEVICE,VGG,NUM_EPOCHS,ADAM_LR,style,STYLE_WEIGHT,content,CONTENT_WEIGHT,target)
    
    if method == 'after':
        print('=> After style transfer')
        if color == 'histogram':
            ihis = fn.After(generate_img)
            generate_his = fn.ColorPreservation.histogramMatching(ihis, Content.Img)
            fn.show2Image(generate_img,generate_his,'Original Generated','Generated(histogram matching)')
            fn.show3Image(Content.Img, Style.Img, generate_his)
            return generate_his
        elif color == 'luminance':
            ilumi = fn.After(generate_img)
            generate_lumi = fn.ColorPreservation.luminanceOnlyTransfer(ilumi, Content.Img)
            fn.show2Image(generate_img,generate_lumi,'Original Generated','Generated(luminance matching)')
            fn.show3Image(Content.Img, Style.Img, generate_lumi)
            return generate_lumi
        else:
            # print('Do not use color preservation')
            fn.show3Image(Content.Img, Style.Img, generate_img)
            return generate_img
    else:
        fn.show3Image(Content.Img, Style.Img, generate_img)
        return generate_img


if __name__=="__main__":
    pass
    # get_ipython().run_line_magic('matplotlib', 'inline')