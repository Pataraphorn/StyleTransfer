import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models

import time
import utils as fn

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('=> Using ',device,' to process')
    if device.type == 'cuda':
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:',torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)
    return device

def VGG19(pool="max",vgg19_path=None):
    vgg19 = models.vgg19(pretrained=False)
    # print(vgg19_path)
    if vgg19_path is not None:
        vgg19.load_state_dict(torch.load(vgg19_path),strict=False)
    
    vgg19_features = vgg19.features
    for param in vgg19_features.parameters():
        param.requires_grad = False
    
    if pool=="max":
        print('Model vgg19 using max pooling')
    elif pool=="avg":
        print('Model vgg19 using average pooling')
        layers = {'4':'max_1','9':'max_2','18':'max_3','27':'max_4','36':'max_5'}
        for name, layer in vgg19_features._modules.items():
            if name in layers: 
                vgg19_features._modules[name] = nn.AvgPool2d(kernel_size=2, stride=2,padding=0)
    return vgg19_features

def get_features(model_features,x):
    layers = {'0':'conv1_1','5':'conv2_1','10':'conv3_1','19':'conv4_1','21':'conv4_2','28':'conv5_1'}
    features = {}
    for name, layer in model_features._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):  
    b,c,h,w=tensor.shape   
    tensor=tensor.view(b,c,h*w)    
    tensor_transpose = tensor.transpose(1,2) 
    return torch.bmm(tensor,tensor_transpose) / (c*h*w)

def train(VGG,NUM_EPOCHS,ADAM_LR,style,STYLE_WEIGHT,content,CONTENT_WEIGHT,target):
    content_features = get_features(VGG,content)
    style_features = get_features(VGG,style)
    style_grams={layer:gram_matrix(style_features[layer]) for layer in style_features} 

    #Initializing style_weights dictionary  
    style_weights={'conv1_1':1.,      #Key 1 with max value 1  
                'conv2_1':0.75,  #Key 2 with max value 0.75  
                'conv3_1':0.2,    #Key 3 with max value 0.2  
                'conv4_1':0.2,   #Key 4 with max value 0.2  
                'conv5_1':0.2}   #Key 5 with max value 0.2  

    # Optimizer
    optimizer = optim.Adam([target],lr=ADAM_LR)

    content_loss_history = []
    style_loss_history = []
    total_loss_history = [] 

    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        # print("Epoch : {}/{} running..........".format(epoch+1,NUM_EPOCHS))
        
        # torch.cuda.empty_cache()

        generated_features = get_features(VGG,target)
        # content loss
        content_loss = torch.mean((generated_features['conv4_2']-content_features['conv4_2'])**2)
        # style loss
        style_loss = 0
        for layer in style_weights:
            target_feature = generated_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
            _, d, h, w = target_feature.shape
            style_loss += layer_style_loss / (d * h * w)
        
        # total loss
        total_loss = CONTENT_WEIGHT*content_loss + STYLE_WEIGHT*style_loss

        optimizer.zero_grad()
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

def main(pool,size,stylePath,contentPath,method,color,NUM_EPOCHS,ADAM_LR,STYLE_WEIGHT,CONTENT_WEIGHT):
    DEVICE = get_device()

    VGG = VGG19(pool=pool).to(DEVICE)

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
    return generate_img



if __name__=="__main__":
    pass
    # get_ipython().run_line_magic('matplotlib', 'inline')
    # STYLE_IMG = r'./StyleImage/Chakrabhan/0001.jpg'
    # CONTENT_IMG = r'./ContentImage/animals/Abyssinian_14.jpg'
    # POOL = 'max' # or 'avg'
    # METHOD = 'before' # or 'after'
    # COLOR = None # or 'histogram' or 'luminance'

    # NUM_EPOCHS = 5000
    # ADAM_LR = 0.003
    # STYLE_WEIGHT = 1e2
    # CONTENT_WEIGHT = 1e-2
    # IMG_SIZE = (224,224)

    # DEVICE = get_device()
    # VGG = model.VGG19(POOL).to(DEVICE)
    # print(VGG.name) # VGG.features

    # BASE_PATH = os.getcwd()
    # a = main(VGG,IMG_SIZE,STYLE_IMG,CONTENT_IMG,METHOD,COLOR,NUM_EPOCHS,ADAM_LR,STYLE_WEIGHT,CONTENT_WEIGHT)
    # fn.FImg.save(a, BASE_PATH+'/output/max_before_no.jpg')

    # b = main(VGG,IMG_SIZE,STYLE_IMG,CONTENT_IMG,METHOD,'histogram',NUM_EPOCHS,ADAM_LR,STYLE_WEIGHT,CONTENT_WEIGHT)
    # fn.FImg.save(a, BASE_PATH+'/output/max_before_no.jpg')

    # c = main(VGG,IMG_SIZE,STYLE_IMG,CONTENT_IMG,METHOD,'luminance',NUM_EPOCHS,ADAM_LR,STYLE_WEIGHT,CONTENT_WEIGHT)
    # fn.FImg.save(c, BASE_PATH+'/output/max_before_lumi.jpg')

    # d = main(VGG,IMG_SIZE,STYLE_IMG,CONTENT_IMG,'after',COLOR,NUM_EPOCHS,ADAM_LR,STYLE_WEIGHT,CONTENT_WEIGHT)
    # fn.FImg.save(d, BASE_PATH+'/output/max_after_no.jpg')

    # e = main(VGG,IMG_SIZE,STYLE_IMG,CONTENT_IMG,'after','histogram',NUM_EPOCHS,ADAM_LR,STYLE_WEIGHT,CONTENT_WEIGHT)
    # fn.FImg.save(e, BASE_PATH+'/output/max_after_his.jpg')

    # f = main(VGG,IMG_SIZE,STYLE_IMG,CONTENT_IMG,'after','luminance',NUM_EPOCHS,ADAM_LR,STYLE_WEIGHT,CONTENT_WEIGHT)
    # fn.FImg.save(f, BASE_PATH+'/output/max_after_lumi.jpg')

    # POOL = 'avg'
    # VGG = model.VGG19(POOL).to(DEVICE)
    # print(VGG.name)

    # g = main(VGG,IMG_SIZE,STYLE_IMG,CONTENT_IMG,METHOD,COLOR,NUM_EPOCHS,ADAM_LR,STYLE_WEIGHT,CONTENT_WEIGHT)
    # fn.FImg.save(g, BASE_PATH+'/output/avg_before_no.jpg')

    # h = main(VGG,IMG_SIZE,STYLE_IMG,CONTENT_IMG,METHOD,'histogram',NUM_EPOCHS,ADAM_LR,STYLE_WEIGHT,CONTENT_WEIGHT)
    # fn.FImg.save(h, BASE_PATH+'/output/avg_before_his.jpg')

    # i = main(VGG,IMG_SIZE,STYLE_IMG,CONTENT_IMG,METHOD,'luminance',NUM_EPOCHS,ADAM_LR,STYLE_WEIGHT,CONTENT_WEIGHT)
    # fn.FImg.save(i, BASE_PATH+'/output/avg_before_lumi.jpg')

    # j = main(VGG,IMG_SIZE,STYLE_IMG,CONTENT_IMG,'after',COLOR,NUM_EPOCHS,ADAM_LR,STYLE_WEIGHT,CONTENT_WEIGHT)
    # fn.FImg.save(j, BASE_PATH+'/output/avg_after_no.jpg')

    # k = main(VGG,IMG_SIZE,STYLE_IMG,CONTENT_IMG,'after','histogram',NUM_EPOCHS,ADAM_LR,STYLE_WEIGHT,CONTENT_WEIGHT)
    # fn.FImg.save(k, BASE_PATH+'/output/avg_after_his.jpg')

    # l = main(VGG,IMG_SIZE,STYLE_IMG,CONTENT_IMG,'after','luminance',NUM_EPOCHS,ADAM_LR,STYLE_WEIGHT,CONTENT_WEIGHT)
    # fn.FImg.save(l, BASE_PATH+'/output/avg_after_lumi.jpg')

    
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
