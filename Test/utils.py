from PIL import Image
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure as exposure
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn.functional as F
import os
import requests
from io import BytesIO
from skimage import io,transform

# show 3 images(numpy)
def show3Image(content,style,target,title1='Content Image',title2='Style Image',title3='Generated Image'):
    fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(10,5))  
    # title = 'Show style, content, generated image'
    # fig.suptitle(title, fontsize=16)   
    #Plotting content image   
    ax1.set_title(title1)
    ax1.imshow(content)  
    ax1.axis('off')  
    #Plotting style image  
    ax2.set_title(title2)
    ax2.imshow(style)
    ax2.axis('off')  
    #Plotting target image  
    ax3.set_title(title3)
    ax3.imshow(target) 
    ax3.axis('off') 
    plt.show()
    
# show 2 images(numpy)
def show2Image(content,style,title1='Content Image',title2='Style Image'):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(5,4))  
    # title = 'Show style, content, generated image'
    # fig.suptitle(title, fontsize=16)   
    #Plotting content image   
    ax1.set_title(title1)
    ax1.imshow(content)  
    ax1.axis('off')  
    #Plotting style image  
    ax2.set_title(title2)
    ax2.imshow(style)
    ax2.axis('off')  
    plt.show()

# set numpy, tensor, bytes, original size
class AImage():
    def __init__(self,path):
        img = FImg.load(path)
        self.Img = img # numpy
        self.size = self.Img.shape
        self.Tensor = im_convertT(self.Img) # tensor

# manage image
class FImg(object):
    def __init__(self):
        print('Manage one image...')
        print('1. load(path) : get numpy array image')
        print('2. show(img,title_name)')
        print('3. save(img,save_name) #save_name must have .jpg')
        print('4. ImgByte(image) :  change numpy or PIL Image to class bytes')
        print('5. resize(img,size) #size use format (224,224)')

    # load image(numpy) from path or url
    def load(path):
        if os.path.exists(path):
            print('Load image from path : ',path)
            img = Image.open(path) # PIL Image
            img = np.asarray(img) # numpy
            return img
        elif requests.get(path).status_code == 200:
            print('Load image from url : ',path)
            img = Image.open(BytesIO(requests.get(path).content))
            img = np.asarray(img)
            print(type(img))
            return img
        else:
            print('Cannot load image in ',path)
        
    # show image(tensor or numpy)
    def show(img,title_name=None):
        if type(img).__module__=='torch': img = im_convert(img)
        if title_name != 'None': plt.title(title_name)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    # save image(tensor or numpy) to jpg
    def save(img,save_name):
        if type(img).__module__=='torch': img = im_convert(img)
        plt.figure(figsize = (10,10))
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(save_name, bbox_inches='tight', pad_inches=0, format='jpg')
        # plt.show()

    # change numpy or PIL Image to class bytes
    def ImgByte(image):
        if type(image).__module__=='numpy': img = Image.fromarray(image, 'RGB')
        bytes_IO = BytesIO()
        img.save(bytes_IO, format='JPEG')
        byte_image = bytes_IO.getvalue()
        # img = Image.open(bytes_IO)
        # plt.imshow(img)
        # plt.axis('off')
        # plt.show()
        # print(type(byte_image)) # <class 'bytes'>
        return byte_image

    # resize image(tensor or numpy)    
    def resize(img,size):
        if type(img).__module__=='numpy':
            img = transform.resize(img,size)
        elif  type(img).__module__=='torch':
            img = (F.adaptive_avg_pool2d(Variable(img), size)).data
        else:
            print('Cannot resize image')
        return img

# change tensor to image(numpy)
def im_convert(tensor):  
    image=tensor.cpu().clone().detach().numpy()     
    image=image.squeeze()  
    image=image.transpose(1,2,0)  
    image=image*np.array((0.5,0.5,0.5))+np.array((0.5,0.5,0.5))  
    image=image.clip(0,1)  
    return image  

# change image(numpy) to tensor
def im_convertT(img):
    # Rescale the image
    in_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    # Convert image to tensor
    tensor = in_transform(img)
    tensor = tensor.unsqueeze(dim=0)
    return tensor

# color preservation : histogram matching, luminance only transfer
class ColorPreservation(object):
    def __init__(self):
        print('Color preservation => matching source(numpy) and reference(numpy)')
        print('1. histogramMatching(src, ref)')
        print('2. luminanceOnlyTransfer(src, ref)')

    # color histogram matching by put source(numpy) and reference(numpy)
    def histogramMatching(src, ref):
        print('Using color preservation method Histogram Matching')
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
        print('Using color preservation method Luminance Only Transfer')
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

def plotLoss(contentLoss,styleLoss,totalLoss,title='Loss'):
    x = [i for i in range(len(totalLoss))]
    plt.figure(figsize=[8,6])
    plt.plot(x, contentLoss, label="Content Loss")
    plt.plot(x, styleLoss, label="Style Loss")
    plt.plot(x, totalLoss, label="Total Loss")
    
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(title)
    plt.show()

if __name__=="__main__":
    pass