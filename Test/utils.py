from PIL import Image
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure as exposure
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torchvision.utils import save_image
import os
import requests
from io import BytesIO
from skimage import io

# load image(numpy) from path image in BGR color space
def loadImg(path):
    if os.path.exists(path):
        img = cv.imread(path,1)
        # print(path,type(img))
        # plt.imshow(img)
        # plt.show()
        return img
    else:
        print('Cannot load image in ',path)

# load image(numpy) from url image in RGB color space
def loadImgURL(url):
    response = requests.get(url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        # plt.imshow(img)
        # plt.show()
        return img
    else:
        print('An error has occurred.')

# show image(numpy) and normalize
def showImg(img,name=None):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = np.array(img/255).clip(0,1)
    if name != 'None': plt.title(name)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# save image(numpy)
def saveImg(img,img_path):
    if type(img).__module__=='numpy':
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = img.astype(np.uint8)
        io.imsave(img_path, img)
    else:
        print('Image should be numpy class.')
        save_image(img, img_path)

# show 3 images(numpy)
def show3Image(style,content,target,title1='Content Image',title2='Style Image',title3='Generated Image'):
    style = cv.cvtColor(style, cv.COLOR_BGR2RGB)
    content = cv.cvtColor(content, cv.COLOR_BGR2RGB)
    target = cv.cvtColor(target, cv.COLOR_BGR2RGB)
    fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5))  
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
    plt.close()

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





if __name__=="__main__":
    get_ipython().run_line_magic('matplotlib', 'inline')

    BASE_PATH = os.getcwd()
    STYLE_IMG = r'.\StyleImage\Chakrabhan\0001.jpg'
    CONTENT_IMG = r'.\ContentImg.jpg' 
    style = loadImg(STYLE_IMG)
    content = loadImg(CONTENT_IMG)
    

    new = ColorPreservation.match('his',style,content)
    show3Image(style,content,new)

    new =  ColorPreservation.match('lumi',style,content)
    show3Image(style,content,new)