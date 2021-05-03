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

def loadImg(path):
    img = cv.imread(path)
    print(path,type(img))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # img=Image.open(path).convert('RGB')
    # cv.imshow("Img",img)
    return img

def loadImgURL(url):
    response = requests.get(url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
    else:
        print('An error has occurred.')
    # cv.imshow("Img",img)
    return img

def showImg(img,name='Image'):
    plt.title(name)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

from skimage import io
def saveImg(img,img_path):
    if type(img).__module__=='numpy':
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # cv.imwrite(img_path, img)
        img = img.astype(np.uint8)
        io.imsave(img_path, img)
    else:
        print('Image should be numpy class.')
        save_image(img, img_path)

# change tensor to image
def im_convert(tensor):  
    image=tensor.cpu().clone().detach().numpy()     
    image=image.squeeze()  
    image=image.transpose(1,2,0)  
    image=image*np.array((0.5,0.5,0.5))+np.array((0.5,0.5,0.5))  
    image=image.clip(0,1)  
    return image  

# change image to tensor
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

#Initializing gram_matrix function for our tensor image   
def gram_matrix(tensor):  
    b,c,h,w=tensor.shape   
    tensor=tensor.view(b,c,h*w)    
    tensor_transpose = tensor.transpose(1,2) 
    return torch.bmm(tensor,tensor_transpose) / (c*h*w)

def show3Image(style,content,target,title='Show Style Content & Target Image',title1='Content Image',title2='Style Image',title3='Generated Image'):
    fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5))  
    fig.suptitle(title, fontsize=16)   
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

def histogramMatching(src, ref):
    multi = True if src.shape[-1] > 1 else False
    matched = exposure.match_histograms(src, ref, multichannel = multi)
    return matched

# def histogramEqualizeGray(img):
#     gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#     dst = cv.equalizeHist(gray)
#     return dst

def plotLoss(contentLoss,styleLoss,totalLoss,title='Loss'):
    x = [i for i in range(len(totalLoss))]
    plt.figure(figsize=[10,6])
    plt.plot(x, contentLoss, label="Content Loss")
    plt.plot(x, styleLoss, label="Style Loss")
    plt.plot(x, totalLoss, label="Total Loss")
    
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(title)
    plt.show()

# path = os.getcwd()
# STYLE_IMG_PATH = path+r'\StyleImage\Chakrabhan\0001.jpg'
# CONTENT_IMG_PATH = path+r'Test\ContentImg.jpg'
# # TARGET_IMG_PATH = path+r'\result_test1.png'
# style_img = loadImg(STYLE_IMG_PATH)
# content_img = loadImg(CONTENT_IMG_PATH)
# # target_img = loadImg(TARGET_IMG_PATH )
# showImg(style_img,'Chakrabhan Style Image')
# show3Image(style_img,content_img,style_img)
# preserved_img = histogramMatching(target_img,content_img)
# img = cv.cvtColor(preserved_img, cv.COLOR_BGR2RGB)
# saveImg(img, 'Presult_test1.png')
# show3Image(style_img,content_img,preserved_img,title="Histogram Matching",title3="Preserved Image")

# # concatanate image Horizontally
# Hori = np.concatenate((style_img,preserve_img), axis=1)
 
# cv.imshow('HORIZONTAL', Hori)
  
# cv.waitKey(0)
# cv.destroyAllWindows()

class OneImage():
    def __init__(self,path):
        img = loadImg(path)
        self.img = img
        self.Tensor = im_convertT(self.img)

    def show(self):
        showImg(self.img)

    def save(self,path):
        saveImg(self.img,path)


class ContentDataset(Dataset):
    def __init__(self,path,size=[224,224]):
        self.root_path = path
        self.path = []
        self.filename = []
        self.img = []
        for di in os.listdir(self.root_path):
            directory = self.root_path+'/'+di
            for i in os.listdir(directory):
                self.path.append(directory+'/'+i)
                self.filename.append(i)
                img = loadImg(directory+'/'+i)
                self.img.append(img)

        preprocess = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
        self.trainDataset = datasets.ImageFolder(self.root_path, transform=preprocess)
        
    def getpath(self,idx):
        return self.path[idx]
    
    def getimage(self,idx):
        return self.img[idx]
    
    def showimage(self,idx):
        img = self.img[idx]
        filename = self.filename[idx]
        showImg(img,name=filename)

    def __len__(self):
        return len(self.trainDataset)

    def dataset(self):
        return self.trainDataset

        
    