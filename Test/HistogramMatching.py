import cv2 as cv
import matplotlib.pyplot as plt


# def rescaleFrame(frame, scale = 0.75, width = 224, height = 224):
#     if scale != 0:
#         width = int(frame.shape[1] * scale)
#         height = int(frame.shape[0] * scale)
#     dimension = (width,height)
#     return cv.resize(frame, dimension, interpolation=cv.INTER_AREA)


# imgC = cv.imread(r'Test\ContentImg.jpg')
# # cv.imshow('Content Image',imgC)
# # resized_imgC = rescaleFrame(imgC)
# # cv.imshow('Content Image',resized_imgC)

# imgS = cv.imread(r'StyleImage\Chakrabhan\0001.jpg')
# # cv.imshow('Style Image',imgS)
# # resized_imgS = rescaleFrame(imgS,0)
# # cv.imshow('Style Image',resized_imgS)

# grayC = cv.cvtColor(imgC,cv.COLOR_BGR2GRAY)
# # cv.imshow('Content Gray Image',grayC)

# grayS = cv.cvtColor(imgS,cv.COLOR_BGR2GRAY)
# # cv.imshow('Style Gray Image',grayS)

# # GrayScale Histogram
# grayC_hist = cv.calcHist([grayC],[0],None,[256],[0,256])
# plt.figure()
# plt.title('GrayScale Content Histogram')
# plt.xlabel('Bins')
# plt.ylabel('# of pixels')
# plt.plot(grayC_hist)
# plt.xlim([0,256])
# plt.show()

# grayS_hist = cv.calcHist([grayS],[0],None,[256],[0,256])
# plt.figure()
# plt.title('GrayScale Style Histogram')
# plt.xlabel('Bins')
# plt.ylabel('# of pixels')
# plt.plot(grayS_hist)
# plt.xlim([0,256])
# plt.show()


# # histogram equalize
# img = cv.imread(r'StyleImage/Chakrabhan/0001.jpg')
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# dst = cv.equalizeHist(gray)
# cv.imshow('Source Image',gray)
# cv.imshow('Equalized Image',dst)
# cv.waitKey(0)

# # histogram matching
# src = imgS
# ref = imgC
# from skimage import exposure
# multi = True if src.shape[-1] > 1 else False
# matched = exposure.match_histograms(src, ref, multichannel = multi)

# cv.imshow('Source imgS',src)
# cv.imshow('Ref imgC',ref)
# cv.imshow('Match',matched)
# cv.waitKey(0)

# (fig, axs) = plt.subplots(nrows = 3, ncols = 3, figsize= (8,8))
# for (i, image) in enumerate((src, ref, matched)):
#     image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
#     for (j, color) in enumerate(("red","green","blue")):
#         # compute histogram for each color channel
#         (hist, bins) = exposure.histogram(image[...,j],source_range = "dtype")
#         axs[j,i].plot(bins, hist/hist.max())

#         # compute cumulative distribution for each color channel
#         (cdf, bins) = exposure.cumulative_distribution(image[...,j])
#         axs[j,i].plot(bins, cdf)

#         axs[j,0].set_ylabel(color)
# axs[0,0].set_title("Source")
# axs[0,1].set_title("Reference")
# axs[0,2].set_title("Matching")

# plt.tight_layout()
# plt.show()

# Watermarking
from imutils import paths
watermark = cv.imread(r'C:\Users\USER\Desktop\Senior_project\Model\Test\ContentImg.jpg',cv.IMREAD_UNCHANGED)
cv.imshow('Image',watermark)
(wH,wW)= watermark.shape[:2]
print(watermark.shape)
(B, G, R) = cv.split(watermark)
B = cv.bitwise_and(B,B,mask=A)
G = cv.bitwise_and(G,G,mask=A)
R = cv.bitwise_and(R,R,mask=A)
watermark = cv.merge([B,G,R,A])
cv.imshow('ImageB',watermark)
cv.waitKey(0)