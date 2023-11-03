#%%
import numpy as np
import cv2 as cv
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
#%%
image = cv.imread('image.jpg')
#image_rescale = rescale(image, 0.25, anti_aliasing=False)
#image_resize = resize(image, (image.shape[0] // 4, image.shape[1] // 4), anti_aliasing=True)
#image_downscale = downscale_local_mean(image,(4,3))

cv.imshow('image',image)
cv.waitKey(0)
cv.destroyAllWindows()
#%%
#Find canny images
edged = cv.Canny(image,30,50)
cv.imshow('img',edged)
cv.waitKey(0)
cv.destroyAllWindows()
#%%
#Finding the contours
_,contours,hierarchy = cv.findContours(image,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
cv.imshow('canny after contouring', image)
cv.waitKey(0)
cv.destroyAllWindows()
#%%
print(contours)
print('Numbers of contours found=' + str(len(contours)))
#%%
cv.drawContours(image,contours,-1,(0,255,0),3)
plt.imshow(image)


#%%
#Finding the contours
_,contours,hierarchy = cv.findContours(edged,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
cv.imshow('canny after contouring', edged)
cv.waitKey(0)
cv.destroyAllWindows()
#%%
print(contours)
print('Numbers of contours found=' + str(len(contours)))
#%%
cv.drawContours(image,contours,-1,(0,255,0),3)
cv.imshow('contours',image)
cv.waitKey(0)
cv.destroyAllWindows()