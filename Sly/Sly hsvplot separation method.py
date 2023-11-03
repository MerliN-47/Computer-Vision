#%%
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
#%%
brown = cv.imread('D:/Synced data/Folders/SLYTHERIN/Snake Dataset/Green/snake_s124.jpg')
cv.imshow('brown',brown)
cv.waitKey(0)
cv.destroyAllWindows()
#%%
r,g,b = cv.split(brown)
fig = plt.figure()
axis = fig.add_subplot(1,1,1,projection = "3d")

pixel_colors = brown.reshape((np.shape(brown)[0]*np.shape(brown)[1],3))
norm = colors.Normalize(vmin = -1, vmax = 1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()

axis.scatter(r.flatten(),g.flatten(),b.flatten(),facecolors = pixel_colors,marker = ".")
axis.set_xlabel("red")
axis.set_ylabel("green")
axis.set_zlabel("blue")
plt.show()
#%%
hsv_brown = cv.cvtColor(brown, cv.COLOR_RGB2HSV)
#%%
h, s, v = cv.split(hsv_brown)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()
#%%

