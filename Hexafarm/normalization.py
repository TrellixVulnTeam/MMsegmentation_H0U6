import numpy as np
from PIL import Image
import glob
import cv2

# img_dir = r"data\LCCV\train\*"
img_dir = r"data\LCCV\test\*"
imgs_loc = glob.glob(img_dir)

mean_cum, std_cum = 0, 0

for img_loc in imgs_loc:
    img = cv2.imread(img_loc)
    mean, std = cv2.meanStdDev(img)
    mean_cum += mean
    std_cum += std

num_imgs = len(imgs_loc)
mean = mean_cum / num_imgs
std = std_cum / num_imgs

mean = np.array([mean[2], mean[1], mean[0]])
std = np.array([std[2], std[1], std[0]])

print(mean, std)

'''
[[78.01544853]
 [70.84201877]
 [52.7681944 ]] 
 
 [[34.96587739]
 [34.51387563]
 [28.53118475]]

'''