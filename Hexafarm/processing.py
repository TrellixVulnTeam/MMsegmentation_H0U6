import glob
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image 

ann_list = glob.glob(r"data\LCCV\ann\*")
img_list = glob.glob(r"data\LCCV\train\*")
img = Image.open(img_list[0])
ann = Image.open(ann_list[0])

fig, axs = plt.subplots(2)
axs[0].imshow(img)
axs[1].imshow(ann)
# plt.show()

mini = r"data\LCCV\ann\A3plant013_rgb.png"
mini_img = Image.open(mini)
breakpoint()