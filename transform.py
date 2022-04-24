from os import path
import glob
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import *
from ipywidgets import interact
from torchvision import datasets, models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

index = 0
for file in glob.glob("./session2/2/*.bmp"):
   print(file)
   img = cv2.imread(file)
   plt.imshow(img,'gray')
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   source = cv2.equalizeHist(gray)


   grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   gaussianBlur = cv2.GaussianBlur(grayImage, (3,3), 0)

# Scharr
   x = cv2.Scharr(gaussianBlur, cv2.CV_32F, 1, 0)
   y = cv2.Scharr(gaussianBlur, cv2.CV_32F, 0, 1)
   absX = cv2.convertScaleAbs(x)
   absY = cv2.convertScaleAbs(y)
   Scharr = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

   images = Scharr
   plt.imshow(images,'gray')
   plt.xticks([]),plt.yticks([])
   plt.savefig('./new/'+str(index)+'.jpg')
# plt.title(images)
   plt.figure(figsize=(1,1))
   # plt.show()

   plt.clf()
   plt.cla()
   plt.close()
   index = index +1

