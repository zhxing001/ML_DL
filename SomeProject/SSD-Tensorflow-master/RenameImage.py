import os
import cv2
import time
import matplotlib.pyplot as plt
#原图路径和保存图片的路径
imgPath="C:\\Users\\zhxing\\Desktop\\VOCtrainval_06-Nov-2007\\VOCdevkit\\MyDate\\JPEGImages\\img\\"
savePath="C:\\Users\\zhxing\\Desktop\\VOCtrainval_06-Nov-2007\\VOCdevkit\\MyDate\\JPEGImages\\"
imgList=os.listdir(imgPath)

for i in range(1,len(imgList)):
    img=cv2.imread(imgPath+imgList[i])
    str_tmp="000000"+str(i)
    cv2.imwrite(savePath+str_tmp[-6:]+".jpg",img)      #后六位命名
print("done!!")
