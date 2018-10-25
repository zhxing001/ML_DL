import os
import cv2
import time
import matplotlib.pyplot as plt
imgPath="C:\\Users\\zhxing\\Desktop\\DJ\\"
imgList=os.listdir(imgPath)

for i in range(1,len(imgList)):
    img=cv2.imread(imgPath+imgList[i])
    plt.imshow(img)
    str_tmp="000000"+str(i)
    cv2.imwrite(imgPath+str_tmp[-6:]+".jpg",img)      #后六位命名
print("done!!")
