import cv2
import os
import sys
import random
import time


#新建存储文件夹
out_dir = './my_faces'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


# 改变图片的对比度和亮度，增加图像多样性
def relight(img, alpha=1, bias=0):
    w = img.shape[1]
    h = img.shape[0]
    #image = []
    for i in range(0,w):
        for j in range(0,h):
            for c in range(3):
                tmp = int(img[j,i,c]*alpha + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j,i,c] = tmp
    return img


# 获取分类器，用的是opencv的harr默认特征
haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 打开摄像头 参数为输入流，可以为摄像头或视频文件
camera = cv2.VideoCapture('liuyang.mp4')

n = 1

while 1:
    if (n <= 10000):
        print('It`s processing %s image.' % n)
        # 读帧
        start=time.time()
        success, img = camera.read()
        cv2.imshow('face',img)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(gray_img, 1.3, 5)       #多尺度检测人脸

        for f_x, f_y, f_w, f_h in faces:
            face = img[f_y:f_y+f_h, f_x:f_x+f_w]
            face = cv2.resize(face, (64,64))
            '''
            if n % 3 == 1:
                face = relight(face, 1, 50)
            elif n % 3 == 2:
                face = relight(face, 0.5, 0)
            '''
            face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
            cv2.imshow('img', face)
            cv2.imwrite(out_dir+'/'+str(n)+'.jpg', face)
            #time_cost
            end=time.time()
            time_cost=end-start
            print('time_cost:\t'+str(time_cost))

            n+=1
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break
    else:
        break