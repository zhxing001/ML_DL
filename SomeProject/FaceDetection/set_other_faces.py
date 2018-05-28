# -*- codeing: utf-8 -*-
import sys
import os
import cv2


input_dir = 'C:/Users/zhxing/Desktop/python/data/face/'
output_dir = './other_faces'
size = 64

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

index = 1

for (path, dirnames, filenames) in os.walk(input_dir):
    for filename in filenames:
        if filename.endswith('.jpg'):
            print('Being processed picture %s' % index)
            img_path = path+'/'+filename
            # 从文件读取图片
            img = cv2.imread(img_path)
            # 转为灰度图片
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 使用detector进行人脸检测 dets为返回的结果
            faces = haar.detectMultiScale(gray_img, 1.3, 5)


            for f_x, f_y, f_w, f_h in faces:
                face = img[f_y:f_y + f_h, f_x:f_x + f_w]
                face = cv2.resize(face, (64, 64))

                cv2.imshow('image',face)
                # 保存图片
                cv2.imwrite(output_dir+'/'+str(index)+'.jpg', face)
                index += 1

            key = cv2.waitKey(30) & 0xff
            if key == 27:
                sys.exit(0)
