import tensorflow as tf
import numpy as np
import time
import math
from get_date import get_date_and_label
import os
import cv2
import matplotlib.pyplot as plt

notes=(['zhxing!','liuyang!','juchunwu!','hailong!','lyzhen!','I dont know this people!'])

#定义卷积操作
def conv2d(input,name,kh,kw,n_out,dh,dw):
	n_in=input.get_shape()[-1].value    #获取输入尺度

	with tf.name_scope(name) as scope:
		# 初始化卷积核
		kernel=tf.get_variable(scope+'w',shape=[kh,kw,n_in,n_out],dtype=tf.float32,
							   initializer=tf.contrib.layers.xavier_initializer_conv2d())
		conv = tf.nn.conv2d(input, kernel, (1, dh, dw, 1), padding='SAME')
		bias_init_val = tf.constant(0, shape=[n_out], dtype=tf.float32)
		biases = tf.Variable(bias_init_val, trainable=True, name='b')
		z=tf.nn.bias_add(conv,biases)
		relu=tf.nn.relu(z,name=scope)
		return relu


def fc(input,name,n_out):
	n_in=input.get_shape()[-1].value
	with tf.name_scope(name) as scope:
		kernel=tf.get_variable(scope+'w',shape=[n_in,n_out],dtype=tf.float32,
							   initializer=tf.contrib.layers.xavier_initializer())
		biases=tf.Variable(tf.constant(0.1,shape=[n_out],dtype=tf.float32),name='b')
		activation=tf.nn.relu_layer(input,kernel,biases,name=scope)
		return activation



with tf.name_scope('placeholder'):
	with tf.name_scope('img_holder'):
		image=tf.placeholder(tf.float32,[None,64,64,3])
	with tf.name_scope('label_holder'):
		label=tf.placeholder(tf.float32,[None,6])
	with tf.name_scope('drop'):
		drop=tf.placeholder(tf.float32)


#第一层，5*5卷积带3*3池化
conv1_out=conv2d(image,'conv1_3_3',5,5,64,1,1)
max_pool1=tf.nn.max_pool(conv1_out,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

#第二层，3*3卷积带2*2池化
conv2_out=conv2d(max_pool1,'conv2_3_3',3,3,128,1,1)
max_pool2=tf.nn.max_pool(conv2_out,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#第三层。3*3卷积带2*2池化
conv3_out=conv2d(max_pool2,'conv3_3_3',3,3,64,1,1)
max_pool3=tf.nn.max_pool(conv3_out,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

shape=max_pool3.get_shape()
flattened_shape=shape[1].value*shape[2].value*shape[3].value
resh1=tf.reshape(max_pool3,[-1,flattened_shape],name='reshpe1')


#全连接层,fc操作自带relu激活函数
fc1=fc(resh1,name='FC1',n_out=1024)
fc1_drop=tf.nn.dropout(fc1,keep_prob=drop,name='FC1_drop')

fc2=fc(fc1_drop,name='FC2',n_out=512)
fc2_drop=tf.nn.dropout(fc2,keep_prob=drop,name='FC2_drop')

fc3 = fc(fc2_drop, name='FC3', n_out=6)

softmax=tf.nn.softmax(fc3)
predictions=tf.arg_max(softmax,1)

cross_entropy=tf.reduce_mean(-tf.reduce_sum(label*tf.log(softmax),reduction_indices=[1]))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

current_accuracy=tf.equal(predictions,tf.arg_max(label,1))
accuracy=tf.reduce_mean(tf.cast(current_accuracy,tf.float32))



model_save_path='C:/Users/zhxing/Desktop/Model/model_faceDec.ckpt'
saver=tf.train.Saver()



train_time_start=time.time()
def train(model_save_path):
	# 获取数据和标签
	train_img, train_label, test_img, test_label = get_date_and_label()

	# 数据归一化，统一/255然后减掉0.5
	train_img =np.float32(train_img) / 255 - 0.5
	test_img = np.float32(test_img) / 255 - 0.5

	batch_sz = 100
	drop = 0.5
	numOfBatch = np.shape(train_img)[0] // batch_sz
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(1):
			for j in range(numOfBatch):
				start=time.time()
				image_=train_img[j*batch_sz:(j+1)*batch_sz,:,:,:]
				label_=train_label[j*batch_sz:(j+1)*batch_sz,:]

				if j%50==0:
					train_acc=sess.run(accuracy,feed_dict={image:image_,label:label_,drop:1.0})
					print('train_acc of '+str(j*50)+'batches:\t'+str(train_acc))
					end=time.time()
					print('time_cost of every 50 batches:\t'+str(end-start))

				sess.run(train_step,feed_dict={image:image_,label:label_,drop:0.5})

		saver.save(sess,model_save_path)
		train_time_end=time.time()
		print('train_time is \t'+str(train_time_end-train_time_start)+'\tsecond')

		test_accuracy=0
		for i in range(67):
			test_acc=sess.run(accuracy,feed_dict={image:test_img[i*100:(i+1)*100,:,:,:],label:test_label[i*100:(i+1)*100,:],drop:1.0})
			print('test acc of current_batch is:\t'+str(test_acc))
			test_accuracy+=test_acc

		print('test_accuracy is:\t'+str(test_accuracy/(i+1)))


#主函数入口
font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体

#存在模型则加载模型，不存在的话需要进行训练。
if os.path.exists('C:/Users/zhxing/Desktop/Model/model_faceDec.ckpt.meta'):
	Saver=tf.train.Saver()
	with tf.Session() as test_sess:
		Saver.restore(test_sess,model_save_path)

		# _,_, test_img, test_label = get_date_and_label()
		# # 数据归一化，统一/255然后减掉0.5
		# test_img = np.float32(test_img) / 255 - 0.5
		# pre=test_sess.run(predictions,feed_dict={image:test_img[0:100,:,:,:],drop:1})



		#打开摄像头，加载特征吃
		camera = cv2.VideoCapture(0)
		harr = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		n=1
		while 1:
			if(n<10000):
				success,image_face=camera.read()
				cv2.imshow('Face_detection', image_face)
				gray_img=cv2.cvtColor(image_face,cv2.COLOR_BGR2GRAY)
				faces=harr.detectMultiScale(gray_img,1.3,5)
				for f_x,f_y,f_w,f_h in faces:
					face=image_face[f_y:f_y+f_h,f_x:f_x+f_w]
					face=cv2.resize(face,(64,64))
					face_rs=np.reshape(face,[1,64,64,3])
					face_F=np.float32(face_rs)/255-0.5
					pre=test_sess.run(predictions,feed_dict={image:face_F,drop:1.0})
					cv2.rectangle(image_face,(f_x,f_y),(f_x+f_w,f_y+f_h),(0,0,255),thickness=2)
					cv2.putText(image_face,notes[pre[0]],(f_x,f_y),font,1,(0,255,0))
					#各参数依次是：照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
					cv2.imshow('Face_detection', image_face)
				key=cv2.waitKey(30) &0xff
				if key==27:
					break
			else:
				break
else:
	train(model_save_path)



















