# -*- coding: utf-8 -*-
"""
Created on Sat May 26 21:52:40 2018

@author: zhxing
"""
import tensorflow as tf
import numpy as np

<<<<<<< HEAD
<<<<<<< HEAD

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



import cv2
test_img=cv2.imread('1.jpg')
test_img=np.reshape(test_img,[1,64,64,3])


saver=tf.train.Saver()

with tf.Session() as sess:
	saver.restore(sess,'./model_path/model_faceDec.ckpt')
	sess.run(predictions,feed_dict={image:test_img,drop:1.0})
	
	
=======
saver=tf.train.Saver()

>>>>>>> 29a11ab86a0964e9e173562127b90483ab1c4d4b
=======
saver=tf.train.Saver()

>>>>>>> parent of f5a9c28... model

