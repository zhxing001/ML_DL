# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 10:42:21 2018

@author: zhxing
"""

#回归和神经网络初始，用tensorflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#生成两百个随机点
x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]     #从-0.5到0.5产生200个点（均匀）
noise=np.random.normal(0,0.02,x_data.shape)
y_data=np.square(x_data)+noise
plt.plot(y_data)   

#定义place holder
x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])

#构建神经网络，回归    第一层  1*(1*10)  十个神经元
weight_L1=tf.Variable(tf.random_normal([1,10]))
bias_L1=tf.Variable(tf.zeros([1,10]))
output_L1=tf.matmul(x,weight_L1)+bias_L1

L1=tf.nn.tanh(output_L1)          #激活韩火速

#定义输出层   （1*10）*（10*1）  只需要一个输出神经元
weight_L2=tf.Variable(tf.random_normal([10,1]))
bias_L2=tf.Variable(tf.zeros([1,1]))
output_L2=tf.matmul(L1,weight_L2)+bias_L2

prediction=tf.nn.tanh(output_L2)   #激活函数

#定义代价函数和训练方法，还是用二次代价函数
loss=tf.reduce_mean(tf.square(y-prediction))

#使用梯度下降法
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session()  as  sess:
    sess.run(tf.global_variables_initializer())    #变量初始化
    for i in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
        
        #获取预测值：
    prediction_val=sess.run(prediction,feed_dict={x:x_data})
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_val,'r-',lw=5)
    plt.show()
        