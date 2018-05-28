# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 21:56:28 2018

@author: zhxing
"""
#回归--------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#使用numpy生成200个随机点
x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]   #增加一个维度
print(x_data)
noise=np.random.normal(0,0.02,x_data.shape)
y_data=np.square(x_data)+noise            #二次，加噪声

#定义两个占位符
x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])


#构建神经网络
Weight_L1=tf.Variable(tf.random_normal([1,10]))
biases_L1=tf.Variable(tf.zeros([1,10]))
wx_plus=tf.multiply(x,Weight_L1)+biases_L1
L1=tf.nn.tanh(wx_plus)      #激活层

#定义输出层
Weight_L2=tf.Variable(tf.random_normal([10,1]))    #十个神经元到一个输出
biases_L2=tf.Variable(tf.zeros([1,1]))
wx_plus_L2=tf.matmul(L1,Weight_L2)+biases_L2
predition=tf.nn.tanh(wx_plus_L2)

#代价函数
loss=tf.reduce_mean(tf.square(y-predition))
#梯度下降
train=tf.train.GradientDescentOptimizer(0.2).minimize(loss)
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(2000):
        sess.run(train,feed_dict={x:x_data,y:y_data})
    
    prediction_value=sess.run(predition,feed_dict={x:x_data})
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_value,'r-',lw=3)
    

        

        
    
