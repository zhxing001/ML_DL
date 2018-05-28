# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 15:53:28 2018

@author: zhxing
"""

#简单版本的手写体识别
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data

mnist=input_data.read_data_sets("C:/Users/zhxing/Desktop/python/data/minist/",one_hot=True)

#定义两个变量
batch_size=100        #训练的时候是一次性放入，并不是一张一张来训练，以一个矩阵的形式放入
#计算移动多少个批次
n_batch=mnist.train.num_examples//batch_size    #批次数量

#定义placeholder
x=tf.placeholder(tf.float32,[None,784])     #28*28的图片
y=tf.placeholder(tf.float32,[None,10])      #标签

#构建一个简单的神经网络，输入层784，然后一个输出层，10个标签，没有隐藏层

W1=tf.Variable(tf.zeros([784,10]))
b1=tf.Variable(tf.zeros([10]))
output_L1_tmp=tf.matmul(x,W1)+b1
#output_L1=tf.nn.softmax(output_L1_tmp)      #激活函数用的是softmax
prediction = tf.nn.softmax(output_L1_tmp)
# =============================================================================
# '''# =============================================================================
# W2=tf.Variable(tf.zeros([1,10]))
# b2=tf.Variable(tf.zeros([10]))
# output_L2_tmp=tf.matmul(output_L1_tmp,W2)+b2
# prediction=tf.nn.softmax(output_L2_tmp)
# # =====我不知道为什么，这里加的隐藏是不工作的，刚开始看还不知道问题在哪，以后再说===
# '''#二次代价函数
# =============================================================================


loss=tf.reduce_mean(tf.square(y-prediction))

#梯度下降法
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#初始化变量
init=tf.global_variables_initializer()

#求准确率
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))     
#这求得是预测位置的index，如果相等返回的是true
#这两个实际比较的是预测值和我们的判断值是否是一样的，结果存放在一个bool型的列表中

#把bool型转换为float，然后求平均值就是准确率
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session()  as sess:
    sess.run(init)
    for epoch in range(40):
        for batch in range(n_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)     
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})  #数据和标签分别存储

        acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Inter  "+str(epoch)+",    Testing Accuracy：  "+ str(acc))

        
            



