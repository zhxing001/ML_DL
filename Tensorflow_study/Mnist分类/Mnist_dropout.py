# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 11:22:13 2018

@author: zhxing

1. 使用截断的高斯函数进行权重和偏置初始化
2. 使用交叉熵作为损失函数
3. 使用学习率退火算法
4. 4层神经网络，两个隐藏层
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data
import time

mnist=input_data.read_data_sets("C:/Users/zhxing/Desktop/python/data/minist/",one_hot=True)

#定义两个变量
batch_size=100        #训练的时候是一次性放入，并不是一张一张来训练，以一个矩阵的形式放入
#计算移动多少个批次
n_batch=mnist.train.num_examples//batch_size    #批次数量

#定义placeholder
x=tf.placeholder(tf.float32,[None,784])     #28*28的图片
y=tf.placeholder(tf.float32,[None,10])      #标签
keep_prob=tf.placeholder(tf.float32)
lr=tf.Variable(0.001,dtype=tf.float32)       #学习率

#构建一个简单的神经网络，输入层784，然后一个输出层，10个标签，没有隐藏层
W1=tf.Variable(tf.truncated_normal([784,500],stddev=0.1))
b1=tf.Variable(tf.zeros([500])+0.1)
L1=tf.nn.tanh(tf.matmul(x,W1)+b1)           #双曲正切激活函数
L1_drop=tf.nn.dropout(L1,keep_prob)         #drop参数


W2=tf.Variable(tf.truncated_normal([500,300],stddev=0.1))
b2=tf.Variable(tf.zeros([300])+0.1)
L2=tf.nn.tanh(tf.matmul(L1_drop,W2)+b2)           #双曲正切激活函数
L2_drop=tf.nn.dropout(L2,keep_prob)         #drop参数


W3=tf.Variable(tf.truncated_normal([300,100],stddev=0.1))
b3=tf.Variable(tf.zeros([100])+0.1)
L3=tf.nn.tanh(tf.matmul(L2_drop,W3)+b3)           #双曲正切激活函数
L3_drop=tf.nn.dropout(L3,keep_prob)         #drop参数

W4=tf.Variable(tf.truncated_normal([100,10],stddev=0.1))
b4=tf.Variable(tf.zeros([10])+0.1)
prediction=tf.nn.softmax(tf.matmul(L3_drop,W4)+b4)
# =============================================================================
# '''# =============================================================================
# W2=tf.Variable(tf.zeros([1,10]))
# b2=tf.Variable(tf.zeros([10]))
# output_L2_tmp=tf.matmul(output_L1_tmp,W2)+b2
# prediction=tf.nn.softmax(output_L2_tmp)
# # =====我不知道为什么，这里加的隐藏是不工作的，刚开始看还不知道问题在哪，以后再说===
# '''#二次代价函数
# =============================================================================


#loss=tf.reduce_mean(tf.square(y-prediction))         
#二次代价函数

#使用交叉熵代价函数
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))  #softmax代价函数
#梯度下降法
train_step=tf.train.AdamOptimizer(lr).minimize(loss)

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
        start=time.time()
        sess.run(tf.assign(lr,0.001*(0.95**epoch)))       #学习率退火
        for batch in range(n_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)     
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1})  #数据和标签分别存储
        end=time.time();
        print("time:\t"+str(end-start))
        learning_rate=sess.run(lr)
        test_acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1})
    train_acc=sess.run(accuracy,feed_dict={x:mnist.train.images,y:mnist.train.labels,keep_prob:1})
    print("Inter  "+str(epoch)+",    Testing Accuracy：  "+ str(test_acc)+"\t"+ "Learning Rate：  "+ str(learning_rate))
        
        

'''
dropout可以有效减少过拟合现象，特别是当模型比较复杂而我们的训练数据又比较少的时候
这样做可以有效提高测试准确率，有效减小过拟合现象。
'''