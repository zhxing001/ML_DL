import numpy as np
import tensorflow as tf

#creat data
x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3

#creat tensorflow structure start
Weight=tf.Variable(tf.random_uniform([1],-1.0,1.0))   #初始化权重和偏置
biases=tf.Variable(tf.zeros([1]))          #初始化偏重
y=Weight*x_data+biases                   #目标函数

loos=tf.reduce_mean(tf.square(y-y_data))    #损失函数，这里基本是最小二乘
optimizer=tf.train.GradientDescentOptimizer(0.3)  #学习率
train=optimizer.minimize(loos)   #训练
init=tf.initialize_all_variables()       #初始化,如果有定义变量，一定要这个
#####-----------------------------

sess=tf.Session()
sess.run(init)      #激活神经网络，很重要

for step in range(300):    #训练，每20次打印以此信息
    sess.run(train)
    if step%20==0:
        print(step,sess.run(Weight),sess.run(biases),sess.run(loos))

