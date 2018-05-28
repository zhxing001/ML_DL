# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
#创建一个常量

#2-1:会话定义，常量使用---------------------------------------------------------
m1=tf.constant([[3,3]])
m2=tf.constant([[2],[3]])
#矩阵乘法
product=tf.matmul(m1,m2)

#定义一个会话
sess=tf.Session()
#sees.run会执行product，然后会执行matmul
result=sess.run(product)
print(result)
sess.close()
#python 必须加waitkey才可以用

#这样写法的好处就是不用执行关闭操作了
with tf.Session() as sess:
    result=sess.run(product)
    print(result)
    
#2-2:变量使用-------------------------------------------------------------------
x=tf.Variable([[1,2]])
a=tf.constant([[3,3]])

sub=tf.subtract(x,a)
add=tf.add(x,sub)
init=tf.global_variables_initializer()           #初始化所有变量

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))

#创建一个变量，初始化为0    
state=tf.Variable(0,name='conter')
#相加
new_value=tf.add(state,1)
#赋值
update=tf.assign(state,new_value)
init=tf.global_variables_initializer()    
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))
        
#2-3 fetch and feed -----------------------------------------------------------
#Fetch  可以在会话中运行多个op,run里用中括号括起来
input1=tf.constant(3.0)
input2=tf.constant(5.0)
input3=tf.constant(2.0)

add=tf.add(input1,input2)
mul=tf.multiply(input1,add)

with tf.Session() as sess:
    result=sess.run([add,mul])
    print(result)

#feed  在启动会话的时候以字典形式传入值
#创建占位符
input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)

output=tf.multiply(input1,input2)

with tf.Session() as sess:
    #feed的数据以字典的形式传入
    print(sess.run(output,feed_dict={input1:[7],input2:[4]}))
    
#2-4 简单示例------------------------------------------------------------------
import numpy as np
#使用numpy生成100个随机点,一个线性样本
x_data=np.random.rand(100)
y_data=x_data*0.2+0.4

#构造一个线性模型
b=tf.Variable(0.)
k=tf.Variable(0.)
y=k*x_data+b

#先定义一个二次代价函数,误差平方，求一个均值，相当于是最小二乘法
loss=tf.reduce_mean(tf.square(y-y_data))

#定义梯度下降法来定义一个优化器
optimizer=tf.train.GradientDescentOptimizer(0.1)

#定义一个最小化代价函数
train=optimizer.minimize(loss)

init=tf.global_variables_initializer()    #初始化变量

with tf.Session()  as sess1:
    sess1.run(init)
    for step in range(200):
        sess1.run(train)
        if step%20 == 0:
            print(step,'\t',sess1.run([k,b]))
        
        