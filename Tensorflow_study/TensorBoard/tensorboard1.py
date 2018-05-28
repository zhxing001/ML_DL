# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 10:24:40 2018

@author: zhxing
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("C:/Users/zhxing/Desktop/python/data/minist/", one_hot=True)

# 定义两个变量
batch_size = 100  # 训练的时候是一次性放入，并不是一张一张来训练，以一个矩阵的形式放入
# 计算移动多少个批次
n_batch = mnist.train.num_examples // batch_size  # 批次数量


# 函数，传入参数可以计算这个参数的标准差，平均值，标准差等等。
def variable_summaries(var):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)  # 记录这个值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('stddev', var)


# 定义placeholder和命名空间
with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, [None, 784], name='x-')  # 28*28的图片
    y = tf.placeholder(tf.float32, [None, 10], name='y-')  # 标签

with tf.name_scope('layer'):
    with tf.name_scope('weight'):
        W1 = tf.Variable(tf.zeros([784, 10]), name='W1')
        variable_summaries(W1)  # 分析一下权值和偏置
    with tf.name_scope('biases'):
        b1 = tf.Variable(tf.zeros([10]), name='B1')
        variable_summaries(b1)
    with tf.name_scope("w_plus_b"):
        output_L1_tmp = tf.matmul(x, W1) + b1
    # output_L1=tf.nn.softmax(output_L1_tmp)      #激活函数用的是softmax
    with tf.name_scope("prediction"):
        prediction = tf.nn.softmax(output_L1_tmp)
# 构建一个简单的神经网络，输入层784，然后一个输出层，10个标签，没有隐藏层
# =============================================================================
# '''# =============================================================================
# W2=tf.Variable(tf.zeros([1,10]))
# b2=tf.Variable(tf.zeros([10]))
# output_L2_tmp=tf.matmul(output_L1_tmp,W2)+b2
# prediction=tf.nn.softmax(output_L2_tmp)
# # =====我不知道为什么，这里加的隐藏是不工作的，刚开始看还不知道问题在哪，以后再说===
# '''#二次代价函数
# =============================================================================


# loss=tf.reduce_mean(tf.square(y-prediction))          #二次代价函数
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))  # softmax代价函数
    tf.summary.scalar('loss', loss)  # 只有一个值，记录下来就可以了。
# 梯度下降法
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 求准确率
with  tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    # 这求得是预测位置的index，如果相等返回的是true
    # 这两个实际比较的是预测值和我们的判断值是否是一样的，结果存放在一个bool型的列表中

    # 把bool型转换为float，然后求平均值就是准确率
    with tf.name_scope('accuracy1'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('acc', accuracy)

merged = tf.summary.merge_all()

with tf.Session()  as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('log/', sess.graph)
    for epoch in range(50):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys})  # 数据和标签分别存储

        writer.add_summary(summary, epoch)  # 写入summary和运行周期
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Inter  " + str(epoch) + ",    Testing Accuracy：  " + str(acc))







