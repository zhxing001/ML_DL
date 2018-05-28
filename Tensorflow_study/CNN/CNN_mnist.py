#导入包

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

mnist = input_data.read_data_sets("C:/Users/zhxing/Desktop/python/data/minist/", one_hot=True)

batch_sz=100
n_batch=mnist.train.num_examples//batch_sz

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)    #生成一个截断的分布
    return tf.Variable(initial)

def bias_variable(shape):
	initial=tf.constant(0.1,shape=shape)
	return tf.Variable(initial)
#卷积操作
def conv2d(x,W):
    '''
    """Computes a 2-D convolution given 4-D `input` and `filter` tensors.

      Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
      and a filter / kernel tensor of shape
      `[filter_height, filter_width, in_channels, out_channels]`, this op
      performs the following:

      1. Flattens the filter to a 2-D matrix with shape
         `[filter_height * filter_width * in_channels, output_channels]`.
      2. Extracts image patches from the input tensor to form a *virtual*
         tensor of shape `[batch, out_height, out_width,
         filter_height * filter_width * in_channels]`.
      3. For each patch, right-multiplies the filter matrix and the image patch
         vector.
    '''
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#最大值池化2*2的
def max_pool(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


with tf.name_scope('input'):
    x=tf.placeholder(tf.float32,[None,784],name='x_input')
    y=tf.placeholder(tf.float32,[None,10],name='y_input')
    with tf.name_scope('image_input'):
#把x改变成为4D的向量。-1代表可接受外部值。
     x_image=tf.reshape(x,[-1,28,28,1],name='image_input')

with tf.name_scope('Oonv1'):
    with tf.name_scope('W_conv1'):
        W_conv1=weight_variable([5,5,1,32])       #5乘5的卷积核，一共有32个卷积核，得到32维的特征图
    with tf.name_scope('b_conv1'):
        b_conv1=bias_variable([32])   #每一个卷积核一个偏置

#卷积+pooling
    with tf.name_scope('h_relu'):
        h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
    with tf.name_scope('pool_conv1'):
        h_pool1=max_pool(h_conv1)

#第二层卷积核和偏置
with tf.name_scope('Oonv2'):
    with tf.name_scope('W_conv2'):
        W_conv2=weight_variable([5,5,32,64])
    with tf.name_scope('b_conv2'):
        b_conv2=bias_variable([64])

#第二层卷积和输出
    with tf.name_scope('h_relu'):
        h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
    with tf.name_scope('pool_conv2'):
        h_pool2=max_pool(h_conv2)

'''
第一层卷积完成之后28*28*32，池化之后变成14*14*32
第二层卷积完成之后尺度不变14*14*64，池化之后变成7*7*64
池化改变主要是和步长有关
'''
#全连接层的权值和偏置
with tf.name_scope('fc1_LAY'):
    with tf.name_scope('W_FC1'):
        W_fc1=weight_variable([7*7*64,500])
    with tf.name_scope('b_FC1'):
        b_fc1=bias_variable([500])

#输出层扁平化
    with tf.name_scope('flat'):
        h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
    with tf.name_scope('FC1_relu'):
        h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

#dropout设置
    with tf.name_scope('keep_drop'):
        keep_drop=tf.placeholder(tf.float32)
    with tf.name_scope('dropout'):
        h_fc1_drop=tf.nn.dropout(h_fc1,keep_drop)

#第二层
with tf.name_scope('FC1_lay'):
    with tf.name_scope('FC1_W'):
        W_fc2=weight_variable([500,10])
    with tf.name_scope('FC1_b'):
        b_fc2=bias_variable([10])
#计算输出
    with tf.name_scope('prediction'):
        prediction=tf.nn.softmax(tf.matmul(h_fc1,W_fc2)+b_fc2)
#代价函数\
with tf.name_scope('loss'):
    with tf.name_scope('loss'):
        loss=tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction)
    cross_entropy=tf.reduce_mean(loss)

with tf.name_scope('train'):
    train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    with tf.name_scope('current_acc'):
        current_acc=tf.equal(tf.arg_max(y,1),tf.arg_max(prediction,1))
    with tf.name_scope('ACC'):
        accuracy=tf.reduce_mean(tf.cast(current_acc,tf.float32))
        tf.summary.scalar('accuracy1',accuracy)
merged=tf.summary.merge_all()
time_cost=0

with tf.Session()  as sess:
    sess.run(tf.global_variables_initializer())
    train_writer=tf.summary.FileWriter('logs/train',sess.graph)
    test_writer=tf.summary.FileWriter('logs/test',sess.graph)
    start=time.time()
    for i in range(10000):
        
        #train
        batch_xs,batch_ys=mnist.train.next_batch(batch_sz)
        ans=batch_xs
        sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_drop:0.5})
        summary=sess.run(merged,feed_dict={x:batch_xs,y:batch_ys,keep_drop:1.0})
        train_writer.add_summary(summary,i)

        #test
        batch_xs, batch_ys = mnist.test.next_batch(batch_sz)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_drop: 0.5})
        summary = sess.run(merged,feed_dict={x: batch_xs, y: batch_ys, keep_drop: 1})
        test_writer.add_summary(summary, i)

        if i%100==0:
            end = time.time()
            acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_drop:1})
            print("epoch:\t"+str(i)+"\t"+"Test Acc:\t"+str(acc)+"\ttime cost：\t"+str(end-start))
            start=time.time()