# -*- coding: utf-8 -*-
"""
@author: zhxing
这个程序主要是学习使用tensorboard来的
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

max_step=500
learning_rate=0.001
dropout=0.9

mnist = input_data.read_data_sets("C:/Users/zhxing/Desktop/python/data/minist/", one_hot=True)

sess=tf.InteractiveSession()

'''
为了在tensorboard里面展示节点名称，我们设计网络的时候经常会使用with tf.name_scope限定命名空间。
在这个命名空间下的所有节点会被重新自动命名。比如我们定义的name_scope是 input那么其下的节点就会被定义
为input/xxx。这里我们先定义两个placeholder，并将输入的一维数据变为28*28的图片来存储，这样就可以使用
tf.summary.image来将图片数据汇总给tensorboard展示。
'''

# 定义placeholder和命名空间
with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, [None, 784],name='x-')  # 28*28的图片
    y_ = tf.placeholder(tf.float32, [None, 10],name='y-')  # 标签

#把向量reshape成图片以供显示
with tf.name_scope('input_resahpe'):
	image_shaped_input=tf.reshape(x,[-1,28,28,1])
	tf.summary.image('input',image_shaped_input,10)


#定义变量的初始化方式，截断的正态分布和常量
def weight_variable(shape):
	return tf.Variable(tf.truncated_normal(shape,stddev=0.1))

def bias_variable(shape):
	return tf.Variable(tf.constant(0.1,shape=shape))


#接着定义对变量数据的汇总函数，计算变量的均值，方差，最大和最小，对这些变量使用tf.summary.scalar进行汇总，同时使用
#tf.summary.histogram来直接记录变量Var的直方图数据

def variable_summary(var):
	with tf.name_scope('summary'):
		mean=tf.reduce_mean(var)
		tf.summary.scalar('mean',mean)
		with tf.name_scope('stddev'):
			stddev=tf.squeeze(tf.reduce_mean(tf.square(var-mean)))
		tf.summary.scalar('stddev',stddev)
		tf.summary.scalar('max',tf.reduce_max(var))
		tf.summary.scalar('Min',tf.reduce_min(var))
		tf.summary.histogram('His',var)

'''
然后我们设计一个多层的神经网络来悬链数据，在每一层中对模型参数进行数据汇总，所以我们创建一个进行数据汇总
的函数，nn_layer，这个函数输入数据，输入数据维度，输出数据维度和层的名称，激活函数默认使用relu。
'''

def nn_layer(input_tensor,input_dim,output_dim,layer_name,act=tf.nn.relu):
	with tf.name_scope(layer_name):
		with tf.name_scope('weight'):
			weight=weight_variable([input_dim,output_dim])
			variable_summary(weight)
		with tf.name_scope('bias'):
			bias=bias_variable([output_dim])
			variable_summary(bias)
		with tf.name_scope('W_plus_x_add_b'):
			preactivate=tf.matmul(input_tensor,weight)+bias
			tf.summary.histogram('preactivate',preactivate)

		activation=act(preactivate,name='activation')      #激活函数relu
		tf.summary.histogram('activations',activation)
		return activation


#使用上面定义好的layer来创建一层神经网络，输入维度是748，输出的隐藏节点是500，再创建一个Dropout层。

hidden1=nn_layer(x,784,500,'layer1')

with tf.name_scope('dropout'):
	keep_prob=tf.placeholder(tf.float32)
	tf.summary.scalar('dopour_keep',keep_prob)
	dropped=tf.nn.dropout(hidden1,keep_prob)

y=nn_layer(dropped,500,10,'layer2',act=tf.identity)    #全映射相当于是


with tf.name_scope('cross_entropy'):
	diff=tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_)
	with tf.name_scope('total'):
		cross_entropy=tf.reduce_mean(diff)
tf.summary.scalar('cross_entropy',cross_entropy)


with tf.name_scope('train'):
	train_step=tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
with tf.name_scope('accuracy'):
	with tf.name_scope('corrent_prediction'):
		correct_prediction=tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))
	with tf.name_scope('accuracy'):
		accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

tf.summary.scalar('accuracy',accuracy)

#前面有许多tf.summary操作，一个一个做太麻烦了，所以tensorflow提供给了merge_all操作

merged=tf.summary.merge_all()

train_writer=tf.summary.FileWriter('train/',sess.graph)
test_writer=tf.summary.FileWriter('test/')

tf.global_variables_initializer().run()            #初始化所有变量


#定义frrd的损失函数，先判断标记训练，如果训练的话标记为True，从mnist中获取一个batch的样本
#如果标记为False，则获取测试数据，并把dropout设置为1.
def feed_dict(train):
	if train:
		xs,ys=mnist.train.next_batch(100)
		k=dropout
	else:
		xs,ys=mnist.test.images,mnist.test.labels
		k=1.0
	return {x:xs,y_:ys,keep_prob:k}

#最后执行世纪的训练，测试及日志记录的操作
saver=tf.train.Saver()
for i in range(max_step):
	if i%10==0:
		summary,acc=sess.run([merged,accuracy],feed_dict=feed_dict(False))
		test_writer.add_summary(summary,i)
		print('accuracy at step\t'+str(i)+'is:\t'+str(acc))
	else:
		summary,_=sess.run([merged,train_step],feed_dict=feed_dict(True))
		train_writer.add_summary(summary,i)
		
train_writer.close()
test_writer.close()


