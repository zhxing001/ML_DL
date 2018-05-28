'''
VGG-16的简单实现和测试
2018/4/27
'''

#导入库
from datetime import datetime
import math
import time
import tensorflow as tf


'''
VGG16包含了许多层的卷积，所以我们先写一个函数，用来创建卷积层并把本层的参数放入参数列表，
函数输入：tensor，以及这一层的名字，name
kh,kw分别是卷积核的高和宽，n_out是卷积核的个数即输出通道数。
dh,dw分别是卷积的步长。
p：参数列表，使用get_shape()[-1].value来获取输入的通道数（这获得的应该是最后一维的数目）
'''

def conv_op(input_op,name,kh,kw,n_out,dh,dw,p):
	n_in=input_op.get_shape()[-1].value     #输入维度

	with tf.name_scope(name) as scope:
		kernel=tf.get_variable(scope+"W",shape=[kh,kw,n_in,n_out],
							   dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
		conv=tf.nn.conv2d(input_op,kernel,(1,dh,dw,1),padding='SAME')      #卷积，padding
		bias_intit_val=tf.constant(0.1,shape=[n_out],dtype=tf.float32)
		biases=tf.Variable(bias_intit_val,trainable=True,name='b')
		z=tf.nn.bias_add(conv,biases)     #加上偏置
		activation=tf.nn.relu(z,name=scope)       #线性整流函数
		p+=[kernel,biases]        #参数列表加起来
		return activation         #返回

#下面定义全连接层的函数，基本和上面卷积网络是差不多的。

def fc_op(input_op,name,n_out,p):
	n_in=input_op.get_shape()[-1].value

	with tf.name_scope(name) as scope:
		kernel=tf.get_variable(scope+'W',shape=[n_in,n_out],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
		biases=tf.Variable(tf.constant(0.1,shape=[n_out],dtype=tf.float32),name='b')
		activation=tf.nn.relu_layer(input_op,kernel,biases,name=scope)
		p+=[kernel,biases]
		return activation

#定义最大池化层的创建函数，mpool_op
def mpool_op(input_op,name,kh,kw,dh,dw):
	return tf.nn.max_pool(input_op,ksize=[1,kh,kw,1],strides=[1,dh,dw,1],padding='SAME',name=name)

#我们创建完三个函数后，就可以创建VGG-16的主要结构了，分为6个部分，前5段为卷积网络，最后一段是全连接网络，
#第一个卷积层的输入尺寸是224*224*3，输出是224*224*64，卷积是3*3*3的卷积核。

def inference_op(input_op,keep_prob):
	p=[]

	conv1_1=conv_op(input_op,name='conv1_1',kh=3,kw=3,n_out=64,dh=1,dw=1,p=p)        #3*3*3的卷积核，共64个
	conv1_2=conv_op(conv1_1,name='conv1_2',kh=3,kw=3,n_out=64,dh=1,dw=1,p=p)
	pool1=mpool_op(conv1_2,name='pool1',kh=2,kw=2,dh=2,dw=2)     #2*2的卷积核，步长为2
	#第一层：卷积核3*3*3*64, 3*3*63*64。  变为224*224*64
	#经过maxpool之后变为 112*112*64

	conv2_1=conv_op(pool1,name='conv2_1',kh=3,kw=3,n_out=128,dh=1,dw=1,p=p)
	conv2_2=conv_op(conv2_1,name='conv2_2',kh=3,kw=3,n_out=128,dh=1,dw=1,p=p)
	pool2=mpool_op(conv2_2,name='pool2',kh=2,kw=2,dh=2,dw=2)
	#第二层卷积核3*3*64,3*3*128   尺寸变为112*112*64
	#池化之后变为56*56*128

	conv3_1 = conv_op(pool2, name='conv3_1', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
	conv3_2 = conv_op(conv3_1, name='conv3_2', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
	conv3_3 = conv_op(conv3_2, name='conv3_3', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
	pool3 = mpool_op(conv3_3, name='pool3', kh=2, kw=2, dh=2, dw=2)
	#第三层堆积了三个3*3的卷积核，不同的是输出通道是256，而最大池化层是保持不变的，这样的话
	#pool3输出的尺寸是28*28*256

	conv4_1 = conv_op(pool3, name='conv4_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
	conv4_2 = conv_op(conv4_1, name='conv4_2', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
	conv4_3 = conv_op(conv4_2, name='conv4_3', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
	pool4 = mpool_op(conv4_3, name='pool4', kh=2, kw=2, dh=2, dw=2)
	#第四层卷积同样是堆积了3*3的卷积核，输出通道为512，池化层不变，这样的话最终尺寸变为：
	#14*14*512

	conv5_1 = conv_op(pool4, name='conv5_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
	conv5_2 = conv_op(conv5_1, name='conv5_2', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
	conv5_3 = conv_op(conv5_2, name='conv5_3', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
	pool5 = mpool_op(conv5_3, name='pool5', kh=2, kw=2, dh=2, dw=2)
	#第五层卷积核前面的第四层基本是一样的，但是卷积核的个数没有增加，处于最大池化的作用，这里的尺寸就会变成
	#7*7*512
	'''
	看了网络结构的卷积部分，其实规律性很强，逐层减伤特征图尺寸（height*width），逐层增加特征图维度。
	'''


	#把7*7*512的向量reshape成单行的。
	shp=pool5.get_shape()
	flattened_shape=shp[1].value*shp[2].value*shp[3].value
	resh1=tf.reshape(pool5,[-1,flattened_shape],name='resh1')

	#全连接层，加dropout
	fc6=fc_op(resh1,name='fc6',n_out=4096,p=p)
	fc6_drop=tf.nn.dropout(fc6,keep_prob,name='fc6_drop')

	#全连接层
	fc7 = fc_op(fc6_drop, name='fc7', n_out=4096, p=p)
	fc7_drop = tf.nn.dropout(fc7, keep_prob, name='fc7_drop')

	#全连接层
	fc8 = fc_op(fc7_drop, name='fc8', n_out=1000, p=p)
	softmax=tf.nn.softmax(fc8)
	predictions=tf.arg_max(softmax,1)
	return predictions,softmax,fc8,p

#最终把predictions，softmax以及FC8及参数列表统一返回，这样VGG-16就基本构架完成了，下面我们来写测试函数



def time_tensorflow_run(session,target,feed,info_string):
	num_step_burn_in=10
	total_duration=0
	total_duration_squared=0.0
	for i in range(num_batch+num_step_burn_in):
		start_time=time.time()
		_= session.run(target,feed_dict=feed)
		duration=time.time()-start_time
		if i>=num_step_burn_in:
			if not i%10:  #每10个输出一次结果
				print('time:'+str(datetime.now())+'\tstep:\t'+str(i-num_step_burn_in)+'\t time_cost:\t'+str(duration))

		total_duration+=duration
		total_duration_squared+=duration*duration
	mn=total_duration/num_batch
	vr=total_duration_squared/num_batch-mn*mn
	sd=math.sqrt(vr)
	print('time:'+str(datetime.now())+'\tstep:\t'+str(i-num_step_burn_in)+'\t'+info_string+'\t'+'ave_time_cost:\t'+str(mn)+'\tsd:\t'+str(sd))


def run_benchmark():
	with tf.Graph().as_default():
		image_sz=224
		#用随机生成图像来模拟前传递过程
		images=tf.Variable(tf.random_normal([batch_size,
											 image_sz,
											 image_sz,
											 3],dtype=tf.float32,
											stddev=0.1 ))
		keep_prob=tf.placeholder(tf.float32)
		predictions,softmax,fc8,p=inference_op(images,keep_prob)

		init=tf.global_variables_initializer()
		sess=tf.Session()
		sess.run(init)

		time_tensorflow_run(sess,predictions,{keep_prob:1.0},'Foreard')
		object=tf.nn.l2_loss(fc8)
		grad=tf.gradients(object,p)
		time_tensorflow_run(sess,grad,{keep_prob:0.5},'Forward-back')


batch_size=32
num_batch=100
run_benchmark()




