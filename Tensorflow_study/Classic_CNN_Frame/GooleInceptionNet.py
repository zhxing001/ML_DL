#这个网络和前面的几个网络还是有非常大的不一样的


import tensorflow as tf
import tensorflow.contrib.slim as slim
trunc_normal=lambda stddev:tf.truncated_normal_initializer(0.0,stddev)


'''
下面定义函数inception_v3_arg_scope，用来生成网络中常用的函数的默认参数，比如卷积激活函数，权重初始化形式，标准化器等。
设置L2正则化的默认权重系数为0.0004，标准差stddev=0.1，参数batch_norm_var_collection的默认值为moving_vars，接下来定义batch
normalization的参数字典，衰减系数0.9997，epsilon为0.001，updata_collctions为tf.GraphKeys.UPDATE_OPS,然后字典variable_collection 
中的beta和gamma均设置为None。 Moving_mean和moving_variance设置为batch_norm_var_collection
'''

def inception_v3_arg_scope(weitht_decay=0.00004,
						   stddev=0.1,
						   batch_norm_var_collection='moving_vars'):

	batch_norm_params={
		'decay':0.9997,
		'epsilon':0.001,
		'updates_collections':tf.GraphKeys.UPDATE_OPS,
		'variables_collections':{
			'beta':None,
			'gamma':None,
			'moving_mean':[batch_norm_var_collection],
			'moving_variance':[batch_norm_var_collection]
		}
	}

	with slim.arg_scope([slim.conv2d,slim.fully_connected],
						weights_regularizer=slim.l2_regularizer(weitht_decay)):
		with slim.arg_scope(
				[slim.conv2d],
				weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
				activation_fn=tf.nn.relu,
				normalizer_fn=slim.batch_norm,
				normalizer_params=batch_norm_params) as sc:
			return sc

'''
定义inception_v3_base函数，生成V3网络的卷积部分，参数INPUT为输入图片的sensor，scope包含为包含了函数默认参数的
环境，定义一个字典表end_points，用来保存某些关键阶段供以后使用，接着再使用slim.agr_scope,对slim.conv2d,slim.max_poolsd
和slim_avg_pool2d这三个函数的参数设置默认值，
'''

def inception_v3_base(inputs,scope=None):
	end_points={}             #保存一些关键节点
	with tf.variable_scope(scope,'InceptionV3',[inputs]):
		with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride=1,padding='VALID'):
			net=slim.conv2d(inputs,32,[3,3],stride=2,scope='Conv3-3')
			net=slim.conv2d(net,32,[3,3],scope='conv3-3-1')
			net=slim.conv2d(net,64,[3,3],padding='SAME',scope='conv3-3-2')
			net=slim.max_pool2d(net,[3,3],stride=2,scope='MaxPool3')
			net=slim.conv2d(net,80,[1,1],scope='Conv1-1')
			net=slim.conv2d(net,192,[3,3],scope='Conv3-3-3')
			net=slim.max_pool2d(net,[3,3],stride=2,scope='Maxpool3-3')
			'''
			上一部分主要是卷积部分，和图中给的稍有不同，连续用了三个3*3的卷积层，默认stride=1，默认不加padding，
			在第四层用了一个[3,3]的重叠最大池化层，完了用[1,1]的卷积信息来组合通道之间的信息，然后加一个3-3的卷积层和
			[3,3]的池化层。
			输入图像尺寸是299-299-3，经过卷积层和池化层的尺寸分别会变为：
			3*3/2 nopadding    149*149
			3*3/1 nopadding    147*147
			3*3/1 padding      147*147
			3*3/2 nopadding pool   73*73
			1*1/1 nopadding     73*73
			3*3/1 nopadding     71*71
			3*3/2 nopadding pool    35*35
			最后一个共有192个通道，这个实现和给出的图片是不太一样的。
			'''
		with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride=1,padding='SAME'):
			with tf.variable_scope('Mixed_5b'):   #第一个Inception模块组
				with tf.variable_scope('branch_0'):
					branch_0=slim.conv2d(net,64,[1,1],scope='Conv5b01-1')
				with tf.variable_scope('branch_1'):
					branch_1=slim.conv2d(net,48,[1,1],scope='conv5b11-1')
					branch_1=slim.conv2d(branch_1,64,[5,5],scope='conv5b15-5')
				with tf.variable_scope('branch_2'):
					branch_2=slim.conv2d(net,64,[1,1],scope='conv5b21-1')
					branch_2=slim.conv2d(branch_2,96,[3,3],scope='conv5n23-3_1')
					branch_2=slim.conv2d(branch_2, 96, [3, 3], scope='conv5b23-3')
				with tf.variable_scope('branch_3'):
					branch_3=slim.avg_pool2d(net,[3,3],scope='avg_pool5b33-3')
					branch_3=slim.conv2d(branch_3,32,[1,1],scope='conv2d5b31-1')
	
				net=tf.concat([branch_0,branch_1,branch_2,branch_3],3)
				'''
				第一个INCEPTION modlue，三个分支分别是：
				1.  1*1       输出64通道
				2.  1*1-5*5   输出64通道
				3.  1*1-3*3-3*3  输出96通道
				4.  3*3平均池化，1*1卷积   输出32通道
				所有步长都是1，且padding，所以最后的尺寸是没有变化的，最后一句把所有的按照通道加起来，所以一共是：
				35*35*(64+64+96+32)=35*35*256的数据，共256个通道。
				'''
			with tf.variable_scope('Mixed_5c'):  # 第一个Inception模块组
				with tf.variable_scope('branch_0'):
					branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv5c01-1')
				with tf.variable_scope('branch_1'):
					branch_1 = slim.conv2d(net, 48, [1, 1], scope='conv2d5c11-1')
					branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='conv25c1d5-5')
				with tf.variable_scope('branch_2'):
					branch_2 = slim.conv2d(net, 64, [1, 1], scope='conv2d5c21-1')
					branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='conv2d5c23-3_1')
					branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='conv2d5c23-3_2')
				with tf.variable_scope('branch_3'):
					branch_3 = slim.avg_pool2d(net, [3, 3], scope='avg_pool5c33-3')
					branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='conv2d5c31-1')

				net = tf.concat([branch_0, branch_1, branch_2, branch_3],3)
				'''
				这个inception和上面的唯一的不同是最后一个通道是输出64个通道。所以一共是有288个通道
				'''

			with tf.variable_scope('Mixed_5d'):  # 第一个Inception模块组
				with tf.variable_scope('branch_0'):
					branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv5d01-1')
				with tf.variable_scope('branch_1'):
					branch_1 = slim.conv2d(net, 48, [1, 1], scope='conv2d5d11-1')
					branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='conv2d5d15-5')
				with tf.variable_scope('branch_2'):
					branch_2 = slim.conv2d(net, 64, [1, 1], scope='conv2d5d21-1')
					branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='conv2d5d23-3')
					branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='conv2d5d23-3_2')
				with tf.variable_scope('branch_3'):
					branch_3 = slim.avg_pool2d(net, [3, 3], scope='avg_pool5d33-3')
					branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='conv2d5d31-1')

				net = tf.concat([branch_0, branch_1, branch_2, branch_3],3)
				'''
				这个和上面是完全一样的，参数也是一样，所以输出也是288个通道。
				'''
			with tf.variable_scope('Mixed_6a'):  # 第一个Inception模块组
				with tf.variable_scope('branch_0'):
					branch_0 = slim.conv2d(net, 384, [3, 3],stride=2,padding='VALID',scope='Conv6a03-3')
				with tf.variable_scope('branch_1'):
					branch_1 = slim.conv2d(net, 64, [1, 1], scope='conv2d6a11-1')
					branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='conv2d6a13-3_1')
					branch_1 = slim.conv2d(branch_1, 96, [3, 3],stride=2,padding='VALID',scope='conv2d6a13-3')
				with tf.variable_scope('branch_2'):
					branch_2 = slim.max_pool2d(net, [3, 3],stride=2, padding='VALID',scope='max_pool6a23-3')
				net = tf.concat([branch_0, branch_1, branch_2],3)
				'''
				35*35,均不加padding且步长为2，所以尺度上会被压缩，压缩为：
				17*17共有384+96+256(这个256哪里来的，第一个inception组的最后一个输出是288啊)个通道，共768个通道
				'''
			with tf.variable_scope('Mixed_6b'):  # 第一个Inception模块组
				with tf.variable_scope('branch_0'):
					branch_0 = slim.conv2d(net, 192, [1, 1],scope='Conv6b01-1')
				with tf.variable_scope('branch_1'):
					branch_1 = slim.conv2d(net, 128, [1, 1], scope='conv2d6b11-1')
					branch_1 = slim.conv2d(branch_1, 128, [1,7], scope='conv2d6b11-7')
					branch_1 = slim.conv2d(branch_1, 192, [7, 1],scope='conv2d6b17-1')
				with tf.variable_scope('branch_2'):
					branch_2 = slim.conv2d(net, 128, [1, 1], scope='conv2d6b21-1')
					branch_2 = slim.conv2d(branch_2, 128, [7,1], scope='conv2d6b27-1')
					branch_2 = slim.conv2d(branch_2, 128, [1, 7],scope='conv2d6b21-7')
					branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='conv2d6b27-1_2')
					branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='conv2d6b21-7_2')
				with tf.variable_scope('branch_3'):
					branch_3 = slim.avg_pool2d(net, [3, 3],scope='avg_pool6b33-3')
					branch_3=slim.conv2d(branch_3,192,[1,1],scope='conv2d6b3')
				net = tf.concat([branch_0, branch_1, branch_2,branch_3],3)
				'''
				17*17的进入这个inception，所有步长为1，且不加padding，输出军事192通道，所以合并起来
				是17*17*（192*4）=17*17*768的输出尺寸。
				'''

				'''
				c,d,e是完全一样的，和b不同的是第2，3分支的前面的通道数变为160了，tensor经过这几个inception后的
				维度并没有发生变化，但是每次特征都重新提取，最终6e的输出还是17*17*768
				'''
			with tf.variable_scope('Mixed_6c'):  # 第一个Inception模块组
				with tf.variable_scope('branch_0'):
					branch_0 = slim.conv2d(net, 192, [1, 1],scope='Conv6c01-1')
				with tf.variable_scope('branch_1'):
					branch_1 = slim.conv2d(net, 160, [1, 1], scope='conv2d6c11-1')
					branch_1 = slim.conv2d(branch_1, 160, [1,7], scope='conv2d6c11-7')
					branch_1 = slim.conv2d(branch_1, 192, [7, 1],scope='conv2d6c17-1')
				with tf.variable_scope('branch_2'):
					branch_2 = slim.conv2d(net, 160, [1, 1], scope='conv2d6c21-1')
					branch_2 = slim.conv2d(branch_2, 160, [7,1], scope='conv2d6c27-1')
					branch_2 = slim.conv2d(branch_2, 160, [1, 7],scope='conv2d6c21-7')
					branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='conv2d6c27-1_2')
					branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='conv2d6c21-7_2')
				with tf.variable_scope('branch_3'):
					branch_3 = slim.avg_pool2d(net, [3, 3],scope='avg_pool6c33-3')
					branch_3=slim.conv2d(branch_3,192,[1,1],scope='conv2d6c3')
				net = tf.concat([branch_0, branch_1, branch_2,branch_3],3)


			with tf.variable_scope('Mixed_6d'):  # 第一个Inception模块组
				with tf.variable_scope('branch_0'):
					branch_0 = slim.conv2d(net, 192, [1, 1],scope='Conv6d01-1')
				with tf.variable_scope('branch_1'):
					branch_1 = slim.conv2d(net, 160, [1, 1], scope='conv2d6d11-1')
					branch_1 = slim.conv2d(branch_1, 160, [1,7], scope='conv2d6d11-7')
					branch_1 = slim.conv2d(branch_1, 192, [7, 1],scope='conv2d6d17-1')
				with tf.variable_scope('branch_2'):
					branch_2 = slim.conv2d(net, 160, [1, 1], scope='conv2d6d21-1')
					branch_2 = slim.conv2d(branch_2, 160, [7,1], scope='conv2d6d27-1')
					branch_2 = slim.conv2d(branch_2, 160, [1, 7],scope='conv2d6d21-7')
					branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='conv2d6d27-1_2')
					branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='conv2d6d21-7_2')
				with tf.variable_scope('branch_3'):
					branch_3 = slim.avg_pool2d(net, [3, 3],scope='avg_pool6d33-3')
					branch_3=slim.conv2d(branch_3,192,[1,1],scope='conv2d6d3')
				net = tf.concat([branch_0, branch_1, branch_2,branch_3],3)

			with tf.variable_scope('Mixed_6e'):  # 第一个Inception模块组
				with tf.variable_scope('branch_0'):
					branch_0 = slim.conv2d(net, 192, [1, 1],scope='Conv6e01-1')
				with tf.variable_scope('branch_1'):
					branch_1 = slim.conv2d(net, 160, [1, 1], scope='conv2d6e11-1')
					branch_1 = slim.conv2d(branch_1, 160, [1,7], scope='conv2d6e11-7')
					branch_1 = slim.conv2d(branch_1, 192, [7, 1],scope='conv2d6e17-1')
				with tf.variable_scope('branch_2'):
					branch_2 = slim.conv2d(net, 160, [1, 1], scope='conv2d6e21-1')
					branch_2 = slim.conv2d(branch_2, 160, [7,1], scope='conv2d6e27-1')
					branch_2 = slim.conv2d(branch_2, 160, [1, 7],scope='conv2d6e21-7')
					branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='conv2d6e27-1_2')
					branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='conv2d6e21-7_2')
				with tf.variable_scope('branch_3'):
					branch_3 = slim.avg_pool2d(net, [3, 3],scope='avg_pool3-36e3')
					branch_3=slim.conv2d(branch_3,192,[1,1],scope='conv2d6e3')
				net = tf.concat([branch_0, branch_1, branch_2,branch_3],3)

			end_points['Mixed_6e']=net
			#我们把minxed_6e的结果保存起来，用作加权最后分类，维度为17*17*768

			with tf.variable_scope('Mixed_7a'):
				with tf.variable_scope('branch_0'):
					branch_0=slim.conv2d(net,192,[1,1],scope='conv2d7a01-1')
					branch_0=slim.conv2d(branch_0,320,[3,3],stride=2,padding='VALID',scope='conv2d7a03-3')
				with tf.variable_scope('branch_1'):
					branch_1=slim.conv2d(net,192,[1,1],scope='conv2d7a11-1')
					branch_1=slim.conv2d(branch_1,192,[1,7],scope='conv2d7a11-7')
					branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='conv2d7a17-1')
					branch_1 = slim.conv2d(branch_1, 192, [3, 3],stride=2,padding='VALID', scope='conv2d7a13-3')
				with tf.variable_scope('branch_2'):
					branch_2=slim.max_pool2d(net,[3,3],stride=2,padding='VALID',scope='maxpool_7a23-3')

				net=tf.concat([branch_0,branch_1,branch_2],3)
				#第一个inception输出为8*8的，因为不加padding且步长为2，一共输出320+192+768=1280，所以8*8*1280

			with tf.variable_scope('Mixed_7b'):
				with tf.variable_scope('branch_0'):
					branch_0=slim.conv2d(net,320,[1,1],scope='conv2d7b01-1')

				with tf.variable_scope('branch_1'):
					branch_1=slim.conv2d(net,384,[1,1],scope='conv2d7b11-1')
					branch_1=tf.concat([
						slim.conv2d(branch_1,384,[1,3],scope='conv2d7b11-3'),
						slim.conv2d(branch_1,384,[3,1],scope='conv2d7b13-1')
						],3)
				with tf.variable_scope('branch_2'):
					branch_2=slim.conv2d(net,448,[1,1],scope='conv2d7b21-1')
					branch_2=slim.conv2d(branch_2,384,[3,3],scope='conv2d7b23-3')
					branch_2=tf.concat([
						slim.conv2d(branch_2,384,[1,3],scope='conv2d7b21-3'),
						slim.conv2d(branch_2,384,[3,1],scope='conv2d7b23-1')
						],3)
				with tf.variable_scope('branch_3'):
					branch_3=slim.avg_pool2d(net,[3,3],scope='avg_pool7b3')
					branch_3=slim.conv2d(branch_3,192,[1,1],scope='conv2d7b31-1')
				net=tf.concat([branch_0,branch_1,branch_2,branch_3],3)
				#上一个输出是8*8*1280，这里四个分支，带padding和步长为1，四个分支的输出分别是：
				#320+768+768+192=2048，中间在分支的中间又加了分支，所以尺寸是8*8*2048


			with tf.variable_scope('Mixed_7c'):
				with tf.variable_scope('branch_0'):
					branch_0=slim.conv2d(net,320,[1,1],scope='conv2d7c01-1-1')

				with tf.variable_scope('branch_1'):
					branch_1=slim.conv2d(net,384,[1,1],scope='conv2d7c11-1')
					branch_1=tf.concat([
						slim.conv2d(branch_1,384,[1,3],scope='conv2d7c11-3'),
						slim.conv2d(branch_1,384,[3,1],scope='conv2d7c13-1')
						],3)
				with tf.variable_scope('branch_2'):
					branch_2=slim.conv2d(net,448,[1,1],scope='conv2d7c21-1')
					branch_2=slim.conv2d(branch_2,384,[3,3],scope='conv2d3-3')
					branch_2=tf.concat([
						slim.conv2d(branch_2,384,[1,3],scope='conv2d7c21-3'),
						slim.conv2d(branch_2,384,[3,1],scope='conv2d7c23-1')
						],3)
				with tf.variable_scope('branch_3'):
					branch_3=slim.avg_pool2d(net,[3,3],scope='avg_pool7c3')
					branch_3=slim.conv2d(branch_3,192,[1,1],scope='conv2d7c31-1')
				net=tf.concat([branch_0,branch_1,branch_2,branch_3],3)
				#7b和7c是完全一样的，所以最后的输出依然是8*8*2048
			return net,end_points
	#这是最后的一个inception module的结果了，我们返回它

#至此V3网络的核心卷积层就结束了，最终的输出是8*8*2048，下面来实现全局平均池化，softmax和logits

def inception_v3(inputs,
				 num_classes=1000,
				 is_training=True,    #是否训练过程
				 dropout_keep=0.8,
				 prediction_fn=slim.softmax,   #分类函数，使用softmax
				 spatial_squeeze=True,        #是否去除维度为1的维度，比如5*4*1转换为5*4
				 reuse=None,              #是否对变量重复利用
				 scope='InceptionV3'):
	with tf.variable_scope(scope,'InceptionV3',[inputs,num_classes],reuse=reuse) as scope:
		with slim.arg_scope([slim.batch_norm,slim.dropout],is_training=is_training):
			net,end_points=inception_v3_base(inputs,scope=scope)

			with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride=1,padding='SAME'):
				aux_logits=end_points['Mixed_6e']
				with tf.variable_scope('aux_logits'):
					aux_logits=slim.avg_pool2d(
						aux_logits,[5,5],stride=3,padding='VALID',scope='AVGpool_m5-5'
					)
					aux_logits=slim.conv2d(aux_logits,128,[1,1],scope='Conv2dm1-1')
					aux_logits = slim.conv2d(aux_logits,
											 768, [5, 5],
											 weights_initializer=trunc_normal(0.01),
											 padding='VALID',
											 scope='Conv2dm5-5')
					aux_logits = slim.conv2d(aux_logits,
											 num_classes, [1, 1],
					                         activation_fn=None,
					                         normalizer_fn=None,
											 weights_initializer=trunc_normal(0.01),
											 scope='Conv2dmx1-1')

					if spatial_squeeze:
						aux_logits=tf.squeeze(aux_logits,[1,2],name='squeeze')

					end_points['aux_logits']=aux_logits

			with tf.variable_scope('logtics'):
				net=slim.avg_pool2d(net,[8,8],padding='VALID',scope='avgPOOL8-8')
				net=slim.dropout(net,keep_prob=dropout_keep,scope='dropout1b')
				end_points['PreLogtis']=net
				logits=slim.conv2d(net,num_classes,[1,1],activation_fn=None,normalizer_fn=None,scope='conv2d1-1')
				if spatial_squeeze:
					logits=tf.squeeze(logits,[1,2],name='spatialSqueeze')
			end_points['logits']=logits
			end_points['prediction']=prediction_fn(logits,scope='predictions')
	return logits,end_points


import time
from datetime import datetime
import math

def time_tensorflow_run(session,target,info_string):
	num_step_burn_in=10
	total_duration=0
	total_duration_squared=0.0
	for i in range(num_batch+num_step_burn_in):
		start_time=time.time()
		_= session.run(target)
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





batch_size=32
height,width=299,299
num_batch=100

inputs=tf.random_uniform((batch_size,height,width,3))
with slim.arg_scope(inception_v3_arg_scope()):
	logits,end_points=inception_v3(inputs,is_training=False)

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
num_batches=100

time_tensorflow_run(sess,logits,'forward')

