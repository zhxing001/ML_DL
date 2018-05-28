#实现AlexNet,基本是按照tensorflow那本书来写的，然后我加详细的注释。


#导入库文件
from datetime import datetime
import math
import time
import tensorflow as tf

#batch和num_batch设置
batch_size=32
batch_num=100

#定义一个函数，输出每一层的结构 t是一个tensor，作为输入
def print_activations(t):
	print(t.op.name,'',t.get_shape().as_list)


'''
接下来是设计AlexNet的结构，我们定义一个函数（inferen），接受image作为输入，
返回最后一层pool(第五个池化层)，以及所有需要训练的模型参数，这个函数比较大，
包括多个卷积层和池化层。每一层我们利用name_scope命名。
'''
def inference(image):
	parameters=[]     #存参数

	#第一层，11*11*3的卷积核，一共有64个，原文是96个，
	with tf.name_scope('conv1') as scope:
		kernel=tf.Variable(tf.truncated_normal([11,11,3,64],dtype=tf.float32,stddev=0.1),name='weights')
		conv=tf.nn.conv2d(image,kernel,[1,4,4,1],padding='SAME')   #四个维度上的步长，分别是batch_num,height,width,纵向一般为1
		biases=tf.Variable(tf.constant(0.0,shape=[64],dtype=tf.float32),trainable=True,name='biases')
		bias=tf.nn.bias_add(conv,biases)
		conv1=tf.nn.relu(bias,name=scope)
		print_activations(conv1)            #打印tensor参数
		parameters+=[kernel,biases]         #参数存储

	lrn1=tf.nn.lrn(conv1,bias=1.0,alpha=0.001/9,beta=0.75,name='lrn1')   #local response normalization--局部响应标准化
	pool1=tf.nn.max_pool(lrn1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool1')
	print_activations(pool1)
    #最大值pool，池化核3*3，池化步长2*2



    #第二层和第一层基本一样，不同的是卷积核和步长设置
	with tf.name_scope('conv2') as scope:
		kernel=tf.Variable(tf.truncated_normal([5,5,64,192],dtype=tf.float32,stddev=0.1),name='weights')
		conv=tf.nn.conv2d(pool1,kernel,[1,1,1,1],padding='SAME')   #四个维度上的步长，分别是batch_num,height,width,纵向一般为1
		biases=tf.Variable(tf.constant(0.0,shape=[192],dtype=tf.float32),trainable=True,name='biases')
		bias=tf.nn.bias_add(conv,biases)
		conv2=tf.nn.relu(bias,name=scope)
		print_activations(conv2)            #打印tensor参数
		parameters+=[kernel,biases]

	lrn2 = tf.nn.lrn(conv2, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn2')  # local response normalization--局部响应标准化
	pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
	print_activations(pool2)

	with tf.name_scope('conv3') as scope:
		kernel=tf.Variable(tf.truncated_normal([3,3,192,384],dtype=tf.float32,stddev=0.1),name='weights')
		conv=tf.nn.conv2d(pool2,kernel,[1,1,1,1],padding='SAME')   #四个维度上的步长，分别是batch_num,height,width,纵向一般为1
		biases=tf.Variable(tf.constant(0.0,shape=[384],dtype=tf.float32),trainable=True,name='biases')
		bias=tf.nn.bias_add(conv,biases)
		conv3=tf.nn.relu(bias,name=scope)
		print_activations(conv3)            #打印tensor参数
		parameters+=[kernel,biases]

	with tf.name_scope('conv4') as scope:
		kernel=tf.Variable(tf.truncated_normal([3,3,384,256],dtype=tf.float32,stddev=0.1),name='weights')
		conv=tf.nn.conv2d(conv3,kernel,[1,1,1,1],padding='SAME')   #四个维度上的步长，分别是batch_num,height,width,纵向一般为1
		biases=tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32),trainable=True,name='biases')
		bias=tf.nn.bias_add(conv,biases)
		conv4=tf.nn.relu(bias,name=scope)
		print_activations(conv4)            #打印tensor参数
		parameters+=[kernel,biases]

	with tf.name_scope('conv2') as scope:
		kernel=tf.Variable(tf.truncated_normal([3,3,256,256],dtype=tf.float32,stddev=0.1),name='weights')
		conv=tf.nn.conv2d(conv4,kernel,[1,1,1,1],padding='SAME')   #四个维度上的步长，分别是batch_num,height,width,纵向一般为1
		biases=tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32),trainable=True,name='biases')
		bias=tf.nn.bias_add(conv,biases)
		conv5=tf.nn.relu(bias,name=scope)
		print_activations(conv5)            #打印tensor参数
		parameters+=[kernel,biases]

	pool5=tf.nn.max_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool5')

	print_activations(pool5)

	return pool5,parameters



#三个全链接层，4096-4096-1000，接受一个pool层，返回1*1000个向量(评分)

def FC(pool,parameters):    #全连接层函数，接受卷积层最后池化输出的tensor
	with tf.name_scope('FC'):
		with tf.name_scope('FC1'):
			pool_x=tf.reshape(pool,[-1,7*7*256])
			W_FC1=tf.Variable(tf.truncated_normal([7*7*256,4096],dtype=tf.float32,stddev=0.1),name='W_FC1')
			b_1=tf.Variable(tf.constant(0.01,shape=[4096]),dtype=tf.float32,trainable=True,name='b_1')
			FC1=tf.nn.relu(tf.matmul(pool_x,W_FC1)+b_1)
			parameters+=[W_FC1,b_1]
			print_activations(FC1)

		with tf.name_scope('FC2'):
			W_FC2=tf.Variable(tf.truncated_normal([4096,4096],dtype=tf.float32,stddev=0.1,name='W_FC2'))
			b_2=tf.Variable(tf.constant(0.01,shape=[4096]),dtype=tf.float32,trainable=True,name='b_2')
			FC2=tf.nn.relu(tf.matmul(FC1,W_FC2)+b_2)
			parameters += [W_FC2, b_2]
			print_activations(FC2)

		with tf.name_scope('FC3'):
			W_FC3=tf.Variable(tf.truncated_normal([4096,1000],dtype=tf.float32,stddev=0.1,name='W_FC3'))
			b_3=tf.Variable(tf.constant(0.01,shape=[1000]),dtype=tf.float32,trainable=True,name='b_2')
			FC3=tf.nn.relu(tf.matmul(FC2,W_FC3)+b_3)
			parameters += [W_FC3, b_3]
			print_activations(FC3)

	return FC3,parameters



'''
session 是tensorflow的session，第二个参数是需要评测的运算算子，第三个变量是测试的名称
我们通过batch_num+num_step_burn_in来计算次数，使用time.time()来计算时间，每轮Session.run(target
来执行，初始热身的num_step_burn_in次迭代后，每十轮显示当前迭代所需要的时间，同时每轮将total_duration
和total_duration_squared累加，以便后面计算每轮平均耗时和耗时标准差。
'''
def time_tensorflow_run(session,target,info_string):
	num_step_burn_in=10
	total_duration=0.0
	total_duration_squared=0.1
	for i in range(batch_num+num_step_burn_in):
		start_time=time.time()
		_=session.run(target)
		duration=time.time()-start_time
		if(i>=num_step_burn_in):
			if not i%10:          #每十个显示一次
				print('%s:step %d, duration=%.3f' %(datetime.now(),i-num_step_burn_in,duration))
			total_duration+=duration
			total_duration_squared+=duration*duration
	mn=total_duration/batch_num      #平均时间
	vr=total_duration_squared/batch_num-mn*mn        #计算方差
	sd=math.sqrt(vr)         #计算标准差
		#print('sd:'+str(sd))
		#print(sd)

	print(str(datetime.now())+'\t'+info_string+'\t'+'\t:'+'Ave_time:'+str(mn)+'\tsd:\t'+str(sd))

'''
接下来我们定义主函数 run_benchmark,首先我们使用 with tf.Graph.as_default()来定义默认的Graph来方便后面使用，
首先我们先不使用ImageNet来进行训练，只是测试其前馈和反馈的耗时，我们使用tf.randon_normal来随机生成一些图像数据，
然后使用前面的inference和FC函数来构建整个AlexNet网络，得到一个输出层和两个参数（卷积参数和全脸阶层参数）,接下来
我们利用tf.sesion()来创建新的session并初始化所有参数。
'''

def run_benchmark():
	with tf.Graph().as_default():
		image_size=224
		images=tf.Variable(tf.random_normal(
											[batch_size,
			 								image_size,
											image_size,
											3],               #batch_sz和图像大小224*224*3
											dtype=tf.float32,
											stddev=1e-1
											))
		pool5,parameters=inference(images)
		FC3,parameters=FC(pool5,parameters)
		init=tf.global_variables_initializer()
		sess=tf.Session()
		sess.run(init)
		'''
		下面是会模型的评测，直接使用time_tensorflow_run来统计运行时间，传入的target是FC3，即全连接的最后一个输出层，然后进行反馈
		即训练过程的评测，和前馈不同的是我们需要给最后的输出设置一个loss，一般的loss需要用到数据损失和模型损失，我们这里不传入labels，所以
		只计算模型损失，使用L2来计算，
		'''
		time_tensorflow_run(sess,FC3,'forward')
		objecttive=tf.nn.l2_loss(FC3)
		grad=tf.gradients(objecttive,parameters)
		time_tensorflow_run(sess,grad,'forward-backward')

run_benchmark()






