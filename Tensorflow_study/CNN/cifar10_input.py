import tensorflow as tf
import numpy as np
import time


class_num = 10
image_size = 32
img_channels = 3
iterations = 200
batch_size = 200
total_epoch = 164
weight_decay = 0.0003
dropout_rate = 0.5
momentum_rate = 0.9
log_save_path = './vgg_logs'
model_save_path = './model/'

def unpickle(file):
    with open(file, 'rb') as fo:      #以二进制格式打开文件
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_data_one(file):
    batch = unpickle(file)
    data = batch[b'data']
    labels = batch[b'labels']
    print("Loading %s : %d." % (file, len(data)))
    return data, labels


def load_data(files, data_dir, label_count):
    global image_size, img_channels
    data, labels = load_data_one(data_dir + '/' + files[0])
    for f in files[1:]:
        data_n, labels_n = load_data_one(data_dir + '/' + f)
        data = np.append(data, data_n, axis=0)
        labels = np.append(labels, labels_n, axis=0)
    labels = np.array([[float(i == label) for i in range(label_count)] for label in labels])
    data = data.reshape([-1, img_channels, image_size, image_size])
    data = data.transpose([0, 2, 3, 1])
    return data, labels


def prepare_data():
    data_dir = 'C:\\Users\zhxing\Desktop\python\data\cifar-10-batches-py'
    meta = unpickle(data_dir + '/batches.meta')         #解压meta文件
    print(meta)
    label_names = meta[b'label_names']             #labels名字
    
    label_count = len(label_names)              #多少个标签，10
    train_files = ['data_batch_%d' % d for d in range(1, 6)]      #六个文件名

    train_data, train_labels = load_data(train_files, data_dir, label_count)
    test_data, test_labels = load_data(['test_batch'], data_dir, label_count)

    print("Train data:", np.shape(train_data), np.shape(train_labels))
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")

    print("======Shuffling data======")
    indices = np.random.permutation(len(train_data))
    train_data = train_data[indices]
    train_labels = train_labels[indices]
    print("======Prepare Finished======")

    return train_data, train_labels, test_data, test_labels

train_data, train_labels, test_data, test_labels=prepare_data()

'''
#获得训练数据和测试数据，以矩阵保存；
# =============================================================================
# Train data: (50000, 32, 32, 3) (50000, 10)
# Test data : (10000, 32, 32, 3) (10000, 10)
# =============================================================================
'''


#定义权重，偏置，卷积，以及池化操作。
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

def data_pre(train,test):
	x_train=train.astype('float32')
	x_test=test.astype('float32')
	x_train[:, :, :, 0] = (x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
	x_train[:, :, :, 1] = (x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
	x_train[:, :, :, 2] = (x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])
	x_test[:, :, :, 0] = (x_test[:, :, :, 0] - np.mean(x_test[:, :, :, 0])) / np.std(x_test[:, :, :, 0])
	x_test[:, :, :, 1] = (x_test[:, :, :, 1] - np.mean(x_test[:, :, :, 1])) / np.std(x_test[:, :, :, 1])
	x_test[:, :, :, 2] = (x_test[:, :, :, 2] - np.mean(x_test[:, :, :, 2])) / np.std(x_test[:, :, :, 2])
	return x_train, x_test

with tf.name_scope('input'):
	x=tf.placeholder(tf.float32,[None,image_size,image_size,img_channels],'image')
	#32*32*3，一次可以放个图像，最终传入batch_sz
	y=tf.placeholder(tf.float32,[None,10],'labels')
with tf.name_scope('Conv1'):
    with tf.name_scope('W_conv1'):
        W_conv1=weight_variable([3,3,3,32])       #3乘3的卷积核，一共有32个卷积核，得到32维的特征图
    with tf.name_scope('b_conv1'):
        b_conv1=bias_variable([32])
    L1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)



with tf.name_scope('Conv2'):
    with tf.name_scope('W_conv2'):
        W_conv2=weight_variable([3,3,32,64])       #3*3*32的卷积核，一共有64个卷积核，得到64维的特征图
    with tf.name_scope('b_conv2'):
        b_conv2=bias_variable([64])   #每一个卷积核一个偏置
    L2_relu=tf.nn.relu(conv2d(L1,W_conv2)+b_conv2)
    L2=max_pool(L2_relu)

with tf.name_scope('Conv3'):
	with tf.name_scope('W_conv3'):
		W_conv3 = weight_variable([3, 3, 64, 64])  # 3*3*32的卷积核，一共有64个卷积核，得到64维的特征图
	with tf.name_scope('b_conv3'):
		b_conv3 = bias_variable([64])  # 每一个卷积核一个偏置
	L3_relu = tf.nn.relu(conv2d(L2, W_conv3) + b_conv3)
	L3 = max_pool(L3_relu)

with tf.name_scope('Plat'):
	L3_plat=tf.reshape(L3,[-1,8*8*64])

with tf.name_scope('FC1'):
	with tf.name_scope('W_fc1'):
		W_fc1=weight_variable([8*8*64,1024])
	with tf.name_scope('b_fc1'):
		b_fc1=bias_variable([1024])
	fc1_o=tf.nn.relu(tf.matmul(L3_plat,W_fc1)+b_fc1)


with tf.name_scope('FC2'):
	with tf.name_scope('W_fc2'):
		W_fc2=weight_variable([1024,512])
	with tf.name_scope('b_fc1'):
		b_fc2=bias_variable([512])
	fc2_o=tf.nn.relu(tf.matmul(fc1_o,W_fc2)+b_fc2)

with tf.name_scope('FC3'):
    with tf.name_scope('W_fc1'):
        W_fc3=weight_variable([512,10])
    with tf.name_scope('W_fc1'):
        b_fc3=bias_variable([10])
#计算输出
with tf.name_scope('prediction'):
    prediction=tf.nn.softmax(tf.matmul(fc2_o,W_fc3)+b_fc3)


with tf.name_scope('loss'):
	loss=tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction)
	cross_entropy=tf.reduce_mean(loss)

with tf.name_scope('train'):
	train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    
    current_acc=tf.equal(tf.arg_max(y,1),tf.arg_max(prediction,1))
    accuracy=tf.reduce_mean(tf.cast(current_acc,tf.float32))
	
	#对上面的结果取个均值就是准确率（所有的1加起来除以总数）。



with tf.Session()  as  sess:
	data_train,data_test=data_pre(train_data,test_data)
	sess.run(tf.global_variables_initializer())	     #所有变量初始化
	for epoch in range(1):
		for batch in range(50000//batch_size):
			start=time.time()
			x_tr=data_train[0:batch*batch_size]
			y_lr=train_labels[0:batch*batch_size]

			sess.run(train_step,feed_dict={x:x_tr,y:y_lr})
			end=time.time()
			print('batch_cost_time:\t'+str(end-start))
<<<<<<< HEAD
<<<<<<< HEAD
		acc = sess.run(accuracy, feed_dict={x: data_test, y: test_labels})
		print('epoch:\t' + str(epoch) + '\tACC:\t' + str(acc))
                
            
            
=======
=======
>>>>>>> ef1119c198b3e6b74dde9824c0fa53b02444d135
			
		acc = sess.run(accuracy, feed_dict={x: test_data, y: test_labels})
		print('epoch:\t' + str(epoch) + '\tACC:\t' + str(acc))
		
		
<<<<<<< HEAD
>>>>>>> ef1119c198b3e6b74dde9824c0fa53b02444d135
=======
>>>>>>> ef1119c198b3e6b74dde9824c0fa53b02444d135
print('done')
