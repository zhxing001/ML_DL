
def get_date_and_label():
	import matplotlib.pyplot as plt

	#读入数据和标签，我存了一份JPG格式的，但是这样存储是有问题的，matlab三通道拉长的话是
	#一个通道连续存储，而python是一个位置三个通道连续存储。所以最后还是改用mat型存储
	#==============================================================================
	# train_img=cv2.imread('train_img.jpg',0)
	# train_label=cv2.imread('train_label.jpg',0)
	# test_img=cv2.imread('test_img.jpg',0)
	# test_label=cv2.imread('test_label.jpg',0)
	#
	# train_img_reshape=np.reshape(train_img,[-1,64,64,3])
	# plt.imshow(train_img_reshape[1])
	#
	#==============================================================================



	import scipy.io as sio
	#使用scipy.io来加载mat文件，还是挺好用的


	train_img_mat=sio.loadmat('C:/Users/zhxing/Desktop/data/train_img1.mat')
	train_img=train_img_mat['train_img_rz']

	train_label_mat=sio.loadmat('C:/Users/zhxing/Desktop/data/train_label1.mat')
	train_label=train_label_mat['train_labels']

	test_img_mat=sio.loadmat('C:/Users/zhxing/Desktop/data/test_img1.mat')
	test_img=test_img_mat['test_img_rz']

	test_label_mat=sio.loadmat('C:/Users/zhxing/Desktop/data/test_label1.mat')
	test_label=test_label_mat['test_labels']



	return train_img,train_label,test_img,test_label
	#=====测试下数据对不对，看来是没有问题的=========================================================================
	# for i in range(0,15):
	# 	img=train_img[i,:,:,:]
	# 	plt.figure()
	# 	plt.imshow(img)
	# 	plt.title(train_label[i])
	#
	#==============================================================================
	#==============================================================================
	# for i in range(0,15):
	#     img=test_img[i,:,:,:]
	#     plt.figure()
	#     plt.imshow(img)
	#     plt.title(test_label[i])
	#==============================================================================
