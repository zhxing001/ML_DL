# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 21:03:31 2017

@author: zhxing
"""

# -*- coding: utf-8 -*-
#knn,在Nn的基础上修改就好了

import pickle as p
import matplotlib.pyplot as plt
import numpy as np

class K_NearestNeighbor(object):
    def __init__(self):
        pass
    def train(self,X,y):
        self.Xtr=X
        self.ytr=y
    def predict(self,X,k):
        num_test=X.shape[0]
        #保证输出和输入类型相同
        Ypred=np.zeros(num_test,dtype=self.ytr.dtype)
        index_k=np.zeros(k,dtype=self.ytr.dtype)   #用来存得到的最小的k个距离
        for i in range(num_test):
            #对第i章图片在训练集中找最相似的
            distance=np.sum(np.abs(self.Xtr-X[i,:]),axis=1)   #算差的绝对值之和，和Xtr里面的每一个都去做
            index_sort=np.argsort(distance)     #得到排序，然后选择最前面的几个就行了
            
            index_k=self.ytr[index_sort[:k]]    #这里的标号还要投影到k上看到底是哪些标签，然后去统计标签的个数，选取最大的
            #print(index_k)
            Ypred[i]=self.ytr[np.argmax(np.bincount(index_k))]     #label，得到对应的label
            #bincount是统计每个书出现的次数，而且和他的索引是对应的，再用argmax把索引取出来就行了
        return Ypred
            

def load_CIFAR_batch(filename):
    with open(filename,'rb') as f:
        datadict=p.load(f,encoding='latin1')
        X=datadict['data']
        Y=datadict['labels']
        Y=np.array(Y)         #载入的是list类型，编程array类型的
        return X,Y
    
def load_CIFAR_labels(filename):
    with open(filename,'rb') as f:
        label_names=p.load(f,encoding='latin1')
        names=label_names['label_names']
        return names
    
    
label_names =  load_CIFAR_labels("C:/Users/zhxing/Desktop/python/data/cifar-10-batches-py/batches.meta")   #这里面存的是标签
imgX1, imgY1 = load_CIFAR_batch("C:/Users/zhxing/Desktop/python/data/cifar-10-batches-py/data_batch_1")
imgX2, imgY2 = load_CIFAR_batch("C:/Users/zhxing/Desktop/python/data/cifar-10-batches-py/data_batch_2")
imgX3, imgY3 = load_CIFAR_batch("C:/Users/zhxing/Desktop/python/data/cifar-10-batches-py/data_batch_3")
imgX4, imgY4 = load_CIFAR_batch("C:/Users/zhxing/Desktop/python/data/cifar-10-batches-py/data_batch_4")
imgX5, imgY5 = load_CIFAR_batch("C:/Users/zhxing/Desktop/python/data/cifar-10-batches-py/data_batch_5")
#5万个训练数据
Xte_rows, Yte = load_CIFAR_batch("C:/Users/zhxing/Desktop/python/data/cifar-10-batches-py/test_batch")
#一万个训练数据

Xtr_rows=np.concatenate((imgX1,imgX2,imgX3,imgX4,imgX5))
Ytr_rows=np.concatenate((imgY1,imgY2,imgY3,imgY4,imgY5))
#连接起来，这就是50000个训练数据及相应的标签

              #创建一个分类器对象

Xval_rows=Xtr_rows[:1000,:]       #交叉验证的1000个测试样本及标签
Yval_rows=Ytr_rows[:1000]  
Xtr_x_rows=Xtr_rows[1000:,:]      #交叉验证的49000个训练样本及标签
Ytr_x_rows=Ytr_rows[1000:]

for k in [1,3,5,10,20,50]:
    nn= K_NearestNeighbor()   
    nn.train(Xtr_x_rows[:3000,:],Ytr_x_rows[:3000])      #用前一千个来训练，也可用全部，跑的很慢，这个算法训练什么都没干，就是赋值

    Yte_predict = nn.predict(Xval_rows[:100,:],k)      # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
    print(k,':')
    print('accuracy: %f'  % (np.mean(Yte_predict == Yval_rows[:100])))   #相等我话算出来就是1，所以这样取均值算出来的就是正确率

# show a picture
#==============================================================================
# image=imgX1[6,0:1024].reshape(32,32)       #这个是把第六章图拿出来看了一下，得离得远一点才能看清
# print(image.shape)
# plt.imshow(image,cmap=plt.cm.gray)
# plt.axis('off')    #去除图片边上的坐标轴
# plt.show()
#==============================================================================
