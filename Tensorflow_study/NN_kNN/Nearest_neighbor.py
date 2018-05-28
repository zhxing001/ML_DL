# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 09:50:23 2017

@author: zhxing
"""
import pickle as p
import matplotlib.pyplot as plt
import numpy as np

class NearestNeighbor(object):
    def __init__(self):
        pass
    def train(self,X,y):
        self.Xtr=X
        self.ytr=y
    def predict(self,X):
        num_test=X.shape[0]
        #保证输出和输入类型相同
        Ypred=np.zeros(num_test,dtype=self.ytr.dtype)
        
        for i in range(num_test):
            #对第i章图片在训练集中找最相似的
            distance=np.sum(np.abs(self.Xtr-X[i,:]),axis=1)   #算差的绝对值之和，和Xtr里面的每一个都去做
            min_index=np.argmin(distance)     #得到最小的差值的索引
            Ypred[i]=self.ytr[min_index]     #label，得到对应的label
            
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

nn= NearestNeighbor()                 #创建一个分类器对象
nn.train(Xtr_rows[:1000,:],Ytr_rows[:1000])      #用前一千个来训练，也可用全部，跑的很慢，这个算法训练什么都没干，就是赋值

Yte_predict = nn.predict(Xte_rows[:1000,:])      # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print('accuracy: %f' % (np.mean(Yte_predict == Yte[:1000])))   #相等我话算出来就是1，所以这样取均值算出来的就是正确率

# show a picture
image=imgX1[6,0:1024].reshape(32,32)       #这个是把第六章图拿出来看了一下，得离得远一点才能看清
print(image.shape)
plt.imshow(image,cmap=plt.cm.gray)
plt.axis('off')    #去除图片边上的坐标轴
plt.show()