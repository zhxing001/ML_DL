# -*- coding: utf-8 -*-
"""

Lrn_test
Created on Thu Apr 26 16:19:52 2018

@author: zhxing
"""

import tensorflow as tf

a=tf.constant([
        [[1.0,2,3,4],
         [5,6,7,8],
         [8,7,6,5],
         [4,3,2,1]],
         
         [[4.0,3,2,1],
          [8,7,6,5],
          [1,2,3,4],
          [5,6,7,8]]
        ])

a = tf.reshape(a, [1, 2, 2, 8]) 
normal_a=tf.nn.lrn(a,2,0,1,1)    
with tf.Session() as sess:    
    print("feature map:")    
    image = sess.run(a)    
    print (image)    
    print("normalized feature map:")    
    normal = sess.run(normal_a)    
    print (normal)   
    
    
'''
分析如下：
由调用关系得出 n/2=2，k=0，α=1，β=1，N=8

第一行第一个数来说：i = 0

a = 1，min(N-1, i+n/2) = min(7, 2)=2，j = max(0, i - k)=max(0, 0)=0，下标从0~2个数平方求和， b=1/(1^2 + 2^2 + 3^2)=1/14 = 0.071428571

同理，第一行第四个数来说：i = 3

a = 4，min(N-1, i+n/2) = min(7, 5 )=5, j = max(0,1) = 1，下标从1~5进行平方求和，b = 4/(2^2 + 3^2 + 4^2 + 5^2 + 6^2) = 4/90=0.044444444

再来一个，第二行第一个数来说： i = 0

a = 8, min(N-1, i+n/2) = min(7, 2) = 2, j=max(0,0)=0, 下标从0~2的3个数平方求和，b = 8/(8^2 + 7^2 + 6^2)=8/149=0.053691275
'''